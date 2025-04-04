#!/usr/bin/env python3
import sys
import os
# Ensure the scripts directory is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import (
    load_config, load_json, log_message, resize_image, save_json, parse_output
)
from prompts import (
    FILM_EDITING_PROMPT, FILM_EDITING_DEFS_PROMPT, CUT_AWAY, CUT_IN, INTENSIFY, ALTERNATING_SHOT, SHOT_REVERSE_SHOT, EMOTIONALIZATION, STRATEGIES_PROMPT, STRATEGIES_DEFS_PROMPT,
    FRAGMENTATION, INDIVIDUALIZATION_OF_ELITE, INDIVIDUALIZATION_OF_LAYPERSON, INDIVIDUALIZATION_OF_REPORTER, FRAGMENTATION_SPLITSCREEN)


def load_model(model_name):
    """Loads the Qwen2.5-VL model with optimized memory usage."""
    print(f"Loading model: {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Reduce memory usage with FP16
        device_map="auto",         # Auto-distribute model layers across CPU/GPU
        offload_folder="offload"   # Offload layers to disk if needed
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    model.gradient_checkpointing_enable()  # Enable memory optimization
    return model, processor


def fill_prompt(shot_window, prompt_text):
    """
    Generates a structured prompt for the Qwen 2.5 model.
    The system message contains the film-editing instructions.
    The user message contains up to 3 images plus transcript text.
    """

    # 1) Build a system message with editing instructions
    system_message = {
        "role": "system",
        "content": prompt_text
    }

    prev_prev_shot = shot_window["prev_prev_shot"]
    prev_shot = shot_window["prev_shot"]
    cur_shot = shot_window["cur_shot"]
    next_shot = shot_window["next_shot"]
    next_next_shot = shot_window["next_next_shot"]

    # 2) Prepare user content: images (if available) + textual transcripts
    prev_prev_text = f"Previous-Previous Shot Transcript:\n{prev_prev_shot['speakerturn_text']}" if prev_prev_shot else "No previous-previous shot"
    prev_text = f"Previous Shot Transcript:\n{prev_shot['speakerturn_text']}" if prev_shot else "No previous shot"
    cur_text = f"Current Shot Transcript:\n{cur_shot['speakerturn_text']}"
    next_text = f"Next Shot Transcript:\n{next_shot['speakerturn_text']}" if next_shot else "No next shot"
    next_next_text = f"Next-Next Shot Transcript:\n{next_next_shot['speakerturn_text']}" if next_next_shot else "No next-next shot"

    user_content = []

    # Previous shot image (if it exists)
    if prev_prev_shot and 'shot_image_b64' in prev_prev_shot:
        user_content.append({
            "type": "image",
            "image": f"data:image;base64,{prev_prev_shot['shot_image_b64']}"
        })
    else:
        user_content.append({"type": "text", "text": "Previous-Previous shot image unavailable."})
    if prev_shot and 'shot_image_b64' in prev_shot:
        user_content.append({
            "type": "image",
            "image": f"data:image;base64,{prev_shot['shot_image_b64']}"
        })
    else:
        user_content.append({"type": "text", "text": "Previous shot image unavailable."})

    # Current shot image
    if cur_shot and 'shot_image_b64' in cur_shot:
        user_content.append({
            "type": "image",
            "image": f"data:image;base64,{cur_shot['shot_image_b64']}"
        })
    else:
        user_content.append({"type": "text", "text": "Current shot image unavailable."})

    # Next shot image (if it exists)
    if next_shot and 'shot_image_b64' in next_shot:
        user_content.append({
            "type": "image",
            "image": f"data:image;base64,{next_shot['shot_image_b64']}"
        })
    else:
        user_content.append({"type": "text", "text": "Next shot image unavailable."})

    # Next-next shot image (if it exists)
    if next_next_shot and 'shot_image_b64' in next_next_shot:
        user_content.append({
            "type": "image",
            "image": f"data:image;base64,{next_next_shot['shot_image_b64']}"
        })
    else:
        user_content.append({"type": "text", "text": "Next-next shot image unavailable."})

    # Add transcripts
    user_content.append({
        "type": "text",
        "text": f"{prev_prev_text}\n\n{prev_text}\n\n{cur_text}\n\n{next_text}\n\n{next_next_text}"
    })

    # 3) Combine into a single prompt
    prompt = [
        system_message,  # instructions + rules
        {
            "role": "user",
            "content": user_content
        }
    ]

    return prompt

def process_input( model, processor, shot_window, instruction_prompt):
    """
    Processes a single input by generating model predictions.
    """
    # Format text input
    prompt = fill_prompt(shot_window, instruction_prompt)

    # Format text input
    text_input = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    # Extract image and video inputs
    image_inputs, video_inputs = process_vision_info(prompt) 

    # Resize images before processing
    image_inputs = [resize_image(img) for img in image_inputs if img is not None]

    # Prepare model inputs
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate model output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    # Trim the generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    prediction = output_text[0] if output_text else "No output"
    return prediction
    
def process_story(story_data, model, processor, output_dir, instruction_prompts, strategies_bool):
    """
    Processes a single story by iterating through shots in a sliding window of 3, 
    including images from all three shots, and storing model predictions.
    """
    shots = story_data.get("shots", [])
    
    story_id = story_data.get("story_id", "unknown")
    video_id = story_data.get("video_id", "unknown")
    out_filename = f"{video_id}_{story_id}_predictions.json"

    # if the output file already exists, skip processing
    if os.path.exists( os.path.join(output_dir, out_filename)):
        log_message(f"Skipping story: {video_id}, {story_id} as predictions already exist.")
        return # IMPORTANT TO DELETE OLD FILES, but processing is expensive so we avoid doing the same work multiple times

    log_message(f"Processing story: {video_id}, {story_id} with {len(shots)} shots.")


    # Iterate through shots in a sliding window of 3
    for i in range(len(shots)):
        print(f"\n=== Processing Shot {i} ===")

        # Define previous, current, and next shot
        prev_prev_shot = shots[i - 2] if i > 1 else None
        prev_shot = shots[i - 1] if i > 0 else None
        cur_shot = shots[i]
        next_shot = shots[i + 1] if i < len(shots) - 1 else None
        next_next_shot = shots[i + 2] if i < len(shots) - 2 else None

        shot_window = {
            "prev_prev_shot": prev_prev_shot,
            "prev_shot": prev_shot,
            "cur_shot": cur_shot,
            "next_shot": next_shot,
            "next_next_shot": next_next_shot
        }
        if len(instruction_prompts) == 1: 
            instruction_prompt = instruction_prompts[0] # This is multilabel setting
            prediction = process_input(model, processor, shot_window, instruction_prompt)
            prediction = parse_output(prediction)
        
        else:
            predictions = []
            for prompt in instruction_prompts:
                prompt_name = prompt[0]
                prompt_text = prompt[1]
                prediction = process_input(model, processor, shot_window, prompt_text)
                print(prompt_name, prediction)
                if prediction == "1":
                    predictions.append(prompt_name)
            prediction = predictions
        if strategies_bool:
            target_order = {
                "fragmentation": 0,
                "individualization_of_elite": 1,
                "individualization_of_reporter": 2,
                "fragmentation_splitscreen": 3,
                "emotionalization": 4,
                "individualization_of_layperson": 5,
                "individualization-of-reporter": 2, # had an error because the vlm once predicted this
                "individualization-of-layperson": 5,
                "fragmentation-splitscreen": 3,
                "individualization-of-elite": 1
            }
            prediction_vector = [0, 0, 0, 0, 0, 0]
            for pred in prediction:
                prediction_vector[target_order[pred]] = 1
        else:
            target_order = {
                "alternating-shot": 0,
                "shot-reverse-shot": 1,
                "cut-in": 2,
                "intensify": 3,
                "cut-away": 4
            }
            prediction_vector = [0, 0, 0, 0, 0]
            for pred in prediction:
                prediction_vector[target_order[pred]] = 1
        shots[i]["prediction"] = prediction_vector

        print("Model Output:", prediction)
        print("Model Output Vector:", prediction_vector)

    minimal_output = {
        "video_id": video_id,
        "story_id": story_id,
        "shots": []
    }
    for shot in shots:
        shot_label = shot.get("shot_label", "unknown")  
        prediction = shot.get("prediction", "No output")
        minimal_output["shots"].append({
            "shot_label": shot_label,
            "prediction": prediction
        })

    # Save the minimal output if desired
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, out_filename)

        
        save_json(minimal_output, out_path)
        log_message(f"Predictions saved to {out_path}")

def solve_task(instruction_prompts, task_name, strategies_bool, model, processor):
    config = load_config()
    output_path = os.path.join(config['base_path'], config['paths']['model_ouputs']['vlm'])
    output_path = os.path.join(output_path, task_name) 
    os.makedirs(output_path, exist_ok=True)
    data_base_path = os.path.join(config['base_path'], config['paths']['final']['vlm_data'])
    # Process each channel
    channels = ["Test", "BildTV", "CompactTV", "HeuteJournal", "Tagesschau", "Welt"]
    for split in os.listdir(data_base_path): 
        torch.cuda.empty_cache()
        if split in channels:
            continue  # Only process test data 
        if strategies_bool and split == "test_kfold_splits_fep":
            continue
        if not strategies_bool and split == "test_kfold_splits_strategie":
            continue
        channel_dir = os.path.join(data_base_path, split) 
        if os.path.isdir(channel_dir):
            log_message(f"Processing split: {split}")

            for json_file in os.listdir(channel_dir):
                if json_file.endswith(".json"):
                    story_path = os.path.join(channel_dir, json_file)
                    story_data = load_json(story_path)


                    # Now we pass 'output_dir' so each story’s results are written out
                    process_story(
                        story_data=story_data,
                        model=model,
                        processor=processor,
                        output_dir=output_path,
                        instruction_prompts=instruction_prompts,
                        strategies_bool=strategies_bool
                    )



if __name__ == "__main__":

    config = load_config()
    model_name = config['models']['vlm_name']
    # Free up CUDA memory
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load model and processor
    model, processor = load_model(model_name=model_name)

    #FEP Tasks
    #instruction_prompts = [FILM_EDITING_PROMPT] 
    #solve_task(instruction_prompts, task_name="film_editing", strategies_bool=False, model=model, processor=processor)

    #instruction_prompts = [FILM_EDITING_DEFS_PROMPT] 
    #solve_task(instruction_prompts, task_name="film_editing_defs", strategies_bool=False, model=model, processor=processor)

    
    instruction_prompts_perclass = [
        ("alternating-shot", ALTERNATING_SHOT),
        ("cut-away", CUT_AWAY),
        ("cut-in", CUT_IN),
        ("intensify", INTENSIFY),
        ("shot-reverse-shot", SHOT_REVERSE_SHOT)
        ]
    #solve_task(instruction_prompts_perclass, task_name="film_editing_perclass", strategies_bool=False, model=model, processor=processor)

    #Strategies Tasks
    #instruction_prompts = [STRATEGIES_PROMPT] 
    #solve_task(instruction_prompts, task_name="strategies", strategies_bool=True, model=model, processor=processor)

    #instruction_prompts = [STRATEGIES_DEFS_PROMPT] 
    #solve_task(instruction_prompts, task_name="strategies_defs", strategies_bool=True, model=model, processor=processor)

    instruction_prompts_perclass_strategies = [ 
        ("fragmentation", FRAGMENTATION),
        ("individualization_of_elite", INDIVIDUALIZATION_OF_ELITE),
        ("individualization_of_layperson", INDIVIDUALIZATION_OF_LAYPERSON),
        ("individualization_of_reporter", INDIVIDUALIZATION_OF_REPORTER),
        ("fragmentation_splitscreen", FRAGMENTATION_SPLITSCREEN),
        ("emotionalization", EMOTIONALIZATION)
        ]
    solve_task(instruction_prompts_perclass_strategies, task_name="strategies_perclass", strategies_bool=True, model=model, processor=processor)
    


    
