
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pickle
import base64

# Add the `scripts` folder (one directory up) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_config, load_pickle, save_json, ensure_directory_exists, log_message

def image_to_base64(image_path):
    """
    Convert the image at 'image_path' to a base64-encoded string.
    Returns None if there's an error (e.g., file not found).
    """
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        log_message(f"Error encoding image {image_path}: {e}")
        return None

def get_text(shot, transcript):
    """
    Return the concatenated text from transcript entries that overlap with 'shot'.
    Each transcript entry is expected to have 'start', 'end', and 'text' keys.
    """
    return " ".join(
        turn['text']
        for turn in transcript
        if turn['start'] < shot['end'] and turn['end'] > shot['start']
    )

def process_video(target_json_path, shot_frames_dir, transcript_base_dir, output_base_dir):
    """
    For each annotated story in the JSON file at 'target_json_path':
      1. Load the corresponding frames from 'shot_frames_dir'.
      2. Convert each shot image to base64 and store it (instead of a file path).
      3. Retrieve both shot_text and speakerturn_text from the transcript.
      4. Write out a JSON file that Qwen2.5-VL-7B can consume,
         containing both the images (in base64) and the text per shot.
    """
    with open(target_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    video_filename = data['video_fn']
    shots = data['shots'][0]['segments']  # The shot segments
    channel = os.path.basename(os.path.dirname(target_json_path))
    video_id = os.path.splitext(video_filename)[0]

    # Transcript is assumed to be in 'asr_whisperx.pkl'
    transcript_path = os.path.join(transcript_base_dir, channel, video_filename, 'asr_whisperx.pkl')
    if not os.path.exists(transcript_path):
        log_message(f"Transcript not found: {transcript_path}")
        return
    transcript_data = load_pickle(transcript_path)

    # We use 'segments' for shot_text and 'speaker_turns' for speakerturn_text
    segments = transcript_data['output_data']['segments']
    speakerturns = transcript_data['output_data']['speaker_turns']

    shot_frames_video_dir = os.path.join(shot_frames_dir, channel, video_id)
    output_video_dir = os.path.join(output_base_dir, channel)
    ensure_directory_exists(output_video_dir)

    # Go through each annotated story in the JSON
    for story in data.get('story', []):
        story_name = story["name"]
        log_message(f"Processing video {story_name} from {video_filename}")
        

        # Each story has a 'segments' list with at least one item containing start/end
        if not story.get('segments') or len(story['segments']) < 1:
            continue
        story_start = story['segments'][0]['start']
        story_end = story['segments'][0]['end']

        # Collect shot data in a list (preserving order)
        shots_list = []
        for shot in shots:
            # Only process shots that overlap this story's time window
            if not (shot['end'] > story_start and shot['start'] < story_end):
                continue

            shot_label = shot['label']
            shot_image_path = os.path.join(shot_frames_video_dir, f"shot_{shot_label}.jpg")
            if not os.path.exists(shot_image_path):
                log_message(f"Missing shot frame: {shot_image_path}")
                continue

            # Convert the image to base64
            image_b64 = image_to_base64(shot_image_path)
            if not image_b64:
                continue

            # Retrieve text from transcript segments and speaker turns
            shot_text = get_text(shot, segments)
            speakerturn_text = get_text(shot, speakerturns)

            shot_data = {
                "shot_label": shot_label,
                "shot_image_b64": image_b64,  # embedded image data
                "shot_text": shot_text,
                "speakerturn_text": speakerturn_text
            }
            shots_list.append(shot_data)

        # Build a final JSON structure for this story
        story_id = story_name.split(":")[0]  # Adjust if needed
        output_data = {
            "video_id": video_filename,  # Use the video filename without extension
            "story_id": story_id,
            "shots": shots_list
        }

        # Save the final JSON to the output directory
        output_file_path = os.path.join(output_video_dir, f"{video_filename}_{story_id}.json") #use filename instead?
        save_json(output_data, output_file_path)
        log_message(f"Saved VLM data for story '{story_name}' to {output_file_path}")

def main():
    """Main entry point: read config, loop over channels, run process_video on each JSON."""
    config = load_config()
    converted_targets_dir = config['paths']['processing']['converted_targets']
    shot_frames_dir = config['paths']['processing']['shot_frames']
    transcript_base_dir = config['paths']['raw']['feature_pickles']
    vlm_data_dir = config['paths']['final']['vlm_data']

    for channel in os.listdir(converted_targets_dir):
        if channel == "Test":
            continue
        channel_dir = os.path.join(converted_targets_dir, channel)
        if os.path.isdir(channel_dir):
            log_message(f"Processing channel {channel}")
            for json_file in os.listdir(channel_dir):
                if json_file.endswith(".json"):
                    json_path = os.path.join(channel_dir, json_file)
                    process_video(json_path, shot_frames_dir, transcript_base_dir, vlm_data_dir)

if __name__ == "__main__":
    main()
