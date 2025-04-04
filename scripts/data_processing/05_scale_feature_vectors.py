import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure the scripts directory is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    load_config,
    load_json,
    ensure_directory_exists,
    get_channel_files,
    save_json
)

"""0-3 = "convnextv2_shot_similarity.pkl": [prev_2,prev_1,next_1,next_2],
        4 = "videoshot_scalemovement.pkl": [ 0.625],   # 0 is extreme closeup, 1 is long shot
        5 = "whisperx_sentiment.pkl": [ 0.375], #neutral= 0.5
        6-9 = "ssv2-vmae_action_shot_similarity.pkl": [prev_2, prev_1, next_1, next_2],
        10 = "llm_evaluative.pkl": [0.75],  # 0 is not evaluative, 1 is evaluative
        11= "shot_density.pkl": [ 0.4651540889451104 ], # 0 is low density, 1 is high density
        12-23 = "speaker_turns_meta.pkl": [active_speaker?, unique_speakers, anchor, reporter, expert, layperson, elite, interview, talkin-head, speech, commenting, voice-over],
        24,25 ="whisperxspeaker_segmentClf.pkl": [malespeaker_present?, femalespeaker_present?],
        26-29 = "face_analysis.pkl": [ average_facesize, unique_faces, regiuon:upper/lower, region:left/right],
        30-36 = "face_emotions_deepface.pkl": [ 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        37-45 = "whisperxspeaker_audioclf.pkl": ['Speech', 'Narration', 'Music', 'Animal', 'Vehicle', 'Siren', 'Other_Sound', 'Silence', 'Artillery_fire'],
        46-51  = "whisperx_ner.pkl": [ 1, 2,3,4,5,6 ],
        52-55 = "kinetics-xclip_action_shot_similarity.pkl": [ prev_2, prev_1, next_1, next_2],
        56-59 = "siglip_shot_similarity.pkl": [ prev_2, prev_1, next_1, next_2],
        60-63 ="places_shot_similarity.pkl": [ prev_2, prev_1, next_1, next_2],
        64-67 = "kinetics-vmae_action_shot_similarity.pkl": [prev_2, prev_1, next_1, next_2]
        68-83 = "whisperx_pos.pkl": [ 1, 2... -16],"""

config = load_config()
feature_vec_dir = os.path.join(config["base_path"], config["paths"]["final"]["feature_vectors_unscaled"])  
channel_files = get_channel_files(feature_vec_dir)


global_instance_id = 0
all_rows = []
 
for channel, files in channel_files.items():
    if channel == ("Test"):
        continue
 
    for file_path in files:
        # The news story name can be extracted from the filename
        news_story_name = os.path.splitext(os.path.basename(file_path))[0]
        data = load_json(file_path, printing=False)
       
        
        for shot_idx, shot_info in data.items():
            
            emotions =  shot_info.get("face_emotions_deepface.pkl", [])
            speaker_genders = shot_info.get("whisperxspeaker_segmentClf.pkl", [])
           
            row = {
                "conv_prev2": shot_info.get("convnextv2_shot_similarity.pkl", [])[0],
                "conv_prev1": shot_info.get("convnextv2_shot_similarity.pkl", [])[1],
                "conv_next1": shot_info.get("convnextv2_shot_similarity.pkl", [])[2],
                "conv_next2": shot_info.get("convnextv2_shot_similarity.pkl", [])[3],
                "videoshot_scalemovement": shot_info.get("videoshot_scalemovement.pkl", [])[0],
                "whisperx_sentiment": shot_info.get("whisperx_sentiment.pkl", [])[0],
                "ssv2_act_prev2": shot_info.get("ssv2-vmae_action_shot_similarity.pkl", [])[0],
                "ssv2_act_prev1": shot_info.get("ssv2-vmae_action_shot_similarity.pkl", [])[1],
                "ssv2_act_next1": shot_info.get("ssv2-vmae_action_shot_similarity.pkl", [])[2],
                "ssv2_act_next2": shot_info.get("ssv2-vmae_action_shot_similarity.pkl", [])[3],
                "llm_evaluative": shot_info.get("llm_evaluative.pkl", [])[0],
                "shot_density": shot_info.get("shot_density.pkl", [])[0],
                "active_speaker": shot_info.get("speaker_turns_meta.pkl", [])[0],
                "unique_speakers": shot_info.get("speaker_turns_meta.pkl", [])[1],
                "anchor": shot_info.get("speaker_turns_meta.pkl", [])[2],
                "reporter": shot_info.get("speaker_turns_meta.pkl", [])[3],
                "expert": shot_info.get("speaker_turns_meta.pkl", [])[4],
                "layperson": shot_info.get("speaker_turns_meta.pkl", [])[5],
                "elite": shot_info.get("speaker_turns_meta.pkl", [])[6],
                "interview": shot_info.get("speaker_turns_meta.pkl", [])[7],
                "talkin-head": shot_info.get("speaker_turns_meta.pkl", [])[8],
                "speech": shot_info.get("speaker_turns_meta.pkl", [])[9],
                "commenting": shot_info.get("speaker_turns_meta.pkl", [])[10],
                "voice-over": shot_info.get("speaker_turns_meta.pkl", [])[11],
                "male_speaker_present":  speaker_genders[0],
                "female_speaker_present": speaker_genders[1],
                "average_facesize": shot_info.get("face_analysis.pkl", [])[0],
                "unique_faces": shot_info.get("face_analysis.pkl", [])[1],
                "face_region_y": shot_info.get("face_analysis.pkl", [])[2],
                "face_region_x": shot_info.get("face_analysis.pkl", [])[3],
                "angry": emotions[0],
                "disgust": emotions[1],
                "fear": emotions[2],
                "happy": emotions[3],
                "sad": emotions[4],
                "surprise": emotions[5],
                "neutral": emotions[6],
                "whisper_speech": shot_info.get("whisperxspeaker_audioclf.pkl", [])[0],
                "whisper_narration": shot_info.get("whisperxspeaker_audioclf.pkl", [])[1],
                "whisper_music": shot_info.get("whisperxspeaker_audioclf.pkl", [])[2],
                "whisper_animal": shot_info.get("whisperxspeaker_audioclf.pkl", [])[3],
                "whisper_vehicle": shot_info.get("whisperxspeaker_audioclf.pkl", [])[4],
                "whisper_siren": shot_info.get("whisperxspeaker_audioclf.pkl", [])[5],
                "whisper_other_sound": shot_info.get("whisperxspeaker_audioclf.pkl", [])[6],
                "whisper_silence": shot_info.get("whisperxspeaker_audioclf.pkl", [])[7],
                "whisper_artillery_fire": shot_info.get("whisperxspeaker_audioclf.pkl", [])[8],
                "whisperx_ner_1": shot_info.get("whisperx_ner.pkl", [])[0],
                "whisperx_ner_2": shot_info.get("whisperx_ner.pkl", [])[1],
                "whisperx_ner_3": shot_info.get("whisperx_ner.pkl", [])[2],
                "whisperx_ner_4": shot_info.get("whisperx_ner.pkl", [])[3],
                "whisperx_ner_5": shot_info.get("whisperx_ner.pkl", [])[4],
                "whisperx_ner_6": shot_info.get("whisperx_ner.pkl", [])[5],
                "kinX_act_prev2": shot_info.get("kinetics-xclip_action_shot_similarity.pkl", [])[0],
                "kinX_act_prev1": shot_info.get("kinetics-xclip_action_shot_similarity.pkl", [])[1],
                "kinX_act_next1": shot_info.get("kinetics-xclip_action_shot_similarity.pkl", [])[2],
                "kinX_act_next2": shot_info.get("kinetics-xclip_action_shot_similarity.pkl", [])[3],
                "sig_prev2": shot_info.get("siglip_shot_similarity.pkl", [])[0],
                "sig_prev1": shot_info.get("siglip_shot_similarity.pkl", [])[1],
                "sig_next1": shot_info.get("siglip_shot_similarity.pkl", [])[2],
                "sig_next2": shot_info.get("siglip_shot_similarity.pkl", [])[3],
                "places_prev2": shot_info.get("places_shot_similarity.pkl", [])[0],
                "places_prev1": shot_info.get("places_shot_similarity.pkl", [])[1],
                "places_next1": shot_info.get("places_shot_similarity.pkl", [])[2],
                "places_next2": shot_info.get("places_shot_similarity.pkl", [])[3],
                "kinV_act_prev2": shot_info.get("kinetics-vmae_action_shot_similarity.pkl", [])[0],
                "kinV_act_prev1": shot_info.get("kinetics-vmae_action_shot_similarity.pkl", [])[1],
                "kinV_act_next1": shot_info.get("kinetics-vmae_action_shot_similarity.pkl", [])[2],
                "kinV_act_next2": shot_info.get("kinetics-vmae_action_shot_similarity.pkl", [])[3],
                "whisperx_pos_1": shot_info.get("whisperx_pos.pkl", [])[0],
                "whisperx_pos_2": shot_info.get("whisperx_pos.pkl", [])[1],
                "whisperx_pos_3": shot_info.get("whisperx_pos.pkl", [])[2],
                "whisperx_pos_4": shot_info.get("whisperx_pos.pkl", [])[3],
                "whisperx_pos_5": shot_info.get("whisperx_pos.pkl", [])[4],
                "whisperx_pos_6": shot_info.get("whisperx_pos.pkl", [])[5],
                "whisperx_pos_7": shot_info.get("whisperx_pos.pkl", [])[6],
                "whisperx_pos_8": shot_info.get("whisperx_pos.pkl", [])[7],
                "whisperx_pos_9": shot_info.get("whisperx_pos.pkl", [])[8],
                "whisperx_pos_10": shot_info.get("whisperx_pos.pkl", [])[9],
                "whisperx_pos_11": shot_info.get("whisperx_pos.pkl", [])[10],
                "whisperx_pos_12": shot_info.get("whisperx_pos.pkl", [])[11],
                "whisperx_pos_13": shot_info.get("whisperx_pos.pkl", [])[12],
                "whisperx_pos_14": shot_info.get("whisperx_pos.pkl", [])[13],
            }

            all_rows.append(row)


# Finally, convert the list of dictionaries into a DataFrame:
df = pd.DataFrame(all_rows)

print(df.shape)
print(df.head())

if df.shape[1] != 82:
    print("Error: The number of columns is not 84")
    


position_min_max_std_dict = {}
for i in range(0, 82):
    #column_name = df.columns[i]
    #print(column_name)
    min_val = df.iloc[:,i].min()
    max_val = df.iloc[:,i].max()
    std_val = df.iloc[:,i].std()
    position_min_max_std_dict[i] = [min_val, max_val, std_val]


def scale(x, min_val, max_val):
    if min_val == max_val:
        return x
    
    x = (x - min_val) / (max_val - min_val)     
    #todo scaling incorrect
    return float(x)

output_base_path = os.path.join(config["base_path"], config["paths"]["final"]["feature_vectors"])
for channel, files in channel_files.items():
    if channel == ("Test"):
        continue
    
    for file_path in files:
        # The news story name can be extracted from the filename
        news_story_name = os.path.splitext(os.path.basename(file_path))[0]
        data = load_json(file_path, printing=False)
       
        
        for shot_idx, shot_info in data.items():

            conv_values = shot_info.get("convnextv2_shot_similarity.pkl", [])
            for i in range(0, 4):
                shot_info["convnextv2_shot_similarity.pkl"][i] = scale(conv_values[i], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            shot_info["videoshot_scalemovement.pkl"][0] = scale(shot_info.get("videoshot_scalemovement.pkl", [])[0], position_min_max_std_dict[4][0], position_min_max_std_dict[4][1])
            shot_info["whisperx_sentiment.pkl"][0] = scale(shot_info.get("whisperx_sentiment.pkl", [])[0], position_min_max_std_dict[5][0], position_min_max_std_dict[5][1])
            ssv2_act_values = shot_info.get("ssv2-vmae_action_shot_similarity.pkl", [])
            for i in range(6, 10):
                shot_info["ssv2-vmae_action_shot_similarity.pkl"][i-6] = scale(ssv2_act_values[i-6], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            shot_info["llm_evaluative.pkl"][0] = scale(shot_info.get("llm_evaluative.pkl", [])[0], position_min_max_std_dict[10][0], position_min_max_std_dict[10][1])
            shot_info["shot_density.pkl"][0] = scale(shot_info.get("shot_density.pkl", [])[0], position_min_max_std_dict[11][0], position_min_max_std_dict[11][1])
            speaker_turns = shot_info.get("speaker_turns_meta.pkl", [])
            for i in range(12, 24):
                shot_info["speaker_turns_meta.pkl"][i-12] = scale(speaker_turns[i-12], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            speaker_gender_values = shot_info.get("whisperxspeaker_segmentClf.pkl", [])
            for i in range(24, 26):
                shot_info["whisperxspeaker_segmentClf.pkl"][i-24] = scale(speaker_gender_values[i-24], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            face_analysis_values = shot_info.get("face_analysis.pkl", [])
            for i in range(26, 30):
                shot_info["face_analysis.pkl"][i-26] = scale(face_analysis_values[i-26], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            emotions =  shot_info.get("face_emotions_deepface.pkl", [])
            for i in range(30, 37):
                shot_info["face_emotions_deepface.pkl"][i-30] = scale(emotions[i-30], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            audio_values = shot_info.get("whisperxspeaker_audioclf.pkl", [])
            for i in range(37, 46):
                shot_info["whisperxspeaker_audioclf.pkl"][i-37] = scale(audio_values[i-37], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            ner_values = shot_info.get("whisperx_ner.pkl", [])
            for i in range(46, 52):
                shot_info["whisperx_ner.pkl"][i-46] = scale(ner_values[i-46], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            kinX_act_values = shot_info.get("kinetics-xclip_action_shot_similarity.pkl", [])
            for i in range(52, 56):
                shot_info["kinetics-xclip_action_shot_similarity.pkl"][i-52] = scale(kinX_act_values[i-52], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            sig_values = shot_info.get("siglip_shot_similarity.pkl", [])
            for i in range(56, 60):
                shot_info["siglip_shot_similarity.pkl"][i-56] = scale(sig_values[i-56], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            places_values = shot_info.get("places_shot_similarity.pkl", [])
            for i in range(60, 64):
                shot_info["places_shot_similarity.pkl"][i-60] = scale(places_values[i-60], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            kinV_act_values = shot_info.get("kinetics-vmae_action_shot_similarity.pkl", [])
            for i in range(64, 68):
                shot_info["kinetics-vmae_action_shot_similarity.pkl"][i-64] = scale(kinV_act_values[i-64], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])
            pos_values = shot_info.get("whisperx_pos.pkl", [])
            for i in range(68, 82):
                shot_info["whisperx_pos.pkl"][i-68] = scale(pos_values[i-68], position_min_max_std_dict[i][0], position_min_max_std_dict[i][1])


               
        output_file_path = os.path.join(output_base_path, channel, f"{news_story_name}.json")
        ensure_directory_exists( os.path.join(output_base_path, channel))
        save_json( data, output_file_path)
       