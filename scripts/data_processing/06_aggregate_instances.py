import os
import json
import numpy as np
import sys

# Add the `scripts` folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    load_config,
    ensure_directory_exists,
    aggregate_features,
    save_json,
    log_message,
    get_channel_files,
    load_json
)


FEATURE_AGGREGATION_METHODS = {
    "strategies": "max",
    "feps": "max",
    "whisperx_ner.pkl": "mean",
    "whisperx_sentiment.pkl": "mean",
    "whisperx_pos.pkl": "mean",
    "llm_evaluative.pkl": "mean",
    "face_emotions_deepface.pkl": "mean",
    "videoshot_scalemovement.pkl": "mean",
    "whisperxspeaker_audioclf.pkl": "mean",       
    "whisperxspeaker_segmentClf.pkl": "max",        
    "clip_qas.pkl": "concat",                   
    "shot_density.pkl": "mean",
    "siglip_shot_similarity.pkl": "mean",
    "convnextv2_shot_similarity.pkl": "mean",
    "places_shot_similarity.pkl": "mean",
    "kinetics-vmae_action_shot_similarity.pkl": "mean",
    "ssv2-vmae_action_shot_similarity.pkl": "mean",
    "kinetics-xclip_action_shot_similarity.pkl": "mean",
    "wav2vec2_audio_shot_similarity.pkl": "mean",
    "beats_audio_shot_similarity.pkl": "mean",
    "whisper_audio_shot_similarity.pkl": "mean",
    "face_analysis.pkl": "mean",
    "speaker_turns_meta.pkl": "mean",
}

def safe_get(shot, feature_name):
    """
    Returns the feature array/list from a shot if it exists, otherwise returns
    an empty list (or your desired default).
    """
    return shot.get(feature_name, [])

def custom_aggregate(feature_lists, method):
    """
    Wraps your existing `aggregate_features` function for numeric arrays/lists,
    and does custom handling for non-numeric or special features.
    """
    if method == "concat":
        return [item for sublist in feature_lists for item in sublist]

    return aggregate_features(feature_lists, method=method)

def aggregate_shots_to_instances(input_base_path, output_base_path, window_size, default_method='mean'):
    """
    Aggregates shots into sliding window instances and saves them divided by news source.

    Parameters:
        input_base_path (str): Path to the 'FeatureVectors' directory containing source folders.
        output_base_path (str): Path to the output 'RFDataset' directory.
        default_method (str): Default method to aggregate features if not in FEATURE_AGGREGATION_METHODS.
        window_size (int): Number of shots to aggregate into one instance.
    """
    ensure_directory_exists(output_base_path)

    channel_files = {
        channel: files
        for channel, files in get_channel_files(input_base_path).items()
        if files
    }

    for channel, files in channel_files.items():
        log_message(f"Processing channel: {channel}")
        if channel == ("Test"):
            continue
        channel_output_dir = os.path.join(output_base_path, channel)
        ensure_directory_exists(channel_output_dir)

        for file_path in files:
            data = load_json(file_path)
            instances = []

            # The union of keys across *all* shots might be safer if different shots have different keys
            all_feature_names = set()
            for shot in data.values():
                all_feature_names.update(shot.keys())
                break
            all_feature_names.discard("start")
            all_feature_names.discard("end")

            shots = list(data.values())
            # case: if there are less shots than the window size aggregate all shots into one instance
            if len(shots) < window_size:
                aggregated_instance = {}
                aggregated_instance["start"] = shots[0]["start"]
                aggregated_instance["end"] = shots[-1]["end"]

                for feature_name in all_feature_names:
                    method = FEATURE_AGGREGATION_METHODS.get(feature_name, default_method)
                    aggregated_value = custom_aggregate(
                        [safe_get(shot, feature_name) for shot in shots],
                        method
                    )
                    aggregated_instance[feature_name] = aggregated_value

                instances.append(aggregated_instance)
            else:    
                for i in range(len(shots) - window_size + 1):
                    aggregated_instance = {}
                    aggregated_instance["start"] = shots[i]["start"]
                    aggregated_instance["end"] = shots[i + window_size - 1]["end"]

                    for feature_name in all_feature_names:
                        method = FEATURE_AGGREGATION_METHODS.get(feature_name, default_method)
                        aggregated_value = custom_aggregate(
                            [safe_get(shots[j], feature_name) for j in range(i, i + window_size)],
                            method
                        )
                        aggregated_instance[feature_name] = aggregated_value

                    instances.append(aggregated_instance)

            output_file_path = os.path.join(channel_output_dir, os.path.basename(file_path))
            save_json(instances, output_file_path)

def main():
    # Load configuration
    config = load_config()

    # Paths from config
    feature_vectors_dir = os.path.join(config['base_path'], config['paths']['final']['feature_vectors'])
    rf_data_dir = os.path.join(config['base_path'], config['paths']['final']['rf_dataset'])
    window_size = config['dataset']['window_size']

    # Aggregate shots into instances
    aggregate_shots_to_instances(feature_vectors_dir, rf_data_dir, window_size, default_method='mean')


if __name__ == "__main__":
    main()
