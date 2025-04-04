import sys
import os

# Add the `scripts` folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from collections import Counter
from utils import (
    load_config,
    load_json,
    ensure_directory_exists,
    multi_hot_encode,
    log_message,
    get_channel_files,
    load_feature_names
)



def load_stats(config):
    """
    Load statistics from the configuration file.
    Since I ended up not using tidf this isnt really necessary and the statistics file this is based on is not up to date anymore but we get the atrget order through this as well
    Could easily be removed in the future as longa s we get the target order from somewhere else

    Parameters:
        config (dict): Configuration parameters.
    
    Returns:
        dict: feature statistics.
        dict: target statistics.
    """

    feature_stats_path = os.path.join(config["base_path"], config["paths"]["statistics"]["training_statistics"], "feature_statistics.json")
    target_stats_path = os.path.join(config["base_path"], config["paths"]["statistics"]["training_statistics"], "target_statistics.json")
    feature_stats = load_json(feature_stats_path)
    target_stats = load_json(target_stats_path)

    return feature_stats, target_stats

 
def process_shot( shot_data, encoding_params, feature_dict):
    """
    Process a single shot to calculate TF-IDF and multi-hot encoded features.
    Note: The TF-IDF calculation is not used anymore, but the function is kept for the targets sake.

    Parameters:
        shot_id (str): The shot ID.
        shot_data (dict): Data for the shot.
        tfidf_params (dict): Parameters for TF-IDF calculation.
        encoding_params (dict): Parameters for multi-hot encoding.

    Returns:
        dict: Processed features for the shot.
    """
    
    # Iterate over
    processed_shot_data = {}

    processed_shot_data["start"] = shot_data["start"]
    processed_shot_data["end"] = shot_data["end"]

    strategies_encoded = multi_hot_encode(shot_data.get("strategies", []), encoding_params["strategy_order"])
    feps_encoded = multi_hot_encode(shot_data.get("feps", []), encoding_params["fep_order"])
    processed_shot_data["strategies"] = strategies_encoded
    processed_shot_data["feps"] = feps_encoded

    features_to_process = ["whisperx_ner.pkl", "whisperx_pos.pkl"]
    for modality in feature_dict:
        for feature_name in feature_dict[modality]:
            if feature_name in shot_data:
                    processed_shot_data[feature_name] = shot_data[feature_name]  
            else:
                print(f"Error, Feature name: {feature_name} not in shot data")


    # Multi-hot encode strategies and FEPs
    

    # Metadata fields
    return processed_shot_data


def process_file(file_path, output_dir, encoding_params, feature_dict):
    """
    Process a single file to generate feature vectors for all shots.

    Parameters:
        file_path (str): Path to the input JSON file.
        output_dir (str): Directory to save the processed feature vector file.
        tfidf_params (dict): Parameters for TF-IDF calculation.
        encoding_params (dict): Parameters for multi-hot encoding.
    """
    log_message(f"Processing file: {file_path}")
    data = load_json(file_path)
    tfidf_features = {
        shot_id: process_shot(shot_data, encoding_params, feature_dict)
        for shot_id, shot_data in data.items()
    }

    output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}")
    ensure_directory_exists(output_dir)
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(tfidf_features, out_f, indent=4)
    log_message(f"TF-IDF features saved to {output_file}")


def main():
    # Load configuration
    config = load_config()
    feature_dict = load_feature_names(config)
    stats, target_stats = load_stats(config) 
    agg_files_dir = os.path.join(config["base_path"], config["paths"]["processing"]["aggregated_shots"])
    feature_vectors_dir = os.path.join(config["base_path"], config["paths"]["final"]["feature_vectors_unscaled"])

    encoding_params = {
        "strategy_order": list(target_stats["strategies_counts"].keys()), 
        "fep_order": list(target_stats["feps_counts"].keys()),
    }
    # Save the encoding parameters in a file to reference for the evaluation
    encoding_params_file = os.path.join(config["base_path"], "encoding_params.json")
    with open(encoding_params_file, "w", encoding="utf-8") as out_f:
        json.dump(encoding_params, out_f, indent=4)
    # Process files by channel
    channel_files = get_channel_files(agg_files_dir)
    for channel, files in channel_files.items():
        if channel == ("Test"):
            continue # ignore the test cases
        log_message(f"Processing channel: {channel}")
        channel_output_dir = os.path.join(feature_vectors_dir, channel)
        ensure_directory_exists(channel_output_dir)
        for file_path in files:
            process_file(file_path, channel_output_dir, encoding_params, feature_dict)


if __name__ == "__main__":
    main()
