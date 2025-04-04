import sys
import os

# Add the `scripts` folder to the system path (if your utils etc. live there)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from datetime import datetime


# Utility functions from your utils module
from utils import (
     save_result_append, load_config, load_json, log_message, get_channel_files, save_results,
    ensure_directory_exists, save_rf_model, evaluate_model, load_data_from_files, map_aggregated_predictions_to_shots,
    evaluate_fold_on_instances, evaluate_fold_on_shots, construct_data_paths, average_metrics
)
from MultiLabelWrapper import MultiLabelWrapper

def get_vlm_preds(file_name, task_results_path):
    #file_name = file_name.split("."[-1])
    file_name = file_name.split(".json")[0] + "_predictions.json" #split at last point!
    vlm_results = load_json(os.path.join(task_results_path, file_name))
    vlm_results = vlm_results["shots"]
    y_pred = []
    for shot in vlm_results:
        y_pred.append(shot["prediction"])
    return y_pred # might have to change shape

def get_feature_vec(file_name, feature_vec_path, startegies_bool):
    for channel_dir in os.listdir(feature_vec_path):
        if not os.path.isdir(os.path.join(feature_vec_path, channel_dir)):
            continue
        for file in os.listdir(os.path.join(feature_vec_path, channel_dir)):
            if file == file_name:
                feature_vec = load_json(os.path.join(feature_vec_path, channel_dir, file_name))
                y_test = []
                for shot in feature_vec.values():
                    key = "strategies" if startegies_bool else "feps"
                    y_test.append(shot[key])
                return y_test



def get_file_data(file_name, task_results_path, feature_vec_path, startegies_bool):
    y_pred = get_vlm_preds(file_name, task_results_path)
    y_test = get_feature_vec(file_name, feature_vec_path, startegies_bool)
    if len(y_pred) != len(y_test):
        print("Lengths do not match") #TV-20220208-2022-0400.webl.h264_story02.json
    y_test = np.array(y_test)
    y_all = np.ones(shape=y_test.shape)
    y_none = np.zeros(shape=y_test.shape)
    y_random = np.random.randint(0, 2, size=y_test.shape)

    return y_pred, list(y_test), list(y_all), list(y_none), list(y_random)


def get_split_data(split_dir, task_results_path, feature_vec_path, startegies_bool):
    
    split_data_pred = []
    split_data_test = []
    split_data_random = []
    split_data_all = []
    split_data_none = []
    for json_file in os.listdir(split_dir):
        if json_file.endswith(".json"):
            # get only the file name
            json_file = json_file.split("/")[-1]
            # okay i have to add filename_predictions.json to the end of the file name
            y_pred, y_test,  y_all, y_none, y_random = get_file_data(json_file, task_results_path, feature_vec_path, startegies_bool)  
            split_data_pred += y_pred
            split_data_test += y_test
            split_data_all += y_all
            split_data_none += y_none
            split_data_random += y_random

    return split_data_pred, split_data_test, split_data_all, split_data_none, split_data_random

def evaluate_task(split_dir, task_results_path, feature_vec_path, startegies_bool):
    y_pred, y_test, y_all, y_none, y_random = get_split_data(split_dir, task_results_path, feature_vec_path, startegies_bool)
    metrics_vlm = evaluate_model(y_pred, y_test)
    metrics_all = evaluate_model(y_all, y_test)
    metrics_none = evaluate_model(y_none, y_test)
    metrics_random = evaluate_model(y_random, y_test)
    return metrics_vlm, metrics_all, metrics_none, metrics_random


def main():
    # Load configuration
    config = load_config()
    vlm_results_path = os.path.join(config['base_path'], config['paths']['model_ouputs']['vlm'])
    # need to get each subfolder in vlm_results_path

    data_base_path = os.path.join(config['base_path'], config['paths']['final']['vlm_data'])
    feature_vec_path = os.path.join(config['base_path'], config['paths']['final']['feature_vectors'])
    results_dir = os.path.join(config['base_path'], config['paths']['results'])
    final_results_path = os.path.join(results_dir, "vlm_experiment_results.csv")
    ensure_directory_exists(results_dir)
    all_results = []
    i = 0
    for dir_name in os.listdir(vlm_results_path):
        if not os.path.isdir(os.path.join(vlm_results_path, dir_name)):
            continue
            
        # Process each channel
        fep_task_names = ["film_editing", "film_editing_defs", "film_editing_perclass"]
        strategie_task_names = ["strategies", "strategies_defs", "strategies_perclass"]
        if dir_name in fep_task_names:
            strategies_bool = False
        elif dir_name in strategie_task_names:
            strategies_bool = True
        channels = ["Test", "BildTV", "CompactTV", "HeuteJournal", "Tagesschau", "Welt"]
        for split in os.listdir(data_base_path): 
            if split in channels:
                continue  # Only process test data 
            if strategies_bool and split == "test_kfold_splits_fep":
                continue
            if not strategies_bool and split == "test_kfold_splits_strategie":
                continue
            split_dir = os.path.join(data_base_path, split)
            if os.path.isdir(split_dir):
                log_message(f"Processing split: {split}")
            task_results_path = os.path.join(vlm_results_path, dir_name)
            results_vlm, results_all, results_none, results_random = evaluate_task(split_dir, task_results_path, feature_vec_path, strategies_bool)
            experiment_results = {
                                "model_name": "Qwen2.5-VL-3B-Instruct",
                                "prompt_name": dir_name,
                                "split": split,
                                
                            }
            #"metrics": results_vlm
            # madd the metrics dict to the experiment_results dict
            experiment_results = {**experiment_results, **results_vlm}
            save_result_append( experiment_results, final_results_path)
            if i % 3 == 0:
                experiment_results = {
                                    "model_name": "Baseline-All",
                                    "prompt_name": dir_name,
                                    "split": split,
                                }
                experiment_results = {**experiment_results, **results_all}
                save_result_append( experiment_results, final_results_path)

                experiment_results = {
                                    "model_name": "Baseline-None",
                                    "prompt_name": dir_name,
                                    "split": split,

                                }
                experiment_results = {**experiment_results, **results_none}
                save_result_append( experiment_results, final_results_path)

                experiment_results = {
                                    "model_name": "Baseline-Random",
                                    "prompt_name": dir_name,
                                    "split": split,
        
                                }
                experiment_results = {**experiment_results, **results_random}
                save_result_append( experiment_results, final_results_path)
            i += 1
            
if __name__ == "__main__":
    main()