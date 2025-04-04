import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from utils import load_config, ensure_directory_exists, log_message, load_json

def get_all_file_paths(input_dir):
    """
    Traverses the input directory (which contains subdirectories for each channel)
    and returns a list of full file paths.
    """
    all_files = []
    for channel in os.listdir(input_dir):
        if channel != "Test":
            channel_path = os.path.join(input_dir, channel)
            if os.path.isdir(channel_path):
                for file_name in os.listdir(channel_path):
                    file_path = os.path.join(channel_path, file_name)
                    all_files.append(file_path)
    return all_files


def load_targets(file_paths):
    """
    Load targets from the given file paths.

    Parameters:
        file_paths (list): List of file paths.

    Returns:
        list: List of targets corresponding to the file paths.
    """
    strategy_map = {
    "fragmentation": 0,
    "fragmentation_splitscreen": 1,
    "individualization_of_elite": 2,
    "individualization_of_layperson": 3,
    "individualization_of_reporter": 4,
    "emotionalization": 5
    }
    fep_map = {
    "intensify": 0,
    "alternating-shot": 1,
    "cut-in": 2,
    "shot-reverse-shot": 3,
    "cut-away": 4
    }

    # I want to create a dictonary with teh file name, and the strategy and fep counts
    targets = {}
    for file_path in file_paths:
        data = load_json(file_path)
        strategies = np.zeros((len(strategy_map)), dtype=int)
        feps = np.zeros((len(fep_map)), dtype=int)
        for shot_id, shot_data in data.items():
            # I want to increase the count of the strategy or fep by 1
            for strategy in shot_data.get('strategies', []):
                strategies[strategy_map[strategy]] += 1
            for fep in shot_data.get('feps', []):
                feps[fep_map[fep]] += 1
           
        targets[file_path] = {
            "strategies": strategies.tolist(),
            "feps": feps.tolist()
        }
    return targets

def calculate_class_frequencies(targets):
    """
    Calculate the frequencies of each class in the targets.

    Parameters:
        targets (dict): Dictionary of targets.

    Returns:
        dict: Dictionary of class frequencies.
    """
    strategy_frequencies, fep_frequencies = np.zeros(6), np.zeros(5) 
    for target in targets.values():
        strategy_frequencies += target['strategies']
        fep_frequencies += target['feps']
   
    strategy_freq_dict = {strategy: frequency for strategy, frequency in enumerate(strategy_frequencies)}
    fep_freq_dict = {fep: frequency for fep, frequency in enumerate(fep_frequencies)}
    return strategy_freq_dict, fep_freq_dict

def get_k_from_test_ratio(test_ratio: float, max_k: int = 10) -> int:
    k = round(1.0 / test_ratio)
    return min(k, max_k)

def costum_stratefied_k_fold_split(undistributed_files, targets, target_name, sorted_freqs, k, seed):
    folds = [[] for _ in range(k)]
    for target in sorted_freqs:
        
        target_sub_name = target[0]
        # get the files that have the target
        target_files = []
        for file_path, target_data in targets.items():
            if len(undistributed_files) == 0:
                break
            # target_data is a dict with the target_name as key
            target_data = target_data[target_name]
            # now target_data is a list with the counts of the target
            target_amount = target_data[target_sub_name]
            if target_amount> 0:
                target_files.append(file_path)
        
        # remove those that are not in undistributed files
        target_files = [file_path for file_path in target_files if file_path in undistributed_files]
        # remove the files from the undistributed files
        undistributed_files = [file_path for file_path in undistributed_files if file_path not in target_files]
        # shuffle the target files
        rng = np.random.RandomState(seed)
        target_files = rng.permutation(target_files)

        # split the target files into k folds
        # ensure that the files are distributed as evenly as possible
        # so a split that got less files in the previous iteration gets more files in the next iteration
        folds_by_size = sorted(folds, key=lambda x: len(x))
        for i, file_path in enumerate(target_files):
            fold_name = folds_by_size[i % k]
            fold_name.append(file_path)

    # lastly distribute the remaining files as evenly as possible
    folds_by_size = sorted(folds, key=lambda x: len(x))
    for i, file_path in enumerate(undistributed_files):
        fold_name = folds_by_size[i % k]
        fold_name.append(file_path)
    return folds


def create_stratefied_target_split(all_files, target_name, k, seed, test_ratio):
    
    # First we split of the test set
    # we want to reuse the code so we will simply act like we are creating a kfold split
    targets = load_targets(all_files)
    if target_name == "strategies":
        target_frequencies, _ = calculate_class_frequencies(targets)
    else:
        _, target_frequencies = calculate_class_frequencies(targets)
    # todo sorted frequencies are wrong
    sorted_freqs = sorted(target_frequencies.items(), key=lambda x: x[1])
    test_set_k = get_k_from_test_ratio(test_ratio)
    undistributed_files = list(targets.keys())
    test_folds = costum_stratefied_k_fold_split(undistributed_files, targets, target_name, sorted_freqs, test_set_k, seed)
    test_files = test_folds[0]
    if k == test_set_k - 1:
        return test_folds[1:], test_files
    else:
        undistributed_files = [file_path for file_path in undistributed_files if file_path not in test_files]
        # we recalculated the target frequencies since we removed the test files 
        targets = load_targets(undistributed_files)
        if target_name == "strategies":
            target_frequencies, _ = calculate_class_frequencies(targets)
        else:
            _, target_frequencies = calculate_class_frequencies(targets)
        sorted_freqs = sorted(target_frequencies.items(), key=lambda x: x[1])
        # now we have the test files we can create the kfold splits for the training and validation data
        train_val_folds = costum_stratefied_k_fold_split(undistributed_files, targets,target_name, sorted_freqs, k, seed)
        return train_val_folds, test_files

    
def format_folds(folds, test_files, k):
    formated_folds = [
        {
            "fold": i + 1,
            "train": [file for j, fold in enumerate(folds) if j != i for file in fold],
            "val": folds[i],
            "test": test_files  # same held-out test set for every fold
        }
        for i in range(k)
    ]
    return formated_folds

def create_kfold_splits(all_files, k, seed, output_dir, test_ratio):
    """
    Create k-fold splits using scikit-learn's KFold.

    Steps:
      1. Shuffle all_files using the seed.
      2. Separate out a held-out test set based on the test_ratio.
      3. Apply KFold to the remaining training and validation data.

    Returns a list of dictionaries, one per fold, where each dictionary has:
      - "fold": fold number (1-indexed)
      - "train": list of file paths for training in that fold.
      - "val": list of file paths for validation in that fold.
      - "test": list of file paths for testing (the held-out test set, same for all folds).
    """
   
    
     # Load targets for stratification
   
    strategy_train_val, strategy_test = create_stratefied_target_split(all_files, "strategies",  k, seed, test_ratio)
    fep_train_val, fep_test = create_stratefied_target_split(all_files, "feps",  k, seed, test_ratio)

    # Create the final fold structure
    startegy_folds = format_folds(strategy_train_val, strategy_test, k)
    output_path = os.path.join(output_dir, "kfold_splits_strategies.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(startegy_folds, f, indent=4)
    log_message(f"K-fold splits for strategies saved to {output_path}.")

    fep_folds = format_folds(fep_train_val, fep_test, k)   
    output_path = os.path.join(output_dir, "kfold_splits_feps.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fep_folds, f, indent=4)
    log_message(f"K-fold splits for feps saved to {output_path}.")



def create_generalization_split(all_files, test_channel, output_dir):
    """
    Create a split for the generalization experiment.
    Files from the specified test_channel (subdirectory name) are held out as test data,
    while files from all other channels are used as training data.
    """
    train_files = []
    test_files = []

    for file_path in all_files:
        if test_channel in file_path: #could be probelmatic if something like "Die Welt ist schön" is in vid title but since not so many stories a brief manual check is fine
            test_files.append(file_path)
        else:
            train_files.append(file_path)  

    generalization_split =  {"train": train_files, "test": test_files}

    gen_output_path = os.path.join(output_dir, "generalization_split.json")
    with open(gen_output_path, 'w', encoding='utf-8') as gen_file:
        json.dump(generalization_split, gen_file, indent=4)
    log_message(f"Generalization split saved to {gen_output_path}.")



if __name__ == "__main__":

    # Read config
    config = load_config()
    input_dir = os.path.join(config['base_path'], config['paths']['processing']['aggregated_shots']) 
    output_dir = os.path.join(config['base_path'], config['paths']['processing']['data_splits'])
    ensure_directory_exists(output_dir)
    k = config['dataset']['k']
    test_ratio = config['dataset']['test_ratio']
    seed = config['dataset']['random_seed']
    test_channel = config['dataset']['test_channel']

    # Gather all file paths
    all_files = get_all_file_paths(input_dir)

    # Splits
    create_kfold_splits(all_files, k, seed, output_dir, test_ratio=test_ratio)
    create_generalization_split(all_files, test_channel, output_dir)