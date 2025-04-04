import os
import json
import math
import numpy as np
import yaml
import pickle
from collections import Counter, defaultdict
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import torch
import pickle
import base64
import io
import csv
from PIL import Image
import re
import json
from sklearn.metrics import (
    precision_score,
    fbeta_score,
    recall_score,
    f1_score,
    accuracy_score, 
    hamming_loss,
    multilabel_confusion_matrix
)
from contextlib import contextmanager
import pathlib

@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup



 # 5) Aggregate metrics across folds for this experiment.
def average_metrics(list_of_dicts):
    """
    Given a list of dictionaries each containing metrics,
    compute the average for each metric. Handles scalars,
    lists, and numpy arrays. Extend as needed.
    """
    # Step 1: Collect values in aggregator
    aggregator = defaultdict(list)
    for metrics_dict in list_of_dicts:
        for metric_name, metric_value in metrics_dict.items():
            aggregator[metric_name].append(metric_value)

    # Step 2: Average each metric
    averaged = {}
    for metric_name, values_list in aggregator.items():
        if metric_name == "multilabel_confusion_matrix":
            # average confusion matrices element-wise
            confusion_matrix_aggreator = defaultdict(list)
            for confusion_matrix in values_list:
                for class_index, class_values in enumerate(confusion_matrix):
                    class_list = []
                    for i, row in enumerate(class_values):
                        # need a flattend list for the confusion matrix
                        class_list += row
                    confusion_matrix_aggreator[class_index].append(class_list)
            confusion_matrices = {}
            for class_index, class_values in confusion_matrix_aggreator.items():
                averaged_values = np.mean(class_values, axis=0)
                confusion_matrices[class_index] = averaged_values
            averaged[metric_name] = confusion_matrices
            continue
            
            
        # Just take the first item to inspect its type
        first_val = values_list[0]

        # Case A: All values are scalars (int, float, np.float64, etc.)
        if all(isinstance(v, (int, float, np.floating)) for v in values_list):
            averaged[metric_name] = sum(values_list) / len(values_list)

        # Case B: All values are lists (e.g. "precision_by_class")
        elif all(isinstance(v, list) for v in values_list):
            # Assuming they're all the same length
            average_array = np.mean(values_list, axis=0)
            averaged[metric_name] = average_array.tolist()

        # Case C: All values are NumPy arrays (array-wise average)
        elif all(isinstance(v, np.ndarray) for v in values_list):
            sum_arr = np.zeros_like(first_val, dtype=float)
            for arr in values_list:
                sum_arr += arr
            averaged[metric_name] = sum_arr / len(values_list)

        else:
            # Handle mixed types or unsupported types
            raise TypeError(f"Unsupported type for metric '{metric_name}'. "
                            f"Value: {first_val}")

    return averaged

def construct_data_paths(paths, base_data_path):
        """
        Convert the relative paths from k-fold splits into full file paths using base_data_path.
        Expects each path to include at least a channel and filename.
        """
        new_paths = []
        for p in paths:
            p_norm = os.path.normpath(p)
            parts = p_norm.split(os.sep)
            if len(parts) >= 2:
                channel = parts[-2]
                filename = parts[-1]
                new_path = os.path.join(base_data_path, channel, filename)
                new_paths.append(new_path)
            else:
                log_message(f"WARNING: Unexpected path format: {p}")
        return new_paths


def evaluate_fold_on_instances(clf, X_test, Y_test):
    """
   This method evaluates on the aggregated instances, it is only for debugging purposes.
    It is not used in the final evaluation, that is evaluate_fold_on_shots.

    Parameters:
        clf (MultiOutputClassifier): Trained classifier.
        X_test (np.ndarray): Feature matrix for test instances (aggregated).
        Y_test (np.ndarray): Ground-truth instance-level labels.
    
    Returns:
        dict: Evaluation metrics.
    """

    
    y_pred_agg = clf.predict(X_test)

    results = evaluate_model(y_pred_agg, Y_test)
    return results


def evaluate_fold_on_shots(clf, X_test, test_news_stories, Y_test, compare_news_stories, window_size=2, costum_preds = None):
    """
    Evaluate the classifier on the test set for one fold and return a dictionary of metrics.
    First maps the instances to shots and then evaluates the model on the shots.
    This is the main evaluation method for the classiifiers.

    Parameters:
        clf (MultiOutputClassifier): Trained classifier.
        X_test (np.ndarray): Feature matrix for test instances (aggregated).
        Y_test (np.ndarray): Ground-truth shot-level labels.
        test_news_stories (np.ndarray): Array of shape (n_shots,) with the news story for each shot.
        compare_news_stories (np.ndarray): Array of shape (n_shots,) with the news story for comparison.
        target_name (str): Name of the target being evaluated.
        window_size (int): Number of shots aggregated into one instance.
    
    Returns:
        dict: Evaluation metrics.
    """
    # ensure X_test and test_news_stories are the same length
    if len(X_test) != len(test_news_stories):
        log_message("WARNING: X_test and test_news_stories have different lengths.")
        return None
    
    y_pred_agg = clf.predict(X_test)
    y_pred_shots = map_aggregated_predictions_to_shots(y_pred_agg, test_news_stories, compare_news_stories, window_size)
    
    # Ensure y_pred_shots and Y_test have the same length
    if len(y_pred_shots) != len(Y_test):
        log_message(f"ERROR: Length mismatch between y_pred_shots ({len(y_pred_shots)}) and Y_test ({len(Y_test)}).")
        return None

    results = evaluate_model(y_pred_shots, Y_test)
    return results


def map_aggregated_predictions_to_shots(
    y_pred_agg: np.ndarray,
    test_news_stories: np.ndarray,
    compare_news_stories: np.ndarray,
    window_size: int = 2
) -> np.ndarray:
    """
    Map aggregated (2-shot) instance predictions back to shot-level predictions
    using 'min' for shots that appear in two instances.

    For each news story of length S (S >= 2):
      - There are S-1 aggregated instances, each covering (shot_k, shot_{k+1}).
      - The first shot gets its label from instance 0.
      - The last shot gets its label from instance (S-2).
      - Every middle shot gets the min of its two overlapping instances.
    
    If S = 1, there is no 2-shot instance for that story. Here, we (by default) set
    that single shot's prediction to all zeros, but you can adapt that as needed.

    Parameters
    ----------
    y_pred_agg : np.ndarray
        Shape = (n_instances, n_labels). Predictions for each 2-shot instance.
    test_news_stories : np.ndarray
        Shape = (n_shots,). Story identifier for each shot in the test set.
    compare_news_stories : np.ndarray
        Shape = (n_shots,). Must match `test_news_stories` if they truly correspond.
    window_size : int
        The size of each window (normally = 2 for 2‐shot windows).

    Returns
    -------
    np.ndarray
        Shot-level predictions of shape (n_shots, n_labels).
    """


    n_shots = len(compare_news_stories)
    n_labels = y_pred_agg.shape[1]
    
    # Prepare output array
    y_pred_shots = np.zeros((n_shots, n_labels), dtype=y_pred_agg.dtype)

    # 1) Group shot indices by their story, in the order they appear
    shot_indices_by_story = defaultdict(list)
    for shot_idx, story_id in enumerate(compare_news_stories):
        shot_indices_by_story[story_id].append(shot_idx)

    # Collect unique stories in the order they appear:
    ordered_stories = []
    last_story = None
    for story_id in compare_news_stories:
        if story_id != last_story:
            ordered_stories.append(story_id)
            last_story = story_id

    # 2) Iterate over each story in order, mapping shots back to aggregated instances
    instance_offset = 0
    for story_id in ordered_stories:
        shots = shot_indices_by_story[story_id]
        S = len(shots)

        if S == 1:
            # Corner case: only one shot => no 2-shot instance to map from
            # Decide how you want to handle single-shot stories. 
            # By default, set to zero or replicate the existing instance if available, etc.
            y_pred_shots[shots[0]] = 0  
        else:
            # For S >= 2, we have exactly (S - 1) aggregated instances
            # The i-th instance in this story covers (shots[i], shots[i+1])
            # and is located at y_pred_agg[instance_offset + i].
            # So the first shot = shots[0] only belongs to instance_offset + 0
            # The last shot = shots[S-1] only belongs to instance_offset + (S-2)
            # Each middle shot i is in both instance_offset + (i-1) and instance_offset + i
            for k in range(S):
                shot_idx = shots[k]
                if k == 0:
                    # First shot => from instance k=0
                    y_pred_shots[shot_idx] = y_pred_agg[instance_offset + k]
                elif k == S - 1:
                    # Last shot => from instance (S-2)
                    y_pred_shots[shot_idx] = y_pred_agg[instance_offset + (k - 1)]
                else:
                    # Middle shots => min of two overlapping instances
                    pred_left = y_pred_agg[instance_offset + (k - 1)]
                    pred_right = y_pred_agg[instance_offset + k]
                    y_pred_shots[shot_idx] = np.minimum(pred_left, pred_right)

            # Advance the instance offset by (S - 1) for the next story
            instance_offset += (S - 1)

    return y_pred_shots


def map_aggregated_predictions_to_shots_old(y_pred_agg, test_news_stories, compare_news_stories, window_size=2):
    """
    Map aggregated-instance predictions back to shot-level predictions.

    Each aggregated instance was formed by combining shots using a specified window size.
    To revert, we use a 'min' approach for overlapping shots:
    Per story:
      - Shot 0 gets the label from instance 0.
      - Shot (n_shots-1) gets the label from the last instance.
      - Middle shots get the element-wise minimum of the overlapping aggregated predictions.
    
    Parameters:
        y_pred_agg (np.ndarray): Array of shape (n_instances, n_labels).
        test_news_stories (np.ndarray): Array of shape (n_shots,) with the news story for each shot.
        compare_news_stories (np.ndarray): Array of shape (n_shots,) with the news story for comparison.
        window_size (int): Number of shots aggregated into one instance.
    
    Returns:
        np.ndarray: Shot-level predictions (n_shots, n_labels).
    """
    prev_news_story = None
    next_news_story = None
    shot_predictions = []
    news_stories_new = []

    for i, news_story in enumerate(test_news_stories):
        if i < len(test_news_stories) - 1:
            next_news_story = test_news_stories[i + 1]
        else:
            # very last shot of the test set
            shot_predictions.append(y_pred_agg[i - window_size + 1])
            news_stories_new.append(news_story)
            if i != len(compare_news_stories) - 1: 
                shot_predictions.append(y_pred_agg[i])
                news_stories_new.append(news_story)
            for i, (story_new, story_compare) in enumerate(zip(news_stories_new, compare_news_stories)):
                if story_new != story_compare:
                    print("ERROR: News stories do not match between test and compare data.")
                    print(f"Test: {story_new}, Compare: {story_compare}")
                    print("at index", i)
                    break
            return np.array(shot_predictions)
        
        if prev_news_story != news_story or prev_news_story is None:
            prev_news_story = news_story
            shot_predictions.append(y_pred_agg[i])
            news_stories_new.append(news_story)
            if next_news_story != news_story:
                # If news story only consists of one instance
                # the problem is that we can't know how many shots were in the instance
                # so we can't know how many shots to append
                # for now we will use compare_news_stories to know how many shots to append
                j = len(news_stories_new)
                while news_story == compare_news_stories[j]:
                    shot_predictions.append(y_pred_agg[i])
                    news_stories_new.append(news_story)
                    j += 1

        else:
            pred = y_pred_agg[i] + y_pred_agg[i - 1]
            pred = np.array(pred) / window_size
            pred = np.where(pred > 0.5, 1, 0)
    
            shot_predictions.append(pred)
            news_stories_new.append(news_story)
            if next_news_story != news_story:
                prev_news_story = news_story
                shot_predictions.append(y_pred_agg[i])
                news_stories_new.append(news_story)
    
    return np.array(shot_predictions)

def load_data_from_files(file_paths, selected_feature_indices=None, target=False):
    """
    Load data (features, strategies, feps) from a list of JSON files that contain
    aggregated instances with multiple feature keys.

    """
    X = []
    Y_strategies = []
    Y_feps = []
    news_stories = []

    for path in file_paths:
        news_story = os.path.basename(path)
        if not os.path.exists(path):
            log_message(f"WARNING: file not found, skipping: {path}")
            continue

        data = load_json(path, printing = False)

        # If the loaded JSON is a dict, iterate over its values. Otherwise, it's likely a list.
        if isinstance(data, dict):
            instances = list(data.values())
        else:
            instances = data

        for instance in instances:
            # 1) Collect the target labels
            if "strategies" in instance:
                Y_strategies.append(instance["strategies"])
            else:
                # fallback if missing
                Y_strategies.append([])

            if "feps" in instance:
                Y_feps.append(instance["feps"])
            else:
                Y_feps.append([])

            # 2) If this call is for target-only, skip feature extraction
            if target:
                # We'll still track the news story for each instance
                news_stories.append(news_story)
                continue

            # 3) Otherwise, gather *all numeric features* from the instance
            #    except for keys that are definitely not features (start/end, labels, QAs, etc.).
            feature_vector = []
            for key, value in instance.items():
                # Skip known non-feature keys
                if key in ["start", "end", "strategies", "feps"]:
                    continue

                # value is expected to be a list of numeric values (floats/ints)
                if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                    feature_vector.extend(value)
                else:
                    print("Error, non numeric values")
                    print(key, value)
                    pass

            # 4) If we have a feature selection scenario, select only the chosen indices
            if selected_feature_indices is not None:
                feature_vector = [feature_vector[i] for i in selected_feature_indices]

            X.append(feature_vector)
            news_stories.append(news_story)

    # Convert lists to numpy arrays
    if target:
        X = np.empty((0, 0))
    else:
        X = np.array(X, dtype=float) if X else np.empty((0, 0))

    Y_strategies = np.array(Y_strategies, dtype=int) if Y_strategies else np.empty((0, 0), dtype=int)
    Y_feps = np.array(Y_feps, dtype=int) if Y_feps else np.empty((0, 0), dtype=int)

    return X, Y_strategies, Y_feps, np.array(news_stories)

def evaluate_model(y_pred, y_test):
    """
    Evaluate a multi-label or multi-class classification model, printing and
    returning a dictionary of metrics such as precision, recall, F1, etc.

    Parameters:
    -----------
    y_pred : array-like of shape (n_samples, n_labels)
        Predicted labels.
    y_test : array-like of shape (n_samples, n_labels)
        True labels.

    Returns:
    --------
    results : dict
        Dictionary containing various evaluation metrics.
    """
    results = {}

    # --- Precision ---
    #macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    #print("Macro Precision:", macro_precision)
    #results["macro_precision"] = macro_precision

    micro_precision = precision_score(y_test, y_pred, average="micro", zero_division=0)
    print("Micro Precision:", micro_precision)
    results["micro_precision"] = micro_precision

    # --- Recall ---
    #macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    #print("Macro Recall:", macro_recall)
    #results["macro_recall"] = macro_recall

    micro_recall = recall_score(y_test, y_pred, average="micro", zero_division=0)
    print("Micro Recall:", micro_recall)
    results["micro_recall"] = micro_recall

    # --- F1 Score ---

    #macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    #print("Macro F1 Score:", macro_f1)
    #results["macro_f1"] = macro_f1

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    print("Micro F1 Score:", micro_f1)
    results["micro_f1"] = micro_f1

    # costumn weighted f-1 score to prioritise recall (beta = 2 means recall is twice as important as precision)

    micro_f2 = fbeta_score(y_test, y_pred, beta=2, average="micro", zero_division=0)
    print("Micro F-2 Score:", micro_f2)
    results["micro_f2"] = micro_f2


    # --- Subset Accuracy ---
    subset_acc = accuracy_score(y_test, y_pred)
    print("Subset Accuracy:", subset_acc)
    results["subset_accuracy"] = subset_acc

    # --- Hamming Loss ---
    hloss = hamming_loss(y_test, y_pred)
    print("Hamming Loss:", hloss)
    results["hamming_loss"] = hloss

    # --- Average Precision Score ---
    #avg_precision = average_precision_score(y_test, y_pred, average="macro")
    #print("Macro Average Precision Score:", avg_precision)
    #results["macro_average_precision"] = avg_precision

    # per class metrics

    precision_by_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    print("Precision by class:", precision_by_class)
    results["precision_by_class"] = precision_by_class.tolist()

    recall_by_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    print("Recall by class:", recall_by_class)
    results["recall_by_class"] = recall_by_class.tolist()

    f1_by_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    print("F1 Score by class:", f1_by_class)
    results["f1_by_class"] = f1_by_class.tolist()

    # --- Multilabel Confusion Matrix ---
    cm = multilabel_confusion_matrix(y_test, y_pred)
    print("Multilabel Confusion Matrix:")
    print(cm)
    # we can store as a list of lists (for JSON-safe format)
    results["multilabel_confusion_matrix"] = cm.tolist()
    

    return results


def save_result_append(result_row, filename):
    """
    Appends a single row (dictionary) to a CSV file.
    If the file does not exist yet, creates it and writes the header.
    Otherwise, appends the row to the existing CSV.
    """
    import csv
    import os

    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(result_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

def parse_output(output_text: str) -> list:
    """
    Attempts to parse the model output as JSON and return the "patterns" list.
    If that fails, falls back to a regex-based extraction of the patterns array.
    
    Returns:
        list: the extracted "patterns" as a Python list (could be empty).
              if no patterns found or everything fails, returns an empty list.
    """

    # 1. Remove any triple-backtick code fences (including possible "json" language tags)
    cleaned = re.sub(r"```(?:json)?", "", output_text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```", "", cleaned).strip()

    # 2. Try to parse as JSON directly
    try:
        parsed_json = json.loads(cleaned)
        # If it’s a dictionary with a "patterns" key, return it if it’s a list
        if isinstance(parsed_json, dict) and "patterns" in parsed_json:
            if isinstance(parsed_json["patterns"], list):
                return parsed_json["patterns"]
            else:
                # If "patterns" is not a list, wrap it in one
                return [parsed_json["patterns"]]
        # If JSON is valid but doesn’t have "patterns", fallback to empty
    except json.JSONDecodeError:
        pass

    # 3. If direct parse fails or we have no "patterns", try the regex approach
    regex = r'"patterns"\s*:\s*\[([^\]]*)\]'
    match = re.search(regex, cleaned)
    if match:
        bracket_content = match.group(1).strip()
        try:
            # bracket_content might be something like: '"pattern1", "pattern2"'
            # so we wrap it in brackets: -> ["pattern1", "pattern2"]
            return json.loads(f"[{bracket_content}]")
        except json.JSONDecodeError:
            pass

    # 4. If all else fails, return an empty list
    return []

def load_feature_names(config):
    """
    Load feature names from the configuration file.

    Parameters:
        config (dict): Configuration parameters.

    Returns:
        dict: modalities cóntaining lists of feature names.
    """
    feature_json_path = os.path.join(config["base_path"], config["paths"]["processing"]["feature_json"])
    feature_dict = load_json(feature_json_path)
    return feature_dict

def resize_image(image_input, max_size=224):
    """Resize images safely, handling both Base64 strings and PIL Images."""
    if isinstance(image_input, str):  # Base64 case
        image_data = base64.b64decode(image_input)
        image = Image.open(io.BytesIO(image_data))
    elif isinstance(image_input, Image.Image):  # PIL Image case
        image = image_input
    else:
        raise TypeError(f"Expected Base64 string or PIL Image, got {type(image_input)}")

    image.thumbnail((max_size, max_size))  # Resize image to max_size x max_size
    return image


def load_config(config_path=None):
    """
    Load the configuration file.

    Parameters:
        config_path (str): Path to the configuration file. If None, a default path is used.

    Returns:
        dict: Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = r"C:\Users\Lea\Documents\MA\Repo\TemporalPatternDetection\config.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def read_video_list(video_list_path):
    """
    Read the video list from a text file.

    Each line in the file is expected to have the format: `channel,video_id`.
    """
    with open(video_list_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip().split(',') for line in lines if line.strip()]

def calculate_tfidf(term_counts, doc_counts, total_docs, vocab):
    """
    Calculate the Term Frequency-Inverse Document Frequency (TF-IDF) vector for a given term count and vocabulary.

    Parameters:
        term_counts (dict): Term frequencies in the document.
        doc_counts (dict): Document frequencies of terms in the corpus.
        total_docs (int): Total number of documents in the corpus.
        vocab (list): Vocabulary for which to calculate TF-IDF.

    Returns:
        list: A vector of TF-IDF values corresponding to the given vocabulary.
    """
    tfidf_vector = []
    total_terms = sum(term_counts.values())
    for term in vocab:
        tf = term_counts.get(term, 0) / total_terms if total_terms > 0 else 0
        idf = math.log(total_docs / (1 + doc_counts.get(term, 0)))
        tfidf_vector.append(tf * idf)
    return tfidf_vector


def multi_hot_encode(items, pattern_order):
    """
    Create a multi-hot encoded vector for the given items based on a predefined order.

    Parameters:
        items (list): Items to encode.
        pattern_order (list): A predefined order for encoding the items.

    Returns:
        list: A multi-hot encoded vector where each position corresponds to the presence of an item.
    """
    vector = [0] * len(pattern_order)
    for item in items:
        if item in pattern_order:
            index = pattern_order.index(item)
            vector[index] = 1
    return vector


def save_json(data, file_path):
    """
    Save a Python dictionary or list as a JSON file.

    Parameters:
        data (dict or list): The data to save.
        file_path (str): Path to the JSON file to write.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    print(f"Data written to {file_path}")


def load_json(file_path, printing = True):
    """
    Load data from a JSON file.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: The loaded JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        if printing:
            print(f"Loading data from {file_path}")
        return json.load(file)
    
def load_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
        file_path (str): Path to the pickle file.

    Returns:
        object: The data loaded from the pickle file.
    """
    with set_posix_windows():
        with open(file_path, 'rb') as file:
            return pickle.load(file)


def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists. If the directory does not exist, it is created.

    Parameters:
        directory_path (str): The directory path to check or create.
    """
    os.makedirs(directory_path, exist_ok=True)


def sort_shots_by_key(data):
    """
    Sort a dictionary of shots by their keys, which are assumed to be integers.

    Parameters:
        data (dict): A dictionary where the keys represent shot IDs.

    Returns:
        list: A sorted list of tuples (key, value) based on the integer keys.
    """
    return sorted(data.items(), key=lambda x: int(x[0]))


def aggregate_features(feature_lists, method='mean'):
    """
    Aggregate a list of feature lists using a specified method.

    Parameters:
        feature_lists (list of lists): A list containing lists of features.
        method (str): Aggregation method ('mean' or 'max'). Default is 'mean'.

    Returns:
        list: A list of aggregated features.
    """
    arrays = [np.array(features) for features in feature_lists]
    if method == 'mean':
        return np.mean(arrays, axis=0).tolist()
    elif method == 'max':
        return np.max(arrays, axis=0).tolist()
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")


def log_message(message, log_file=None):
    """
    Log a message to the console and optionally to a log file.

    Parameters:
        message (str): The message to log.
        log_file (str): Optional path to a log file. If provided, the message is also written to the file.
    """
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as file:
            file.write(f"{message}\n")


def extract_metadata(data, key):
    """
    Extract metadata (e.g., strategies, stories, or FEPs) from a dictionary of shot-level data.

    Parameters:
        data (dict): A dictionary containing shot-level data.
        key (str): The metadata key to extract.

    Returns:
        list: A list of metadata items across all shots.
    """
    return [item for shot in data.values() for item in shot.get(key, [])]


def get_channel_files(base_dir, extension='.json'):
    """
    Retrieve all files with a specific extension from subdirectories in a base directory.

    Parameters:
        base_dir (str): The base directory containing subdirectories for channels.
        extension (str): The file extension to filter by (default is '.json').

    Returns:
        dict: A dictionary with channel names as keys and lists of file paths as values.
    """
    channel_files = {}
    for channel in os.listdir(base_dir):
        channel_path = os.path.join(base_dir, channel)
        if os.path.isdir(channel_path):
            channel_files[channel] = [
                os.path.join(channel_path, f)
                for f in os.listdir(channel_path) if f.endswith(extension)
            ]
    return channel_files

def setup_plot_style(title, xlabel, ylabel, figsize, dpi=300):
    """
    Set up the styling for a plot.

    Parameters:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple): Size of the figure.
        dpi (int): Resolution of the figure.
    """
    sns.set(style="whitegrid", font_scale=1.2)
    sns.set_palette("colorblind")
    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title, pad=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    #plt.tight_layout()

def save_model(model, optimizer, epoch, file_path):
    """Save a model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, file_path)

def load_model(file_path, model, optimizer=None):
    """Load a model checkpoint."""
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def save_results(results, file_path, fieldnames=None):
    """
    Save results to a CSV file.

    Parameters:
        results (list of dict): A list of dictionaries containing the results to save.
        file_path (str): Path to the CSV file.
        fieldnames (list): Optional. List of field names for the CSV file.
    """
    ensure_directory_exists(os.path.dirname(file_path))

    if not fieldnames:
        # Automatically infer fieldnames from the first result
        fieldnames = results[0].keys() if results else []

    # Write results to a CSV file
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    log_message(f"Results saved to {file_path}")


def save_rf_model(clf, file_path):
    """
    Save a Random Forest model to a file.
    
    Parameters:
        clf (MultiOutputClassifier): Trained Random Forest model.
        file_path (str): Path to save the model.
    """
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump(clf, f)
    log_message(f"Random Forest model saved to {file_path}")

def load_rf_model(file_path):
    """
    Load a Random Forest model from a file.
    
    Parameters:
        file_path (str): Path to load the model from.
    
    Returns:
        MultiOutputClassifier: Loaded Random Forest model.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
