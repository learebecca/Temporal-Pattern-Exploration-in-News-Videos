import sys
import os

# Add the `scripts` folder to the system path (if your utils etc. live there)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
from imblearn.over_sampling import SMOTE, RandomOverSampler

import xgboost as xgb

# Utility functions from your utils module
from utils import (
    save_result_append, load_config, load_json, log_message, get_channel_files, save_results,
    ensure_directory_exists, save_rf_model, evaluate_model, load_data_from_files, map_aggregated_predictions_to_shots,
    evaluate_fold_on_instances, evaluate_fold_on_shots, construct_data_paths, average_metrics
)
from MultiLabelWrapper import MultiLabelWrapper


def correct_parameters(parameters):
    # i have strings with "None" in the parameters, this is a workaround to convert them to None
    for key in parameters:
        if parameters[key] == "None":
            parameters[key] = None
        if parameters[key] == "True":
            parameters[key] = True
        if parameters[key] == "False":
            parameters[key] = False
    return parameters


def oversample_data(X_train, Y_train, seed):
    # Oversample the minority class using SMOTE
    # here we only have a binary classification problem
    # does smote still have an adventage for binary classification?
    oversampler = SMOTE(random_state=seed)
    X_train_resampled, Y_train_resampled = oversampler.fit_resample(X_train, Y_train)
    return X_train_resampled, Y_train_resampled


def train_model(X_train, Y_train, model_name, model_parameters, oversample, per_class):
    # 
    """
    Train a multi-output classifier on the given data.

    Parameters:
        X_train (np.ndarray): Feature matrix.
        Y_train (np.ndarray): 2D array of multi-label targets.
        model_name (str): Name of the model to train.
        model_parameters (dict): Parameters for the model.

    Returns:
        MultiOutputClassifier: Trained classifier.
    """
    model_parameters = correct_parameters(model_parameters)
    seed = model_parameters["random_state"]

    classifiers = []
    if per_class:
        for target in range(len(Y_train[0])):
            # we now want to train a model for each target
            Y_train_target = Y_train[:,target]
            if oversample:
                X_train_sampled, Y_train_target = oversample_data(X_train, Y_train_target, seed)
        
            if model_name == "RandomForest":
                clf = train_rf(X_train_sampled, Y_train_target, rf_params=model_parameters, per_class=per_class)
            elif model_name == "XGBoost":
                clf = train_XGB(X_train_sampled, Y_train_target, xgb_params=model_parameters, per_class=per_class)
            else:
                raise ValueError(f"Model {model_name} not supported.")
            classifiers.append(clf)
        clf = MultiLabelWrapper(classifiers)
    else:
        if model_name == "RandomForest":
            clf = train_rf(X_train, Y_train, rf_params=model_parameters, per_class=per_class)
            
        elif model_name == "XGBoost":
            clf = train_XGB(X_train, Y_train, xgb_params=model_parameters, per_class=per_class)
        else:
            raise ValueError(f"Model {model_name} not supported.")


    return clf

def train_XGB(X_train, Y_train, xgb_params, per_class):
    """
    Train a MultiOutput XGBoost classifier.

    Parameters:
        X_train (np.ndarray): Feature matrix.
        Y_train (np.ndarray): 2D array of multi-label targets.
        xgb_params (dict, optional): Parameters for XGBClassifier.
    
    Returns:
        MultiOutputClassifier: Trained classifier.
    """
    if xgb_params is None:
        xgb_parameters = {}
    else:
        xgb_parameters = xgb_params.copy()
        # remove all entries that are None
        xgb_parameters= {key: value for key, value in xgb_parameters.items() if value is not None}
        
    
    
    clf = GradientBoostingClassifier(**xgb_parameters) if per_class else MultiOutputClassifier(GradientBoostingClassifier(**xgb_parameters))
    #clf = MultiOutputClassifier(base_estimator)
    log_message(f"Training XGB with params: {xgb_parameters}")
    clf.fit(X_train, Y_train)
    return clf




def train_rf(X_train, Y_train, rf_params, per_class):
    """
    Train a MultiOutput Random Forest classifier.

    Parameters:
        X_train (np.ndarray): Feature matrix.
        Y_train (np.ndarray): 2D array of multi-label targets.
        rf_params (dict, optional): Parameters for RandomForestClassifier.
    
    Returns:
        MultiOutputClassifier: Trained classifier.
    """
    if rf_params is None:
        rf_parameters = {}
    else:
        rf_parameters = rf_params.copy()
        # remove all entries that are None
        rf_parameters= {key: value for key, value in rf_parameters.items() if value is not None}
    
    #class_weight
    clf = RandomForestClassifier(**rf_parameters) if per_class else MultiOutputClassifier(RandomForestClassifier(**rf_parameters))
    #clf = MultiOutputClassifier(base_estimator)
    log_message(f"Training RF with params: {rf_parameters}")
    clf.fit(X_train, Y_train)
    return clf


def get_fold_data(kfold_splits, target_name, fold_idx, instance_data_path, feature_vectors_path, fs_indices):
    fold_obj = kfold_splits[fold_idx]
    train_paths, val_paths = fold_obj["train"], fold_obj["val"]

    train_data_paths = construct_data_paths(train_paths, instance_data_path)
    val_data_paths = construct_data_paths(val_paths, instance_data_path)
    # if we want to compare against shots not instances we need to load the target data from the feature vectors
    val_target_paths = construct_data_paths(val_paths, feature_vectors_path)

    if len(train_data_paths) == 0 or len(val_data_paths) == 0 or len(val_target_paths) == 0:
        log_message(f"WARNING: No training or validation files found for fold {fold_idx+1}. Skipping this fold.")
        return {}

    # For training (aggregated feature data):
    if target_name == "strategies":
        X_train, Y_train, _, _ = load_data_from_files(train_data_paths, selected_feature_indices=fs_indices)
        # For testing (aggregated feature data):
        X_test, Y_test_instances, _, test_news_stories = load_data_from_files(val_data_paths, selected_feature_indices=fs_indices)
        # For shot-level ground-truth (target=True => no X):
        _, Y_test_shots,_ , compare_news_stories = load_data_from_files( val_target_paths, target=True )
    else:
        X_train, _, Y_train, _ = load_data_from_files(train_data_paths, selected_feature_indices=fs_indices)
        X_test, _, Y_test_instances, test_news_stories = load_data_from_files(val_data_paths, selected_feature_indices=fs_indices)
        _, _, Y_test_shots, compare_news_stories = load_data_from_files(val_target_paths, target=True)
   
    data_dict = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test_instances": Y_test_instances,
        "test_news_stories": test_news_stories,
        "Y_test_shots": Y_test_shots,
        "compare_news_stories": compare_news_stories
    }
    return data_dict
    
def get_test_data(kfold_splits, instance_data_path, feature_vectors_path, fs_indices, strategy_bool):

    if len(kfold_splits) == 2:
        #this is the gernarlization test, only has train and test
        train_paths, test_paths = kfold_splits["train"], kfold_splits["test"]

    else:
        train_paths, val_paths, test_paths = kfold_splits[0]["train"], kfold_splits[0]["val"], kfold_splits[0]["test"]
        train_paths = train_paths + val_paths

    test_data_paths = construct_data_paths(test_paths, instance_data_path) 
    test_target_paths = construct_data_paths(test_paths, feature_vectors_path)
    train_val_data_paths = construct_data_paths(train_paths, instance_data_path)

    # For training (aggregated feature data):
    if strategy_bool:
        X_train, Y_train, _ , _ = load_data_from_files(train_val_data_paths, selected_feature_indices=fs_indices)
        X_test, Y_test_instance_based, _, test_news_stories = load_data_from_files(test_data_paths, selected_feature_indices=fs_indices)
        _, Y_test_shots,_, compare_news_stories = load_data_from_files(test_target_paths, target=True)
    else:
        X_train, _,Y_train, _ = load_data_from_files(train_val_data_paths, selected_feature_indices=fs_indices)
        X_test, _, Y_test_instance_based, test_news_stories = load_data_from_files(test_data_paths, selected_feature_indices=fs_indices)
        _, _,Y_test_shots, compare_news_stories = load_data_from_files(test_target_paths, target=True)
    return X_train, Y_train, X_test, Y_test_instance_based, Y_test_shots, test_news_stories, compare_news_stories


def get_best_model(all_results, parameter_names, model_name):

    best_features_strategies = None
    best_features_feps = None

    best_params_strategies = {key: None for key in parameter_names}
    best_params_feps = {key: None for key in parameter_names}


    max_recall_strategies, max_recall_feps = 0, 0
    for result in all_results:
        if result["model_name"] != model_name:
            continue
        strategies_recall = result["micro_f2_Strategy"]
        if strategies_recall > max_recall_strategies:
            max_recall_strategies = strategies_recall
            best_features_strategies = result["feature_selection"]
            best_params_strategies.update({key: result[key] for key in parameter_names})
            classfier_type_strategies = result["model_name"]
           
        feps_recall = result["micro_f2_FEP"]
        if feps_recall > max_recall_feps:
            max_recall_feps = feps_recall
            best_features_feps = result["feature_selection"]
            best_params_feps.update({key: result[key] for key in parameter_names})
            classfier_type_feps = result["model_name"]
    log_message(f"Best strategies model is a {classfier_type_strategies} with features {best_features_strategies} and parameters {best_params_strategies}")
    log_message(f"Best feps model is a {classfier_type_feps} with features {best_features_feps} and parameters {best_params_feps}")

    return best_features_strategies, best_params_strategies, classfier_type_strategies, best_features_feps, best_params_feps, classfier_type_feps


def combine_results(experiment_results, parameters, final_avg_strategies, final_avg_feps):

    final_avg_strategies_renamed = {
        f"{key}_Strategy": value for key, value in final_avg_strategies.items()
    }

    # Rename fields in final_avg_feps
    final_avg_feps_renamed = {
        f"{key}_FEP": value for key, value in final_avg_feps.items()
    }

    # Combine all four dictionaries
    combined_dict = {
        **experiment_results,
        **parameters,
        **final_avg_strategies_renamed,
        **final_avg_feps_renamed,
    }
    return combined_dict

def hyper_parameter_tuning(per_class=True):
    # 1) Load configuration and k-fold splits
    config = load_config()
    base_path = config["base_path"]
    random_seeds = [config["dataset"]["random_seed"], config["dataset"]["random_seed2"], config["dataset"]["random_seed3"]] # just removed it to speed up preliminary results
    instance_data_path = os.path.join(base_path, config["paths"]["final"]["rf_dataset"])
    feature_vectors_path = os.path.join(base_path, config["paths"]["final"]["feature_vectors"])
    kfold_path = os.path.join(base_path,config["paths"]["processing"]["data_splits"] )
    startegies_kfold_splits_path = os.path.join(kfold_path, "kfold_splits_strategies.json")
    feps_kfold_splits_path = os.path.join(kfold_path, "kfold_splits_feps.json")
    results_dir = os.path.join(base_path, config["paths"]["results"])
    window_size = config["dataset"]["window_size"]
    k = config["dataset"]["k"]
    final_results_path = os.path.join(results_dir, "experiment_results_perclass.csv") if per_class else os.path.join(results_dir, "hyperparam_results.csv")

    ensure_directory_exists(results_dir)
    kfold_splits_strategies = load_json(startegies_kfold_splits_path)
    kfold_splits_feps = load_json(feps_kfold_splits_path)
    generalization_split = load_json(os.path.join(kfold_path, "generalization_split.json"))

    feature_selection_scenarios = [
        None, # all features
       

    ]
    all_results = []
    target_names = ["strategies", "feps"]
    parameter_path = os.path.join(base_path, config["paths"]["models"]["parameters"])
    parameters_dict = load_json(parameter_path)
    possible_parameters = parameters_dict["parameter_names"]
    # tests the different classifiers
    for model_name in parameters_dict:
        if model_name == "parameter_names":
            continue # The names of the parameters are just saved to get them easily

        parameters = parameters_dict[model_name]
        
        # Tests the different feature selection scenarios
        for fs_indices in feature_selection_scenarios:

            # Tests the different parameter sets
            for parameter_instance in parameters:

                configuration_results_strategies, configuration_results_feps = [],[]
          
                for seed in random_seeds:
                    # fill a dict with all theparameter names in possible Parameters, afterwards update thoise that are in the actual parameter instance

                    param_dict_with_seed = {key: None for key in possible_parameters}
                    param_dict_with_seed.update(parameter_instance)
                    param_dict_with_seed["random_state"] = seed

                    for target_name in target_names:
                        log_message(f"=== Starting experiment with feature_indices={fs_indices} and rf_params={param_dict_with_seed} ===")
                        k_fold_splits = kfold_splits_strategies if target_name == "strategies" else kfold_splits_feps
        
                        for fold_idx in range(k):
                            
                            data_dict = get_fold_data(k_fold_splits, target_name, fold_idx, instance_data_path, feature_vectors_path, fs_indices)
                            if len(data_dict) == 0:
                                continue

                            X_train = data_dict["X_train"]
                            Y_train = data_dict["Y_train"] 
                            X_test = data_dict["X_test"]
                            Y_test_shots = data_dict["Y_test_shots"]
                            Y_test_instances = data_dict["Y_test_instances"] # todo include
                            compare_news_stories = data_dict["compare_news_stories"]
                            test_news_stories = data_dict["test_news_stories"]

                            # 3c) Train, evaluate and save for STRATEGIES
                            clf = train_model(X_train, Y_train, model_name = model_name, model_parameters = param_dict_with_seed, oversample = True, per_class=per_class)

                            metrics_strat = evaluate_fold_on_shots(clf, X_test, test_news_stories, Y_test_shots, compare_news_stories, window_size=window_size)

                            if target_name == "strategies":
                                configuration_results_strategies.append(metrics_strat)
                            else:
                                configuration_results_feps.append(metrics_strat)

                param_dict_with_seed["random_state"] = random_seeds

                # Average metrics over all seeds
                final_avg_strategies = average_metrics(configuration_results_strategies)
                final_avg_feps = average_metrics(configuration_results_feps)
                # instead of saving the dicts in one field i want to save them in different fields
                # probably just append them to experiment_result
                experiment_results = {
                    "model_name": model_name,
                    "feature_selection": fs_indices,
                    "generalization_test": False,
                    "best_model": "No"
                }
                combined_result_row = combine_results(
                    experiment_results,
                    param_dict_with_seed,
                    final_avg_strategies,
                    final_avg_feps
                )

                save_result_append(combined_result_row, final_results_path)
                all_results.append(combined_result_row)

    # get all results from the csv that i saved them in instead of the list
    # After doing all experiments we test the best model on the test data
    for model_name in parameters_dict:
        if model_name == "parameter_names":
            continue # The names of the parameters are just saved to get them easily

        best_features_strategies, best_params_strategies, model_name_strategies, best_features_feps, best_params_feps, model_name_feps= get_best_model(all_results, parameter_names= possible_parameters, model_name = model_name)
        best_params_strategies["random_state"],  best_params_feps["random_state"] = random_seeds[0], random_seeds[0]


        X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(kfold_splits_strategies, instance_data_path, feature_vectors_path, best_features_strategies, strategy_bool = True)
        clf_strategies = train_model(X_train, Y_train, model_name = model_name_strategies, model_parameters = best_params_strategies, oversample = True, per_class=per_class)
        metrics_strat = evaluate_fold_on_shots( clf_strategies, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)
       
        # Evaluate with strategy params on feps just for comparison
        X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(kfold_splits_feps, instance_data_path, feature_vectors_path, best_features_strategies, strategy_bool = False)
        clf_strategies_feptest = train_model(X_train, Y_train, model_name = model_name_strategies, model_parameters = best_params_strategies, oversample = True, per_class=per_class)
        metrics_feps = evaluate_fold_on_shots( clf_strategies_feptest, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)

        # Save Results
        experiment_results = {
                    "model_name": model_name_strategies,
                    "feature_selection": best_features_strategies,
                    "generalization_test": False, 
                    "best_model": "Strategies"
                }
        combined_result_row = combine_results(experiment_results, best_params_strategies, metrics_strat, metrics_feps)
        save_result_append(combined_result_row, final_results_path)
    
        #Generalization test on Strategies
        X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(generalization_split, instance_data_path, feature_vectors_path, best_features_strategies, strategy_bool = True)
        clf_strategies_generalization = train_model(X_train, Y_train, model_name = model_name_strategies, model_parameters = best_params_strategies, oversample = True, per_class=per_class)
        metrics_strat_generalization = evaluate_fold_on_shots( clf_strategies_generalization, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)

        X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(generalization_split, instance_data_path, feature_vectors_path, best_features_strategies, strategy_bool = False)
        clf_strategies_fep_generalization = train_model(X_train, Y_train, model_name = model_name_strategies, model_parameters = best_params_strategies, oversample = True, per_class=per_class)
        metrics_fep_generalization = evaluate_fold_on_shots( clf_strategies_fep_generalization, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)

        experiment_results["generalization_test"] = True
        combined_result_row = combine_results(experiment_results, best_params_strategies, metrics_strat_generalization, metrics_fep_generalization)
        save_result_append(combined_result_row, final_results_path)



        #Same for FEPS
        X_train, Y_train, X_test,_, Y_test, test_news_stories, compare_news_stories = get_test_data(kfold_splits_feps, instance_data_path, feature_vectors_path, best_features_feps, strategy_bool = False)
        clf_feps = train_model(X_train, Y_train, model_name = model_name_feps, model_parameters = best_params_feps, oversample = True, per_class=per_class)
        metrics_feps = evaluate_fold_on_shots( clf_feps, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)

        # also evaluate best fep model on startegies for comparison
        X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(kfold_splits_strategies, instance_data_path, feature_vectors_path, best_features_feps, strategy_bool = True)
        clf_feps_strategietest = train_model(X_train, Y_train, model_name = model_name_feps, model_parameters = best_params_feps, oversample = True, per_class=per_class)
        metrics_strat = evaluate_fold_on_shots( clf_feps_strategietest, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)

        experiment_results = {
                    "model_name": model_name_feps,
                    "feature_selection": best_features_feps,
                    "generalization_test": False, # TODO real generalization test
                    "best_model": "FEP"

                }
        
        combined_result_row = combine_results(experiment_results, best_params_feps, metrics_strat, metrics_feps)
        save_result_append(combined_result_row, final_results_path)

        #Generalization test on FEP
        X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(generalization_split, instance_data_path, feature_vectors_path, best_features_feps, strategy_bool = False)
        clf_feps_generalization = train_model(X_train, Y_train, model_name = model_name_feps, model_parameters = best_params_feps, oversample = True, per_class=per_class)
        metrics_feps_generalization = evaluate_fold_on_shots( clf_feps_generalization, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)

        X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(generalization_split, instance_data_path, feature_vectors_path, best_features_feps, strategy_bool = True)
        clf_feps_strategie_generalization = train_model(X_train, Y_train, model_name = model_name_feps, model_parameters = best_params_feps, oversample = True, per_class=per_class)
        metrics_strat_generalization = evaluate_fold_on_shots( clf_feps_strategie_generalization, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)

        experiment_results["generalization_test"] = True
        combined_result_row = combine_results(experiment_results, best_params_feps, metrics_strat_generalization, metrics_feps_generalization)
        save_result_append(combined_result_row, final_results_path)

        log_message(f"Best strategies model: Features={best_features_strategies}, Params={best_params_strategies}")
        log_message(f"Best feps model: Features={best_features_feps}, Params={best_params_feps}")

        # save best models

        #save_rf_model(clf_strategies, os.path.join(results_dir, "best_strategies_model.pkl"))
        #save_rf_model(clf_feps, os.path.join(results_dir, "best_feps_model.pkl"))
    
    
    log_message(f"All experiments completed. Full results saved to {final_results_path}.")

def feature_experiment(per_class = True):

    config = load_config()
    base_path = config["base_path"]
    random_seeds = [config["dataset"]["random_seed"], config["dataset"]["random_seed2"], config["dataset"]["random_seed3"]] 
    instance_data_path = os.path.join(base_path, config["paths"]["final"]["rf_dataset"])
    feature_vectors_path = os.path.join(base_path, config["paths"]["final"]["feature_vectors"])
    kfold_path = os.path.join(base_path,config["paths"]["processing"]["data_splits"] )
    startegies_kfold_splits_path = os.path.join(kfold_path, "kfold_splits_strategies.json")
    feps_kfold_splits_path = os.path.join(kfold_path, "kfold_splits_feps.json")
    results_dir = os.path.join(base_path, config["paths"]["results"])
    window_size = config["dataset"]["window_size"]
    k = config["dataset"]["k"]
    final_results_path = os.path.join(results_dir, "feature_experiment_results_perclass.csv") if per_class else os.path.join(results_dir, "feature_experiment_results.csv")

    ensure_directory_exists(results_dir)
    kfold_splits_strategies = load_json(startegies_kfold_splits_path)
    kfold_splits_feps = load_json(feps_kfold_splits_path)
    generalization_split = load_json(os.path.join(kfold_path, "generalization_split.json"))

    audio_mulimodal_based = [5,10,14,15,19,21,23,24,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,68,69,70,71,72,73,74,75,76,77,78,79,80,81] 
    image_based = [0,1,2,3,4,6,7,8,9,11,26,27,28,29,30,31,32,33,34,35,36,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]
    audio_based  =  [5,10,24,25,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,68,69,70,71,72,73,74,75,76,77,78,79,80,81]
    multimodal = [12,13,14,15,16,17,18,19,20,21,22,23]
    thirty7_highest_cors = [68, 69, 70, 71, 72, 73, 48, 51, 75, 76, 77, 78, 79, 80, 81, 14, 2, 3, 1, 0, 20, 54, 55, 52, 62, 63, 61, 60, 58, 59, 57, 56, 8, 9, 7, 6, 23]
    corrs_aboive_fifteen_nopos = [ 0,1,2,3,4,6,7,8,9,11,12,14,15, 19, 21,23, 26 , 28, 30,32,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]
    hand_selected = [0,1,2,3,4,6,7,8,9,10, 11, 12, 14,15, 19,21,23, 28, 30, 32, 33, 37,46, 49,50, 70,  76, 77]
    feature_selection_scenarios = [
        None,
        audio_based,
        image_based,
        multimodal,
        audio_mulimodal_based,
        thirty7_highest_cors,
        hand_selected,
    ]
    features_mapping = {
        1: "audio",
        2: "visual",
        3: "multimodal",
        4: "audio_multi",
        5: "correlations",
        6: "hand_selected",
        0: "all"
    }
    all_results = []
    target_names = ["strategies", "feps"]
    parameter_path = os.path.join(base_path, config["paths"]["models"]["parameters"])
    parameters_dict = load_json(parameter_path)
    possible_parameters = parameters_dict["parameter_names"]
    # tests the different classifiers
    for model_name in parameters_dict:
        if model_name == "parameter_names":
            continue # The names of the parameters are just saved to get them easily

        parameters = parameters_dict[model_name]
        
        # Tests the different feature selection scenarios
        for i, fs_indices in enumerate(feature_selection_scenarios):
            features_name = features_mapping[i]

            # Tests the different parameter sets
            for parameter_instance in parameters:
                param_dict_with_seed = {key: None for key in possible_parameters}
                param_dict_with_seed.update(parameter_instance)
                configuration_results_strategies, configuration_results_feps = [],[]
                generalization_config_results_strategies, generalization_config_results_feps = [],[]
                for random_seed in random_seeds:
                    
                    param_dict_with_seed["random_state"] = random_seed
                    parameter_instance = param_dict_with_seed


                    X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(kfold_splits_strategies, instance_data_path, feature_vectors_path, fs_indices, strategy_bool = True)
                    clf_strategies = train_model(X_train, Y_train, model_name = model_name, model_parameters = parameter_instance, oversample = True, per_class=per_class)
                    metrics_strat = evaluate_fold_on_shots( clf_strategies, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)
                    configuration_results_strategies.append(metrics_strat)

                    # Evaluate with strategy params on feps just for comparison
                    X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(kfold_splits_feps, instance_data_path, feature_vectors_path, fs_indices, strategy_bool = False)
                    clf_strategies_feptest = train_model(X_train, Y_train, model_name = model_name, model_parameters = parameter_instance, oversample = True, per_class=per_class)
                    metrics_feps = evaluate_fold_on_shots( clf_strategies_feptest, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)
                    configuration_results_feps.append(metrics_feps)

                    #Generalization test 
                    X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(generalization_split, instance_data_path, feature_vectors_path, fs_indices, strategy_bool = True)
                    clf_strategies_generalization = train_model(X_train, Y_train, model_name = model_name, model_parameters = parameter_instance, oversample = True, per_class=per_class)
                    metrics_strat_generalization = evaluate_fold_on_shots( clf_strategies_generalization, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)
                    generalization_config_results_strategies.append(metrics_strat_generalization)

                    X_train, Y_train, X_test, _, Y_test, test_news_stories, compare_news_stories = get_test_data(generalization_split, instance_data_path, feature_vectors_path, fs_indices, strategy_bool = False)
                    clf_strategies_fep_generalization = train_model(X_train, Y_train, model_name = model_name, model_parameters = parameter_instance, oversample = True, per_class=per_class)
                    metrics_fep_generalization = evaluate_fold_on_shots( clf_strategies_fep_generalization, X_test, test_news_stories, Y_test, compare_news_stories, window_size=window_size)
                    generalization_config_results_feps.append(metrics_fep_generalization)

                metrics_strat = average_metrics(configuration_results_strategies)
                metrics_feps = average_metrics(configuration_results_feps)
                metrics_strat_generalization = average_metrics(generalization_config_results_strategies)
                metrics_fep_generalization = average_metrics(generalization_config_results_feps)

                
                # Save Results
                experiment_results = {
                            "model_name": model_name,
                            "feature_selection": features_name,
                            "generalization_test": False, 
                            "best_model": "Not applicable",
                            "random_state": random_seeds
                        }
                combined_result_row = combine_results(experiment_results, parameter_instance, metrics_strat, metrics_feps)
                configuration_results_strategies.append(metrics_strat)
                save_result_append(combined_result_row, final_results_path)

                experiment_results["generalization_test"] = True
                combined_result_row = combine_results(experiment_results, parameter_instance, metrics_strat_generalization, metrics_fep_generalization)
                save_result_append(combined_result_row, final_results_path)


        
    
    
    log_message(f"All experiments completed. Full results saved to {final_results_path}.")

if __name__ == "__main__":

    """
    FEATURE INDICES for selection
        0-3 = "convnextv2_shot_similarity.pkl": [prev_2,prev_1,next_1,next_2],
        4 = "videoshot_scalemovement.pkl": [ 0.625],   # 0 is extreme closeup, 1 is long shot
        5 = "whisperx_sentiment.pkl": [ 0.375], #neutral= 0.5
        6-9 = "ssv2-vmae_action_shot_similarity.pkl": [prev_2, prev_1, next_1, next_2],
        10 = "llm_evaluative.pkl": [0.75],  # 0 is not evaluative, 1 is evaluative
        11= "shot_density.pkl": [ 0.4651540889451104 ], # 0 is low density, 1 is high density
        12-23 = "speaker_turns_meta.pkl": [active_speaker?, unique_speakers, anchor, reporter, expert, layperson, elite, interview, talkin-head, speech, commenting, voice-over],
        24,25 ="whisperxspeaker_segmentClf.pkl": [malespeaker_present?, femalespeaker_present?],
        26-29 = "face_analysis.pkl": [ average_facesize, unique_faces, region_y, region_x],
        30-36 = "face_emotions_deepface.pkl": [ 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        37-45 = "whisperxspeaker_audioclf.pkl": ['Speech', 'Narration', 'Music', 'Animal', 'Vehicle', 'Siren', 'Other_Sound', 'Silence', 'Artillery_fire'],
        46-51  = "whisperx_ner.pkl": [ eper, lper, loc, org, event, misc  ],
        52-55 = "kinetics-xclip_action_shot_similarity.pkl": [ prev_2, prev_1, next_1, next_2],
        56-59 = "siglip_shot_similarity.pkl": [ prev_2, prev_1, next_1, next_2],
        60-63 ="places_shot_similarity.pkl": [ prev_2, prev_1, next_1, next_2],
        64-67 = "kinetics-vmae_action_shot_similarity.pkl": [prev_2, prev_1, next_1, n
        ext_2]
        68-81 = "whisperx_pos.pkl": [  adj, adp, adv, aux, conj, det, intj, noun, num, part, pron, propn, verb, x],

      

        Features = "expert", "layperson", "elite", "talkin-head", "commenting" are always 0 and thus excluded

        
        target orders for reference
        "feps": [ fragmentation, individualization_of_elite, individualization_of_reporter, fragmentation_splitscreen, emotionalization, individualization_of_layperson],
        "strategies": [ "alternating-shot", "shot-reverse-shot", "cut-in",  "intensify",  "cut-away" ],
        # active speaker unique speaker not included bacause they are audio + image based
        # female speaker removed to ensiure same size of feature sets also has highest cor with male speaker
        primarlily_audio_based = [5,10,14,15,19,21,23,24,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,68,69,70,71,72,73,74,75,76,77,78,79,80,81] n = 37 #has multimodal in tehre as well
        primarlily_image_based = [0,1,2,3,4,6,7,8,9,11,26,27,28,29,30,31,32,33,34,35,36,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67] n = 37
        corrs_above_fifteen =['kinX_act_next2', 'ssv2_act_next2', 'conv_next2', 'conv_prev1', 'ADV', 'X', 'ssv2_act_prev2', 'places_next2', 'voice-over', 'DET', 'places_prev2', 'places_prev1', 'PROPN', 'ADJ', 'PART', 'LOC', 'ADP', 'sig_prev1', 'NOUN', 'ssv2_act_prev1', 'places_next1', 'MISC', 'CONJ', 'ssv2_act_next1', 'sig_next2', 'PRON', 'VERB', 'sig_next1', 'kinX_act_next1', 'kinX_act_prev2', 'conv_prev2', 'conv_next1', 'AUX', 'sig_prev2', 'interview', 'anchor', 'NUM']
        corrs_above_fifteen = [places, kinV, conv, kinx, ssv2, sig]
        ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'LOC', 'MISC', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'VERB', 'X', 
        'anchor', 
        'conv_next1', 'conv_next2', 'conv_prev1', 'conv_prev2', 
        'interview',
          'kinX_act_next1', 'kinX_act_next2', 'kinX_act_prev2',
            'places_next1', 'places_next2', 'places_prev1', 'places_prev2', 
            'sig_next1', 'sig_next2', 'sig_prev1', 'sig_prev2',
              'ssv2_act_next1', 'ssv2_act_next2', 'ssv2_act_prev1', 'ssv2_act_prev2',
                'voice-over']
37
        corrs_aboive-fifteen_nopos = [ 0,1,2,3,4,6,7,8,9,11,12,14,15, 19, 21,23, 26 , 28, 30,32,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]
        #todo corrs above fifteen all
        hand_selected = [0,1,2,3,4,6,7,8,9,10, 11, 12, 14,15, 19,21,23, 28, 30, 32, 33, 37,46, 49,50, 70,  76, 77]

        visual_features =  [0,1,2,3,4,6,7,8,9,11,26,27,28,29,30,31,32,33,34,35,36,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]
        visual features: shotscale, shot densoity, face_analysis, facemotions, shotsim

        audio_features =  [5,10,24,25,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,68,69,70,71,72,73,74,75,76,77,78,79,80,81]
        audio_based: audio_clf, speaker_gender,  textbased: sentiment, evaluativescores, ner, pos 

        multimodal_features = [12,13,14,15,16,17,18,19,20,21,22,23]
        multimodal: active_speaker, unique_speakers, anchor, reporter, expert, layperson, elite, interview, talkin-head, speech, commenting, voice-over


        threshold 0.245
        highest_cors = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'LOC', 'MISC', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'VERB', 'X', 'anchor', 'conv_next1', 'conv_next2', 'conv_prev1', 'conv_prev2', 'interview', 'kinX_act_next1', 'kinX_act_next2', 'kinX_act_prev2', 'places_next1', 'places_next2', 'places_prev1', 'places_prev2', 'sig_next1', 'sig_next2', 'sig_prev1', 'sig_prev2', 'ssv2_act_next1', 'ssv2_act_next2', 'ssv2_act_prev1', 'ssv2_act_prev2', 'voice-over']

        highest_cors = [68, 69, 70, 71, 72, 73, 48, 51, 75, 76, 77, 78, 79, 80, 81, 14, 2, 3, 1, 0, 20, 54, 55, 52, 62, 63, 61, 60, 58, 59, 57, 56, 8, 9, 7, 6, 23]



         """
    

    #main()
    #main(per_class=False) # for comparison
    #feature_experiment(per_class=True)
    feature_experiment(per_class=False) # for comparison

    #hyper_parameter_tuning(per_class=True)
