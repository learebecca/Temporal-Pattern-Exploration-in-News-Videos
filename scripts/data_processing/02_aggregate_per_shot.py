import sys
import os

# Add the `scripts` folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_config, load_pickle, load_json, save_json, ensure_directory_exists, log_message
from feature_processing import process_feature

DAMAGED_PKLS = []

def process_shot(shot, feature_data_dict):
    """
    Given a single shot and a dictionary of all loaded feature data,
    apply the relevant feature-processor to update the shot dictionary.
    
    feature_data_dict is something like:
    {
        "whisperx_ner.pkl": <data>,
        "whisperx_pos.pkl": <data>,
        "whisperx_sentiment.pkl": <data>,
        ...
    }
    """

    # Initialize  shot dict
    shot_dict = {
        "start":          shot['start'],
        "end":            shot['end'],
        "strategies":     [],
        "stories":        [],
        "feps":           [],
       
    }

    for feature_name in feature_data_dict:
        # Initialize the feature-specific data
        # feature_name = feature_name.replace(".pkl", "")
        shot_dict[feature_name] = []

    # Run each feature's processor if it exists
    for feature_name, data in feature_data_dict.items():
        process_feature(shot_dict, feature_name, data)


    return shot_dict


def process_targets(shot_dict, target_data):
    """
    Process target data (Strategy, story, FEP, etc.) and update shot_dict accordingly.
    Original logic for applying targets to the shot can go here.
    """
    shot_start = shot_dict['start']
    shot_end   = shot_dict['end']

    # If you keep the same approach as before:
    strategy_segments = target_data.get('Strategy', [])
    story_segments    = target_data.get('story', [])
    fep_segments      = target_data.get('FEP', [])

    # For strategies, stories, and feps
    for segments, key in zip([strategy_segments, story_segments, fep_segments],
                             ['strategies', 'stories', 'feps']):
        for segment in segments:
            name = segment['name']
            for item in segment['segments']:
                if item['start'] < shot_end and item['end'] > shot_start:
                    if name not in shot_dict[key]:
                        shot_dict[key].append(name)

    return shot_dict

def load_feature(feature_path, feature_name, feature_data_dict):

    if not os.path.exists(feature_path):
            log_message(f"Feature file {feature_path} not found. Skipping.")
            return
    try: 
        loaded_data = load_pickle(feature_path) 
        feature_data_dict[feature_name] = loaded_data
    except Exception as e:
        log_message(f"Error loading feature {feature_name} from {feature_path}: {e}")
        DAMAGED_PKLS.append(feature_path)

def associate_stories(shot_dict, stories_dict, i):
    # Associate shot with stories
    for story_name in shot_dict['stories']:
        if story_name not in stories_dict:
            stories_dict[story_name] = {}
        stories_dict[story_name][i] = shot_dict

def aggregate_targets_features(channel,
                               video_id,
                               features_base_path,
                               targets_base_path,
                               output_base_path,
                               feature_config):
    """
    Aggregate targets and features shot-by-shot for each video in a modular way.
    """
    log_message(f"Processing video {video_id} from {channel}")

    
    target_file_path = os.path.join(targets_base_path, channel, f'{video_id}.json')
    if not os.path.exists(target_file_path):
        log_message(f"Target file for {video_id} not found. Skipping.")
        return
    target_data = load_json(target_file_path)

    # The list of shots from the target
    shots = target_data['shots'][0]['segments']


    feature_data_dict = {}
    all_feature_filenames = (
        feature_config.get("text_based", []) +
        feature_config.get("image_based", []) +
        feature_config.get("audio_based", []) +
        feature_config.get("multimodal", [])
    )

    for feature_name in all_feature_filenames:
        # Path to feature file
        feature_path = os.path.join(features_base_path, channel, video_id, feature_name)
        load_feature(feature_path, feature_name, feature_data_dict)


    stories_dict = {}

    for i, shot in enumerate(shots):
        # load features, targets and match to stories
        shot_dict = process_shot(shot, feature_data_dict)
        shot_dict = process_targets(shot_dict, target_data)
        associate_stories(shot_dict, stories_dict, i)
        

    output_channel_path = os.path.join(output_base_path, channel)
    ensure_directory_exists(output_channel_path)

    for story_name, shots_data in stories_dict.items():
        story_id = story_name.split(":")[0]
        output_file_path = os.path.join(output_channel_path, f'{video_id}_{story_id}.json')
        save_json(shots_data, output_file_path)


def main():
    # 1. Load configuration
    config = load_config()

    # 2. Paths from config
    features_base_path   = os.path.join(config['base_path'], config['paths']['raw']['feature_pickles'])
    targets_base_path    = os.path.join(config['base_path'], config['paths']['processing']['converted_targets'])
    output_base_path     = os.path.join(config['base_path'], config['paths']['processing']['aggregated_shots'])
    feature_json_path    = os.path.join(config['base_path'], config['paths']['processing']['feature_json'])
    feature_dict = load_json(feature_json_path)
    ensure_directory_exists(output_base_path)

    # 3. Iterate over channels and videos
    for channel in os.listdir(features_base_path):
        if channel != "Test":
            channel_path = os.path.join(features_base_path, channel)
            if os.path.isdir(channel_path):
                for file_name in os.listdir(channel_path):
                    aggregate_targets_features(
                        channel,
                        file_name,
                        features_base_path,
                        targets_base_path,
                        output_base_path,
                        feature_dict
                    )
    if DAMAGED_PKLS:
        log_message(f"Damaged PKL files: {DAMAGED_PKLS}")

if __name__ == "__main__":
    main()
