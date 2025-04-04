import sys
import os

# Add the `scripts` folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil

from utils import load_config, load_json, ensure_directory_exists, log_message


def sort_vlm_data():


    config = load_config() 
    vlm_base_dir = config['paths']['final']['vlm_data']  
    splits_path = config['paths']['processing']['data_splits']
    
    for split in os.listdir(splits_path):
        split_name = split.strip(".json")
        log_message(f"Processing split {split_name}")
        output_dir = os.path.join(vlm_base_dir,  f"test_{split_name}")
        split_path = os.path.join(splits_path,  split)
        ensure_directory_exists(output_dir)
        data = load_json(split_path)
        if type(data) == list: #for k-fold just gte 1 split
            data = data[0]

        test_files = data.get("test", [])

        for test_file in test_files:
            channel = os.path.basename(os.path.dirname(test_file))  # e.g. "BildTV"
            filename = os.path.basename(test_file)                  # e.g. "LXVacEl5UFdzRzQ=_story01.json"
            # Where we expect it to exist in the VLM directory:
            source_path = os.path.join(vlm_base_dir, channel, filename)
            if not os.path.exists(source_path):
                log_message(f"WARNING: {source_path} not found, skipping.")
                return
            ensure_directory_exists(output_dir)

            dest_path = os.path.join(output_dir, filename)
            # Copy or move. Here, we'll copy:
            shutil.copy2(source_path, dest_path)
            log_message(f"Copied {source_path} -> {dest_path}")
         
    log_message("Done sorting VLM data into test_vlm and generalization_test_vlm.")

if __name__ == "__main__":
    sort_vlm_data()
