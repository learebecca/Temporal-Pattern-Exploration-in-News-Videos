import sys
import os

# Add the `scripts` folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import cv2
from utils import load_config, ensure_directory_exists, log_message



def extract_frame(video_path, time_sec, output_path):
    """Extract a frame at a specific time from a video and save it as an image."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()

def find_video_file(base_filename, channel_dir):
    # Common video extensions
    possible_exts = [".mp4", ".m4v", ".mkv", ".mov"]
    for ext in possible_exts:
        candidate = os.path.join(channel_dir, base_filename + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def process_video(target_json_path, video_source_dir, output_base_dir):
    with open(target_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    video_filename = data['video_fn']  # e.g., "myvideo" (no extension)
    channel = os.path.basename(os.path.dirname(target_json_path))
    channel_dir = os.path.join(video_source_dir, channel)

    # Attempt to find an actual file with one of the common extensions
    video_path = find_video_file(video_filename, channel_dir)
    if not video_path:
        log_message(f"Video file not found for base name {video_filename} in {channel_dir}")
        return

    shots = data['shots'][0]['segments']

    video_output_dir = os.path.join(output_base_dir, channel, os.path.splitext(video_filename)[0])
    ensure_directory_exists(video_output_dir)
    log_message(f"Processing video {video_filename} from channel {channel}")
    
    for shot in shots:
        shot_label = shot['label']
        shot_time = (shot['start'] + shot['end']) / 2  # Middle frame of the shot
        output_image_path = os.path.join(video_output_dir, f"shot_{shot_label}.jpg")
        extract_frame(video_path, shot_time, output_image_path)

def main():
    """Main function to process all videos and extract frames."""
    # Load configuration
    config = load_config()
    converted_targets_dir = config['paths']['processing']['converted_targets']
    videos_dir = config['paths']['raw']['videos']
    shot_frames_dir = config['paths']['processing']['shot_frames']

    for channel in os.listdir(converted_targets_dir):
        channel_dir = os.path.join(converted_targets_dir, channel)
        if os.path.isdir(channel_dir):
            log_message(f"Processing channel {channel}")
            for json_file in os.listdir(channel_dir):
                if json_file.endswith(".json"):
                    json_path = os.path.join(channel_dir, json_file)
                    process_video(json_path, videos_dir, shot_frames_dir)

if __name__ == "__main__":
    main()
