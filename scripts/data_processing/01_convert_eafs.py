import xml.etree.ElementTree as ET
import json
import sys
import os
# Add the `scripts` folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import Counter
import pandas as pd
from utils import load_config, save_json, ensure_directory_exists, log_message



def overlap_ratio(segment1, segment2):
    overlap_duration = max(
        0,
        min(segment1["end"], segment2["end"]) - max(segment1["start"], segment2["start"]),
    )
    duration1 = segment1["end"] - segment1["start"]
    duration2 = segment2["end"] - segment2["start"]

    if duration1 <= 0 or duration2 <= 0:
        return 0

    # Compute the average of overlap ratios for both segments
    ratio1 = overlap_duration / duration1
    ratio2 = overlap_duration / duration2
    return (ratio1 + ratio2) / 2  # Symmetric overlap ratio


## Change this function to add more tiers to the JSON output
def import_timelines_from_eaf(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    if xfile == 'NUw4Q3lLWUZWdGc=.eaf':
        print("Processing file", xmlfile)
    # time spans
    timeslots = {}
    for timeslot in root.findall("TIME_ORDER/TIME_SLOT"):
        timeslots[timeslot.attrib["TIME_SLOT_ID"]] = timeslot.attrib

    # Timeline structure
    timelines = {"video_fn": os.path.splitext(os.path.basename(root.find("./HEADER/MEDIA_DESCRIPTOR[@MEDIA_URL]").get("MEDIA_URL")))[0],
                 "shots": [],
                 "story": [], "FEP": [], "Strategy": [], }


    channel = videofn_to_channel[timelines["video_fn"]]
    if channel not in channel_counters:
        channel_counters[channel] = {
            "story": Counter(),
            "FEP": Counter(),
            "Strategy": Counter(),
        }

    # Process Tiers
    for timeline in root.findall("TIER"):
        tier_id = timeline.attrib["TIER_ID"]
        
        # Category and label for each annotation based on tier prefix
        if tier_id.startswith("story"):
            tier_name = tier_id
            label = tier_id.split(":", 1)[-1].lower()
            category = "story"
        elif tier_id.startswith("FEP"):
            tier_name = fep_map.get(tier_id.lower())
            label = fep_map.get(tier_id.lower())
            category = "FEP"
            if "fragmentation" in tier_id.lower():
                print("Mistake in fragmentation tier", xmlfile, tier_id.lower())
                category = "Strategy" # fragementation is a strategy, sometimes falsly annotated as FEP
                tier_name = strategy_map.get(tier_id.lower())
                label = strategy_map.get(tier_id.lower())
            if tier_name is None:
                print("Mistake in FEP tier", xmlfile, tier_id.lower())
        elif tier_id.startswith("Strategy") or tier_id.startswith("strategy"):
            tier_name = strategy_map.get(tier_id.lower())
            label = strategy_map.get(tier_id.lower())
            if label is None or label == 'None':
                print("Mistake in strategy tier", xmlfile, tier_id.lower())
            category = "Strategy"
            if tier_name is None:
                print("Mistake in strategy tier", xmlfile, tier_id.lower())
            elif tier_name in ["dramatization", "evaluative_talk_to_audience"]:
                print("Skipping dramatization or evaluative tier due to lack of samples", xmlfile, tier_id.lower())
                continue
        elif tier_id.startswith("shots"):
            tier_name = tier_id
            category = "shots"
        else:
            continue

        # Annotations for each tier
        timeline_segments = []
        for annotation in timeline.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
            start_time = timeslots[annotation.attrib["TIME_SLOT_REF1"]]["TIME_VALUE"]
            end_time = timeslots[annotation.attrib["TIME_SLOT_REF2"]]["TIME_VALUE"]

            if category == "shots" or category == "speaker_turn":
                label = annotation.find("ANNOTATION_VALUE").text
            else:
                label_counters[category][label] += 1
                channel_counters[channel][category][label] += 1

            timeline_segments.append(
                {"start": int(start_time) / 1000, "end": int(end_time) / 1000, "label": label}
            )

        timelines[category].append({"name": tier_name, "segments": timeline_segments})

    ## Check if strategies and feps overlap with any story annotations (Every FEP or Strategy should be within at least one story)
    for fep in timelines["FEP"]:
        for segment in fep["segments"]:
            in_story = False
            for story in timelines["story"]:
                for story_segment in story["segments"]:
                    if overlap_ratio(segment, story_segment) > 0.3:
                        in_story = True
                        break
            
            if not in_story:
                print("FEP not in story", xmlfile, fep["name"], segment, ", Max overlap ratio", max([overlap_ratio(segment, story_segment) for story_segment in story["segments"]]))

    
    for strategy in timelines["Strategy"]:
        for segment in strategy["segments"]:
            in_story = False
            for story in timelines["story"]:
                for story_segment in story["segments"]:
                    if overlap_ratio(segment, story_segment) > 0.3:
                        in_story = True
                        break
            
            if not in_story:
                print("Strategy not in story", xmlfile, strategy["name"], segment, ", Max overlap ratio", max([overlap_ratio(segment, story_segment) for story_segment in story["segments"]]))

    return timelines

# Directory paths
config = load_config()
annot_dir = os.path.join(config['base_path'], config['paths']['raw']['targets'])
result_dir= os.path.join(config['base_path'], config['paths']['processing']['converted_targets'])


## Original Mapping
"""
strategy_map = {
    "strategy:fragmentation": "fragmentation",
    "strategy:fragmentation-inserted-shots": "fragmentation",
    "strategy:fragmentation-splitscreen-dissonance": "fragmentation_splitscreen",
    "strategy:fragmentation-splitscreen": "fragmentation_splitscreen",
    "strategy:individualization-of-elite": "individualization_of_elite",
    "strategy:individualization-elite": "individualization_of_elite",
    "strategy:individualization-of-layperson": "individualization_of_layperson",
    "strategy:individualization-layperson": "individualization_of_layperson",
    "strategy:emotionalization-v2": "emotionalization_v2",  ## sentiment
    "strategy:emotionalization-v1": "emotionalization_v1",
    "strategy:dramatization": "dramatization",
    "strategy:individualization-of-reporter": "individualization_of_reporter",
    "strategy:individualization-reporter": "individualization_of_reporter",
    "strategy:evaluative-talk-to-audience": "evaluative_talk_to_audience",
    "strategy:evaluative-talk-audience": "evaluative_talk_to_audience",
}

fep_map = {
    "fep:intensify-v1": "intensify-v1",
    "fep:alternating-shota": "alternating-shota",
    "fep:alternating-shotsa": "alternating-shota",
    "fep:alternating-shot-a": "alternating-shota",
    "fep:intensify-v3": "intensify-v3",
    "fep:cut-in": "cut-in",
    "fep:shot-reverse-shot": "shot-reverse-shot",
    "fep:cut-away-v1": "cut-away-v1",
    "fep:cut-away_v1": "cut-away-v1",
    "fep:cut-away_v2": "cut-away-v2",
    "fep:cut-away-v2": "cut-away-v2",
    "fep:alternating-shotc": "alternating-shotc",
    "fep:alternating-shot-c": "alternating-shotc",
    "fep:alternating-shot-d": "alternating-shotd",
}
"""

# Mapping to reduce classes
strategy_map = {
    "strategy:fragmentation": "fragmentation",
    "strategy:fragmentation-inserted-shots": "fragmentation",
    "strategy:fragmentation-splitscreen-dissonance": "fragmentation_splitscreen",
    "strategy:fragmentation-splitscreen": "fragmentation_splitscreen",
    "strategy:individualization-of-elite": "individualization_of_elite",
    "strategy:individualization-elite": "individualization_of_elite",
    "strategy:individualization-of-layperson": "individualization_of_layperson",
    "strategy:individualization-layperson": "individualization_of_layperson",
    "strategy:emotionalization-v2": "emotionalization",  ## sentiment
    "strategy:emotionalization-v1": "emotionalization",
    "strategy:dramatization": "dramatization",
    "strategy:individualization-of-reporter": "individualization_of_reporter",
    "strategy:individualization-reporter": "individualization_of_reporter",
    "strategy:evaluative-talk-to-audience": "evaluative_talk_to_audience",
    "strategy:evaluative-talk-audience": "evaluative_talk_to_audience",
    "fep:fragmentation": "fragmentation", # fragementation once incorrectly annotated as FEP
    "fep:fragmentation-splitscreen": "fragmentation_splitscreen", # fragementation incorrectly annotated as FEP
}

fep_map = {
    "fep:intensify-v1": "intensify",
    "fep:alternating-shota": "alternating-shot",
    "fep:alternating-shotsa": "alternating-shot",
    "fep:alternating-shot-a": "alternating-shot",
    "fep:intensify-v3": "intensify",
    "fep:cut-in": "cut-in",
    "fep:shot-reverse-shot": "shot-reverse-shot",
    "fep:cut-away-v1": "cut-away",
    "fep:cut-away_v1": "cut-away",
    "fep:cut-away_v2": "cut-away",
    "fep:cut-away-v2": "cut-away",
    "fep:alternating-shotc": "alternating-shot",
    "fep:alternating-shot-c": "alternating-shot",
    "fep:alternating-shot-d": "alternating-shot",
}

# Counters
label_counters = {
    "story": Counter(),
    "FEP": Counter(),
    "Strategy": Counter()
}

## Channel counters
channel_counters = {}

## One large json file for all annotations
## Structure of the JSON file:
## {
##     "video_filename": {
##          "shots": [
##              {"start": 0.0, "end": 10.0, "label": 0},
##              {"start": 10.0, "end": 20.0, "label": 1},
##              ...
##          ],
##          "speaker_turns": [
##              {"start": 0.0, "end": 10.0, "label": "speaker1"},
##              {"start": 10.0, "end": 20.0, "label": "speaker2"},
##              ...
##          ],
##          "stories": [
##              {
##                  "name": story_name, 
##                  "segments": [
##                      {"start": 0.0, "end": 10.0, "label": "label1"},
##                      {"start": 10.0, "end": 20.0, "label": "label2"},
##                      ...
##                  ]
##              },
##              ...
##          ],
##          "feps": {
##              fep_name: [
##                  {"start": 0.0, "end": 10.0, "label": "label1"},
##                  {"start": 10.0, "end": 20.0, "label": "label2"},
##                  ...
##              ],
##              ...
##          },
##          "strategies": {
##              strategy_name: [
##                  {"start": 0.0, "end": 10.0, "label": "label1"},
##                  {"start": 10.0, "end": 20.0, "label": "label2"},
##                  ...
##              ],
##              ...
##          },
##          "evaluative-talk": [
##              {"start": 0.0, "end": 10.0, "label": "label1"},
##              {"start": 10.0, "end": 20.0, "label": "label2"},
##              ...
##          ]
##     },
##     ...
## }
data_map = {}
## video name to channel mapping
videofn_to_channel = json.load(open("data/manual_map.json"))

channel_statistics = {}

for xdir in os.listdir(annot_dir):
    dir_path = os.path.join(annot_dir, xdir)
    for xfile in  os.listdir(dir_path):
        if xfile.endswith(".eaf"):
            # Generate JSON output with annotations per file
            timelines = import_timelines_from_eaf(os.path.join(annot_dir, xdir, xfile))
            json.dump(timelines, open(os.path.join(result_dir, xdir, xfile.replace(".eaf", ".json")), "w"), indent=4)
            
            if timelines["video_fn"] not in data_map:
                data_map[timelines["video_fn"]] = {
                    "shots": timelines["shots"][0]["segments"],
                    "stories": timelines["story"],
                }
                
                data_map[timelines["video_fn"]]["feps"] = {}
                for fep in timelines["FEP"]:
                    data_map[timelines["video_fn"]]["feps"][fep["name"]] = fep["segments"]
                data_map[timelines["video_fn"]]["strategies"] = {}
                for strategy in timelines["Strategy"]:
                    data_map[timelines["video_fn"]]["strategies"][strategy["name"]] = strategy["segments"]
               
            
## Save the data map to a JSON file
json.dump(data_map, open("annotations.json", "w"), indent=4)

# Statistics of manual annotations
print("Overall Statistics for Unique Labels and Annotation Counts:")
for category, counter in label_counters.items():
    unique_labels_count = len(counter)
    total_annotations = sum(counter.values())
    print(f"Category '{category}':")
    print(f"  Total Unique Labels: {unique_labels_count}")
    print(f"  Total Annotations: {total_annotations}")
    print(f"  Label Counts: {dict(counter)}\n")


## Save the channel statistics to a JSON file
json.dump(channel_counters, open("channel_statistics.json", "w"), indent=4)
