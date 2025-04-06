# Temporal-Pattern-Exploration-in-News-Videos

**Temporal-Pattern-Exploration-in-News-Videos** is the official codebase accompanying the master's thesis *"Temporal Pattern Exploration in News Videos"*. This project investigates how film editing patterns and narrative strategies shape the construction and perception of news content. It presents a pipeline for detecting and analyzing these temporal patterns using a combination of classical machine learning methods (e.g., Random Forests) and vision-language models (VLMs). The approach is built on a curated dataset of German news videos and is designed to support both the exploration of multimodal feature importance and the automated detection of stylistic editorial choices. This repository facilitates reproducible experiments, structured data processing, and scalable pattern analysis in the news media domain.

---

## Getting Started

Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed. Then run:

```bash
conda create --name TemporalPatterns python=3.11
conda activate TemporalPatterns
pip install -r requirements.txt
```

---

## Usage Example

To convert raw ELAN annotations to JSON and aggregate features per shot:

```bash
python scripts/data_processing/01_convert_eafs.py
python scripts/data_processing/02_aggregate_per_shot.py
```

To run classifier experiments:

```bash
python scripts/model_training/train_eval_classifiers.py
```

---

## Repo Structure

```
TemporalPatternDetection/
├── thesis/                        # Thesis-related documents
├── config.yaml                    # Centralized configuration file with paths and settings
├── encoding_params.json           # Target encoding order

├── data/                          # All data-related folders and files
│   ├── manual_map.json            # Maps video names and channels
│   ├── raw/                       # Raw data (unaltered)
│   │   ├── targets/               # Original ELAN files, organized by news source
│   │   ├── feature_pickles/       # Extracted raw features, by news source
│   │   ├── videos/                # Raw videos, by news source
│   ├── processing/                # Intermediate data
│   │   ├── converted_targets/     # Targets converted to JSON
│   │   ├── aggregated_shots/      # Aggregated targets/features per shot
│   │   ├── data_splits/           # Three data splits used in the experiments
│   │   ├── shot_frames/           # Middle frames of each shot
│   ├── final/                     # Finalized data for training and analysis
│   │   ├── feature_vectors/       # Normalized feature vectors + targets
│   │   ├── feature_vectors_unscaled/ # Unnormalized feature vectors
│   │   ├── rf_dataset/            # Classifier-ready data
│   │   ├── vlm_data/              # VLM-ready data
│   ├── model_outputs/vlm/         # VLM predictions

├── scripts/                       # All processing scripts and notebooks
│   ├── data_processing/           # Data preprocessing scripts
│   │   ├── 01_convert_eafs.py
│   │   ├── 02_aggregate_per_shot.py
│   │   ├── 02_get_frame_per_shot.py
│   │   ├── 03_create_splits.py
│   │   ├── 04_create_feature_vectors.py
│   │   ├── 05_create_feature_vectors.py
│   │   ├── 06_aggregate_instances.py
│   │   ├── 07_create_vlm_data.py
│   │   ├── 08_split_vlm_data.py
│   │   └── ...                   # Notebooks for analysis, plotting, etc.
│   ├── model_training/           # Model training and evaluation
│       ├── make_model_parameters.py
│       ├── train_eval_classifiers.py
│       ├── vlm_inference.py
│       ├── vlm_scoring.py
│       └── ...

├── results/                      # Result CSVs and figures
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## Citation

If you use this codebase in your research, please cite:

```
Lea Reinhart (2025). Temporal Pattern Exploration in News Videos. Master's Thesis, Leibniz University Hannover.
```

Full thesis available in the [`thesis/`](thesis/) folder.

---
