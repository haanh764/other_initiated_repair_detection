# "Mm, Wat?" Detecting Other-initiated Repair Requests in Dialogue
https://arxiv.org/abs/2510.24628
🗣️ **Other-Initiated Repair in Spoken Dialogues** | 🎯 **Multimodal Other Repair Initiation Detection**  |  🤖 **Task-oriented Dialogues**

## 📋 Overview

This repository contains the code for the upcoming paper:

Ngo, A., Rollet, N., Pelachaud, C., & Clavel, C. (n.d.). “Mm, Wat?” Detecting Other-initiated Repair Requests in Dialogue. Accepted to Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing.

## 🚀 Installation
### 📋 Prerequisites
- Python 3.11 or higher

### ⚡ Setup

1. **📥 Clone the repository:**
```
git clone https://github.com/haanh764/other_initiated_repair_detection.git
cd other_initiated_repair_detection
```

2. **🏠 Create virtual environment:**
```
python -m venv venv
source venv/bin/activate
```

3. **📦 Install dependencies:**
```
pip install -r requirements.txt
```

### 📁Data folder structure
The data folder should be structured as follows:
```
data/
├── 📊 combined_handcrafted_features.csv          # Main dataset with handcrafted linguistic/prosodic features
└── 🎵 audio/                                     # Audio files directory
    ├── pair1_synced_ppA.wav
    ├── pair1_synced_ppB.wav
    └── ...
```

### 🎯 Quick Start
There are 3 options to run experiments:

**Option 1. Run Predefined Experiments**
```
# Text-only experiment (using pretrained embeddings)
python experiment_runner.py text

# Audio-only experiment (using pretrained embeddings)  
python experiment_runner.py audio

# Handcrafted linguistic-only experiment 
python experiment_runner.py linguistic

# Handcrafted prosodic-only experiment
python experiment_runner.py prosodic

# Multimodal text + audio
python experiment_runner.py text_audio

# Multimodal linguistic + prosodic
python experiment_runner.py handcrafted

# Full multimodal (ours)
python experiment_runner.py full

# List all available predefined experiments
python experiment_runner.py
```

**Option 2. Use YAML Config**
```
# List available experiment configs in experiment_configs.yaml
python config_loader.py experiment_configs.yaml --list

# Run experiment
python config_loader.py experiment_configs.yaml "experiment_name"

# It's possible to override the default parameters
python config_loader.py experiment_configs.yaml "experiment_name" batch_size="new_batch_size" learning_rate="new_learning_rate"
```

**Option 3. Use CLI**

Run a custom experiment using command line arguments. For example:

```
python experiment_runner.py \
  --data_file data/combined_handcrafted_features.csv \
  --use_text \
  --model_name pdelobelle/robbert-v2-dutch-base \
  --context_mode both \
  --use_kfold \
  --experiment_name my_experiment
```

### 📝 Citation

If you find this code useful for your research, please consider citing our paper:

```
```

