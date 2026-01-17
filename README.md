# VHSR - Vietnamese Hate Speech Recognition

Real-time audio-based hate speech detection system for Vietnamese language, combining Automatic Speech Recognition (ASR) with text classification.

## Overview

VHSR is an end-to-end pipeline that:
1. **Transcribes** Vietnamese speech to text using ASR models
2. **Classifies** the transcribed text into two categories:
   - **Clean**: Normal, non-offensive speech
   - **Hate**: Hate speech, toxic, or harmful content

## Features

- **Real-time Processing**: Live audio recording with Voice Activity Detection (VAD)
- **Robust Noise Calibration**: Adaptive silence threshold based on ambient noise
- **Interactive Jupyter Widget**: User-friendly interface for recording and classification
- **High Accuracy**: Uses state-of-the-art models (ChunkFormer ASR + mDeBERTa classifier)
- **GPU Accelerated**: Optimized for CUDA-enabled devices

## Project Structure

```
VHSR/
├── realtime_asr_classification.ipynb   # Main interactive notebook
├── DeBERTa_Train_Colab.py             # Training script for mDeBERTa classifier
├── preprocess_data.py                  # Dataset preprocessing and tokenization
├── realtime_transcribe.py              # Real-time ASR using Faster Whisper
├── test_audio_speech.py                # ASR testing utilities
├── test_deberta_classification.py      # Classifier testing
├── CleanSTT.csv                        # Training dataset
├── model/                              # Trained classifier models
│   ├── checkpoint-25000/              # Best model checkpoint
│   └── checkpoint-40000/
├── preprocessed_data/                  # Tokenized datasets
│   ├── train/
│   ├── validation/
│   └── test/
└── requirements.txt                    # Python dependencies
```

## Models

### ASR (Automatic Speech Recognition)
- **Model**: [khanhld/chunkformer-ctc-large-vie](https://huggingface.co/khanhld/chunkformer-ctc-large-vie)
- **Architecture**: ChunkFormer with CTC loss
- **Purpose**: Converts Vietnamese audio to text

### Text Classification
- **Model**: microsoft/mdeberta-v3-base (fine-tuned)
- **Architecture**: Multilingual DeBERTa v3
- **Classes**: 2 classes (clean, hate)
- **Training**: Fine-tuned on Vietnamese hate speech dataset with class weights
- **Class Balance**: ~94% clean, ~6% hate (handled with weighted loss)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- PortAudio (for PyAudio)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install PortAudio (for audio recording)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev
```

**MacOS:**
```bash
brew install portaudio
```

**Windows:**
Download from [PortAudio website](http://www.portaudio.com/)

## Quick Start

### 1. Interactive Real-time Classification (Recommended)

Open and run the Jupyter notebook:

```bash
jupyter notebook realtime_asr_classification.ipynb
```

Follow the notebook steps:
1. Load models
2. Calibrate noise (stay silent for 3 seconds)
3. Start recording and speaking
4. View real-time transcription and classification results

### 2. Command-line Real-time Transcription

```bash
python realtime_transcribe.py
```

### 3. Test Classification on Text

```python
from test_deberta_classification import classify_text

text = "Xin chào, hôm nay trời đẹp quá"
result = classify_text(text)
print(result)
# Output: {'predicted_label': 'clean', 'confidence': 0.99, ...}
```

## Training Your Own Model

### 1. Prepare Dataset

Your dataset should be a CSV file with columns:
- `text`: Vietnamese texthate']

Example:
```csv
text,label
"Xin chào bạn",clean
"Mày ngu như cặc",hat
"Mày ngu quá",offensive
"Đập chết nó đi",hate
```

### 2. Preprocess Data

```bash
python preprocess_data.py
```

This will:
- Tokenize the dataset
- Split into train/validation/test sets
- Save to `preprocessed_data/` directory

### 3. Train the Model

```bash
python DeBERTa_Train_Colab.py
```

Configuration options in the script:
- `MODEL_NAME`: Base model to fine-tune
- `OUTPUT_DIR`: Where to save checkpoints
- `BATCH_SIZE`: Adjust based on GPU memory
- `NUM_EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Learning rate for optimization

### 4. Use the Trained Model

Update the model path in `realtime_asr_classification.ipynb`:

```python
config.CLASSIFIER_MODEL_PATH = "path/to/your/checkpoint"
```

## Usage Examples

### Interactive Widget

```python
from realtime_asr_classification import RealtimeASRClassifierWidget

# Initialize
pipeline = RealtimeASRClassifierWidget(config, calibrator)

# Display widget
pipeline.display()

# Start recording (use GUI button or programmatically)
pipeline.start()

# Get results
results = pipeline.get_results()
```

### Programmatic Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "model/checkpoint-25000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Classify
text = "Your Vietnamese text here"128)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=-1)
predicted_class = torch.argmax(probs, dim=-1).item()

# Get label
label = model.config.id2label[predicted_class]  # 'clean' or 'hate'
confidence = probs[0][predicted_class]
predicted_class = torch.argmax(probs, dim=-1).item()
```

## Configuration

### Audio Settings

In `realtime_asr_classification.ipynb`, modify the `Config` class:

```python
@dataclass
class Config:
    SAMPLE_RATE: int = 16000              # Audio sample rate
    CHUNK_SIZE: int = 1024                # Audio chunk size
    MAX_RECORDING_DURATION: float = 30.0  # Max seconds per segment
    SILENCE_DURATION: float = 2.0         # Silence threshold (seconds)
    NOISE_MULTIPLIER: float = 2.5         # Noise threshold multiplier
```

### Model Paths

```python
ASR_MODEL_PATH = "khanhld/chunkformer-ctc-large-vie"
CLASSIFIER_MODEL_PATH = "model/checkpoint-25000"
```

## Troubleshooting

### Audio Input Issues

**No audio detected:**
```python
# List available audio devices
calibrator.list_audio_devices()

# Set specific device
config.INPUT_DEVICE_INDEX = 0  # Use device index from list
```

**High background noise:**
```python
# Increase noise calibration duration
config.NOISE_CALIBRATION_DURATION = 5.0

# Adjust noise multiplier
config.NOISE_MULTIPLIER = 3.0
```

### CUDA Out of Memory

Reduce batch size or use CPU:
```python
device = "cpu"
# Or reduce batch size in training:
BATCH_SIZE = 8  # Instead of 16
```

### Model Loading Issues

Ensure models are downloaded:
```bash
# ASR model will auto-download on first use
# Classifier model should be in model/checkpoint-XXXXX/
```

## Dataset

The training dataset (`CleanSTT.csv`) is compiled from two Vietnamese hate speech datasets:

### Source Datasets

1. **ViHSD** ([sonlam1102/vihsd](https://huggingface.co/datasets/sonlam1102/vihsd))
   - Vietnamese Hate Speech Detection dataset
   - Multi-split dataset with labeled Vietnamese text
   - Labels: 0 (clean), 1+ (offensive/hate) - mapped to binary

2. **VOZ-HSD** ([tarudesu/VOZ-HSD](https://huggingface.co/datasets/tarudesu/VOZ-HSD))
   - VOZ Forum Hate Speech Dataset
   - Community-sourced Vietnamese text
   - Labels: 0 (clean), 1 (hate)

3. **Crossmodal-3600** ([Google Crossmodal-3600](https://google.github.io/crossmodal-3600/))
   - ~40K Vietnamese captions from multilingual image dataset
   - Used to reduce class4M (after deduplication)
- **clean**: ~6.64M samples (~94%) - includes ViHSD, VOZ-HSD clean samples + 40K Crossmodal-3600alanced training data

### Final Dataset Statistics

- **Total samples**: ~7.0M (after deduplication)
- **clean**: ~6.6M samples (~94%)
- **hate**: ~428K samples (~6%)

### Class Imbalance Handling

- Weighted Cross-Entropy Loss (weight ratio 1:6 for clean:hate)
- Bias initialization based on prior probability
- Stratified train/validation/test split

### Data Preprocessing

The raw datasets undergo extensive preprocessing (`ExtractCleanCSV.py`):
- Teencode/slang normalization (e.g., "k" → "không", "dm" → "địt mẹ")
- Emoji removal
- URL and mention removal
- Lowercase conversion
- Special character cleaning
- Duplicate removal

## Acknowledgments

### Models
- **ChunkFormer ASR**: [khanhld/chunkformer-ctc-large-vie](https://huggingface.co/khanhld/chunkformer-ctc-large-vie) - Vietnamese speech recognition
- **mDeBERTa**: [Microsoft DeBERTa](https://github.com/microsoft/DeBERTa) - Multilingual transformer for text classification

### Datasets
- **ViHSD**: [sonlam1102/vihsd](https://huggingface.co/datasets/sonlam1102/vihsd) - Vietnamese Hate Speech Detection dataset
- **VOZ-HSD**: [tarudesu/VOZ-HSD](https://huggingface.co/datasets/tarudesu/VOZ-HSD) - VOZ Forum Hate Speech Dataset
- **Crossmodal-3600**: [Google Crossmodal-3600](https://google.github.io/crossmodal-3600/) - Multilingual image captions for reducing data bias