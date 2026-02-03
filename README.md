# Image Captioning

A deep learning project that automatically generates descriptive captions for images using a CNN-RNN architecture. This project uses the [COCO dataset](http://cocodataset.org/#home) to train a model that combines computer vision and natural language processing.

## Overview

This project implements an end-to-end image captioning system using:
- **Encoder (CNN)**: ResNet-18 for extracting visual features from images
- **Decoder (RNN)**: LSTM network for generating captions based on visual features

The model learns to predict the next word in a caption given previous words and the image features, enabling automatic generation of natural language descriptions for any image.

## Project Structure

```
image-captioning/
├── src/
│   ├── model.py              # CNN-RNN model architecture
│   ├── data_loader.py        # COCO dataset loading and preprocessing
│   └── vocabulary.py         # Vocabulary management for captions
├── preliminaries.ipynb       # Data loading and exploration
├── training.ipynb            # Model training pipeline
├── inference.ipynb           # Caption generation on new images
└── README.md                 # This file
```

## Key Components

### Model Architecture

**EncoderCNN**
- Uses pre-trained ResNet-18 to extract visual features from images
- Final fully connected layer projects features to embedding space
- Frozen convolutional layers preserve learned visual representations

**DecoderRNN**
- LSTM-based decoder that generates captions word-by-word
- Embeds vocabulary words and processes them sequentially
- Projects LSTM hidden states to vocabulary predictions

### Data Processing

**CoCoDataset**
- Loads images and captions from COCO dataset
- Applies transformations: resizing, cropping, normalization
- Tokenizes captions and converts to vocabulary indices

**Vocabulary**
- Builds vocabulary from training captions with frequency thresholds
- Handles special tokens: `<start>`, `<end>`, `<unk>`
- Supports pickle serialization for efficient reuse

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- NLTK
- pycocotools
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning
```

2. Install dependencies:
```bash
pip install torch torchvision nltk pycocotools
python -m nltk.downloader punkt
```

3. Download COCO dataset (skip if already available):
```bash
# Follow instructions at http://cocodataset.org/#download
```

## Usage

### 1. Data Preparation (preliminaries.ipynb)
- Loads and explores the COCO dataset
- Preprocesses images (resize, crop, normalize)
- Builds vocabulary from training captions
- Configurable parameters:
  - `vocab_threshold`: Minimum word frequency (default: 5)
  - `batch_size`: Training batch size (default: 10)

### 2. Model Training (training.ipynb)
- Trains the CNN-RNN model on COCO dataset
- Monitors training loss and validation metrics
- Saves best model checkpoints

### 3. Inference (inference.ipynb)
- Generates captions for new images
- Uses greedy decoding or sampling strategies
- Visualizes results with images and predictions

## Key Features

- **Pre-trained CNN**: Leverages ImageNet-trained ResNet-18 for robust feature extraction
- **Flexible Vocabulary**: Customizable word frequency thresholds
- **Data Augmentation**: Random cropping and horizontal flipping during training
- **Batch Processing**: Efficient GPU utilization with configurable batch sizes
- **Word Embeddings**: Learned embeddings map vocabulary to semantic space

## Training Details

- **Encoder**: Frozen ResNet-18 (transfer learning)
- **Decoder**: 1-layer LSTM with embedding layer and linear output
- **Loss Function**: Cross-entropy loss for word prediction
- **Optimizer**: Adam optimizer for gradient updates
- **Input Features**: 224×224 normalized images
- **Feature Dimension**: Configurable embedding size (typically 256-512)

## Sample Usage

```python
from src.data_loader import get_loader
from src.model import EncoderCNN, DecoderRNN
import torch

# Load data
data_loader = get_loader(
    transform=transform_train,
    mode='train',
    batch_size=10,
    vocab_threshold=5,
    vocab_from_file=False
)

# Initialize models
encoder = EncoderCNN(embed_size=256)
decoder = DecoderRNN(embed_size=256, hidden_size=512, 
                     vocab_size=len(data_loader.dataset.vocab))

# Train on batch
for images, captions in data_loader:
    features = encoder(images)
    outputs = decoder(features, captions)
```

## Results

The trained model learns to generate meaningful captions that:
- Identify objects and people in images
- Describe spatial relationships
- Capture scene context and atmosphere
- Use diverse and natural language

## Future Improvements

- Attention mechanism for focusing on image regions during caption generation
- Beam search decoding for better quality captions
- Extended training on full COCO dataset
- Fine-tuning encoder with stronger backbones (ResNet-50, EfficientNet)
- Multi-head attention and transformer-based decoder

## References

- [COCO Dataset](http://cocodataset.org/)
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Knowing When to Look: Adaptive Attention in Image Captioning](https://arxiv.org/abs/1612.01887)

## Acknowledgments

This project was completed as part of the Udacity Computer Vision Nanodegree Program.

## Author

Tabish Punjani

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
