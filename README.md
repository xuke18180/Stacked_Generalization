# Stacked Generalization for Deep Learning

This project implements a stacked generalization approach for deep learning models, combining multiple base learners with a meta-learner for improved performance.

## Setup Instructions

1. Create the conda environment:
```bash
mamba env create --file environment.yml
```

2. Activate the environment:
```bash
conda activate stacked-gen
```

3. Prepare CIFAR-10 dataset in FFCV format:
```bash
# Create data directory
mkdir data

# Convert CIFAR-10 to FFCV format
python write_datasets.py --data.dataset cifar10
```

## Project Structure
```
.
├── config/
│   ├── config.yaml            # Main configuration
│   ├── dataset/
│   │   ├── cifar10.yaml      # CIFAR-10 dataset config
│   │   └── cifar100.yaml     # CIFAR-100 dataset config
│   └── model/
│       └── default.yaml      # Model architecture config
├── data/                     # Dataset directory
├── models.py                 # Model implementations
├── train.py                  # Training script
├── write_datasets.py         # Dataset conversion script
├── environment.yml          # Environment specification
└── README.md
```

## Training

To train with default configuration:
```bash
python train.py
```

To train with ffcv configuration as in https://docs.ffcv.io/ffcv_examples/cifar10.html
```bash
python train.py model=ffcv training=ffcv
```

To do hyperparameter search: (using SGD optimizer):
```bash
python train.py search_space=sgd
```

Common configuration overrides:
- `training.batch_size`: Batch size for training
- `training.lr`: Learning rate
- `training.epochs`: Number of training epochs
- `model.alpha`: Weight for base learner loss
- `model.meta_learner.hidden_dims`: Architecture of meta-learner MLP
- `model.meta_learner.dropout_rate`: Dropout rate in meta-learner

## Model Architecture

The model consists of:
1. Multiple base learners (ResNet variants)
2. Image feature extractor
3. Meta-learner that combines base predictions with image features

The current implementation uses:
- Three ResNet models (ResNet18, ResNet34, ResNet50) as base learners
- A custom CNN for image feature extraction
- An MLP-based meta-learner with configurable architecture

## Monitoring

Training progress is logged to WandB (Weights & Biases) with the following metrics:
- Training/validation loss
- Training/validation accuracy
- Individual base learner losses
- Meta-learner loss

## Files Used for Training

Make sure these files exist in your data directory:
- `data/cifar10_train.ffcv`: Training dataset
- `data/cifar10_val.ffcv`: Validation dataset

## Notes

- The project uses Hydra for configuration management
- FFCV is used for fast data loading
- Training supports mixed precision and distributed training via Accelerate
- WandB is used for experiment tracking