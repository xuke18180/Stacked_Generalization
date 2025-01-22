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
│   ├── config.yaml                # Main configuration
│   ├── dataset/
│   │   ├── cifar10.yaml          # CIFAR-10 dataset config
│   │   └── cifar100.yaml         # CIFAR-100 dataset config
│   ├── model/
│   │   └── default.yaml          # Model architecture config
│   ├── search_space/
│   │   ├── adamw.yaml            # AdamW hyperparameter search space
│   │   └── sgd.yaml              # SGD hyperparameter search space
│   ├── model_search_config.yaml   # Model architecture search config
│   └── hyper_search_config.yaml   # Hyperparameter search config
├── data/                         # Dataset directory
├── models/                       # Model implementations
├── train.py                      # Training script
├── hyper_search.py              # Hyperparameter optimization
├── model_search.py              # Model architecture search
├── mock_model.py                # Mock models for debugging
├── write_datasets.py            # Dataset conversion script
├── environment.yml              # Environment specification
├── generate_result_table.py     # Generate result table in latex
├── visualization.py             # Visualize experiment results
└── README.md
```

## Training

Basic training with default configuration:
```bash
python train.py
```

Training with FFCV configuration:
```bash
python train.py model=ffcv training=ffcv
```

## Model Architecture Search

The project includes a comprehensive model architecture search framework in `model_search.py`. It supports various experiments defined in `config/model_search_config.yaml`:

1. Meta-learner architecture study:
```bash
python model_search.py run_experiment=meta_learner_study
```

2. Loss weighting study:
```bash
python model_search.py run_experiment=alpha_study
```

3. Initialization strategy study:
```bash
python model_search.py run_experiment=init_study
```

4. Model scaling studies:
```bash
python model_search.py run_experiment=homogeneous_scale_study
python model_search.py run_experiment=fixed_param_scale_study
```

To run all experiments:
```bash
python model_search.py
```

5. Run ViT experiments:
```bash
python model_search.py --config-name model_search_config_vit
```

6. Run Stacked Generalization experiments with cross validation:
```bash
python train.py training=stacked
```

### Configuring Experiments

Experiments are defined in `config/model_search_config.yaml`. Each experiment type has variants that specify different configurations to test. Example structure:

```yaml
experiments:
  meta_learner_study:
    variants:
      - name: "single_layer"
        hidden_dims: [512]
      - name: "two_layer"
        hidden_dims: [512, 256]

  alpha_study:
    variants:
      - name: "alpha_0.3"
        value: 0.3
      - name: "alpha_1.0"
        value: 1.0
```

## Hyperparameter Optimization

Run hyperparameter search with specific optimizer (this is not used to generate result because its functionality is covered by model_search.py. If you want to do hyperparameter search for a specific model configuration, then you can use this.):
```bash
python hyper_search.py search_space=sgd  # or search_space=adamw
```

Search spaces are defined in `config/search_space/`:
- `sgd.yaml`: Search space for SGD optimizer
- `adamw.yaml`: Search space for AdamW optimizer

## Debugging with Mock Models

For debugging experiments without running full training, use the mock implementations in `mock_model.py`:

1. In `model_search.py`, replace the model and trainer creation:
```python
# Replace these lines:
model = create_model_from_config(trial_config)
trainer = Trainer(trial_config)

# With mock versions:
model = MockModel(trial_config)
trainer = MockTrainer(trial_config)
```

2. In `hyper_search.py`, similar replacements can be made in the objective function.

The mock implementations simulate training behavior with realistic accuracy curves and training times, making them useful for:
- Testing experiment configurations
- Debugging search logic
- Validating metrics tracking
- Testing WandB integration

## Monitoring

Training progress is logged to WandB (Weights & Biases) with the following metrics:
- Training/validation loss
- Training/validation accuracy
- Individual base learner losses
- Meta-learner loss

## Results Analysis

After running experiments, results are saved in:
- `outputs/model_search/[DATE]/[TIME]/experiment_results/`
- `outputs/hyper_search/[OPTIMIZER]/[DATE]/[TIME]/`

Each experiment generates:
- Training history plots
- Hyperparameter distribution analysis
- Detailed JSON results
- Model statistics

### Generating Result Reports

The project includes a report generation script that creates formatted LateX (or markdown reports - uncomment some lines in the main() function) from experiment results:

```bash
# Generate reports from latest experiment run
python generate_result_table.py

# Generate reports from specific run
python generate_result_table.py "2024-12-13/14-30-00"  # Format: "YYYY-MM-DD/HH-MM-SS"
```

Reports are generated for each experiment type:
- Homogeneous scaling study
- Fixed parameter scaling study
- Meta-learner architecture study
- Alpha (loss weight) study
- Initialization strategy study
- image features extractor output dimension study

Reports include:
- Configuration comparison tables
- Efficiency metrics for scaling studies
- Training time and accuracy statistics
- Model architecture details

Generated reports are saved in the `reports/` directory with filenames like:
- `homogeneous_scale_study_report.md`
- `meta_learner_study_report.md`
- etc.

### Generating Visualizations of Experiment Results
You can also generate visualization of the studies you run for given experiment:
```bash
# Generate reports from specific run
python visualization.py "2024-12-13/14-30-00"  # Format: "YYYY-MM-DD/HH-MM-SS"
```
This will create a new folder "visualizations/" for the plots.

## Common Configuration Overrides

- `training.batch_size`: Batch size for training
- `training.lr`: Learning rate
- `training.epochs`: Number of training epochs
- `model.alpha`: Weight for base learner loss
- `model.meta_learner.hidden_dims`: Architecture of meta-learner MLP
- `model.meta_learner.dropout_rate`: Dropout rate in meta-learner
- `optuna.n_trials`: Number of trials for hyperparameter search
- `optuna.timeout`: Maximum search time in seconds