# hyper_search.py
import os
import logging
from pathlib import Path
import yaml
import optuna
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from train import Trainer
from models import create_model_from_config
from datetime import datetime
import wandb

class OptunaPruningCallback:
    def __init__(self, trial, patience=5, min_delta=0.1):
        self.trial = trial
        self.best_accuracy = None
        self.patience = patience
        self.min_delta = min_delta
        self.epochs_without_improvement = 0
        
    def __call__(self, epoch: int, accuracy: float):
        self.trial.report(accuracy, epoch)
        
        # Update best accuracy and check for improvement
        if self.best_accuracy is None or accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = accuracy
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
        # Prune if no improvement for patience epochs
        if self.epochs_without_improvement >= self.patience:
            raise optuna.TrialPruned()

def suggest_value(trial: optuna.Trial, name: str, param_config: dict):
    """Suggest a value based on parameter configuration."""
    if param_config["type"] == "categorical":
        return trial.suggest_categorical(name, param_config["choices"])
    elif param_config["type"] == "float":
        log = param_config.get("log", False)
        return trial.suggest_float(name, param_config["low"], param_config["high"], log=log)
    elif param_config["type"] == "int":
        log = param_config.get("log", False)
        return trial.suggest_int(name, param_config["low"], param_config["high"], log=log)
    else:
        raise ValueError(f"Unknown parameter type: {param_config['type']}")

def create_trial_config(trial: optuna.Trial, base_config: DictConfig) -> DictConfig:
    """Create a trial configuration by sampling from the search space."""
    # Create a new config with only non-search_space parts of base_config
    new_config = OmegaConf.create({
        k: v for k, v in base_config.items() 
        if k != 'search_space' and k != 'optuna'
    })
    
    # Get search space
    search_space = base_config.search_space
    
    # Function to suggest values from a parameter space
    def process_params(param_space):
        params = {}
        for key, value in param_space.items():
            if OmegaConf.is_config(value) and 'type' in value:
                params[key] = suggest_value(trial, key, value)
            else:
                params[key] = value
        return params
    
    # Create trial configuration
    trial_dict = {
        'training': {
            **process_params(search_space.training),
            'optimizer': process_params(search_space.optimizer),
            'scheduler': process_params(search_space.scheduler)
        }
    }
    
    # Merge with new config
    return OmegaConf.merge(new_config, trial_dict)

def save_best_config(study: optuna.Study, base_config: DictConfig, output_dir: str):
    """Save the best configuration as YAML files."""
    best_trial = study.best_trial
    best_params = best_trial.params
    
    # Reconstruct best config
    best_config = create_trial_config(best_trial, base_config, base_config.search_space)
    
    # Save training config
    training_config = {
        "optimizer": OmegaConf.to_container(best_config.optimizer),
        "scheduler": OmegaConf.to_container(best_config.scheduler),
        "training": OmegaConf.to_container(best_config.training)
    }
    
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = output_dir / f"best_training.yaml"
    
    with open(config_path, "w") as f:
        yaml.dump(training_config, f, default_flow_style=False)
    
    # Save search results
    results = {
        "best_accuracy": best_trial.value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "optimizer_type": base_config.search_space.optimizer.name,
        "timestamp": timestamp
    }
    
    results_path = output_dir / f"search_results_{timestamp}.yaml"
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # Log results
    logging.info(f"Best trial accuracy: {best_trial.value:.4f}")
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best configuration saved to {config_path}")
    logging.info(f"Search results saved to {results_path}")

def objective(trial: optuna.Trial, base_config: DictConfig) -> float:
    """Objective function for Optuna optimization."""
    try:
        # Create configuration for this trial
        trial_config = create_trial_config(trial, base_config)
        
        # Initialize model and trainer
        model = create_model_from_config(trial_config)
        trainer = Trainer(trial_config)
        
        # Create pruning callback
        pruning_callback = OptunaPruningCallback(
            trial,
            patience=base_config.optuna.pruning_patience,
            min_delta=base_config.optuna.pruning_min_delta
        )
        
        # Train and get best validation accuracy
        best_accuracy = trainer.train(model, callback=pruning_callback)
        return best_accuracy
    except optuna.TrialPruned:
        if wandb.run is not None:
            wandb.finish()
        raise  # Re-raise the pruning exception for Optuna to handle

    except Exception as e:
        # Handle any other exceptions
        logging.error(f"Trial failed with error: {str(e)}")
        if wandb.run is not None:
            wandb.finish()
        raise  


@hydra.main(config_path="config", config_name="hyper_search_config")
def main(cfg: DictConfig):
    # Set up logging
    logging.info(f"Starting hyperparameter search for {cfg.search_space.optimizer.name}")
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=cfg.optuna.n_startup_trials,
            n_warmup_steps=cfg.optuna.n_warmup_steps
        )
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.optuna.n_trials,
        timeout=cfg.optuna.timeout
    )
    
    # Save results and plots
    output_dir = hydra.core.hydra_config.HydraConfig.get().output_dir
    save_best_config(study, cfg, output_dir)
    # plot_optimization_history(study, output_dir)

if __name__ == "__main__":
    main()