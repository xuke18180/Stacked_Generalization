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
from mock_model import MockModel, MockTrainer

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
            # **process_params(search_space.training),
            'optimizer': process_params(search_space.optimizer),
            'scheduler': process_params(search_space.scheduler)
        }
    }
    
    # Merge with new config
    return OmegaConf.merge(new_config, trial_dict)

def save_best_config(study: optuna.Study, base_config: DictConfig):
    """Save the best configuration as YAML files."""
    # Consider all trials that have produced at least one value
    valid_trials = [t for t in study.trials 
                   if t.state == optuna.trial.TrialState.COMPLETE or 
                   (t.state == optuna.trial.TrialState.PRUNED and t.value is not None)]
    
    if not valid_trials:
        logging.warning("No trials produced valid results")
        results = {
            "status": "failed",
            "reason": "no_valid_results",
            "n_trials": len(study.trials),
            "optimizer_type": base_config.search_space.optimizer.name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        results_path = Path("search_results_failed.yaml")
        with open(results_path, "w") as f:
            yaml.dump(results, f, default_flow_style=False)
            
        logging.info(f"Failure results saved to {results_path}")
        return

    # Find best trial among all trials with valid results
    best_trial = max(valid_trials, key=lambda t: t.value)
    best_params = best_trial.params
    
    # Reconstruct best config
    best_config = create_trial_config(best_trial, base_config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Hydra automatically sets up the output directory
    config_path = Path("best_training.yaml")
    results_path = Path(f"search_results_{timestamp}.yaml")
    
    # Save config
    with open(config_path, "w") as f:
        yaml.dump(OmegaConf.to_container(best_config), f, default_flow_style=False)
    
    # Save search results
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    results = {
        "status": "success",
        "best_accuracy": best_trial.value,
        "best_params": best_params,
        "best_trial_state": best_trial.state.name,
        "best_trial_pruned_epoch": list(best_trial.intermediate_values.keys())[-1] if best_trial.state == optuna.trial.TrialState.PRUNED else None,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
        "n_pruned": len(pruned_trials),
        "n_pruned_with_results": len([t for t in pruned_trials if t.value is not None]),
        "optimizer_type": base_config.search_space.optimizer.name,
        "timestamp": timestamp
    }
    
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # Log results
    logging.info(f"Best trial accuracy: {best_trial.value:.4f} (from {best_trial.state.name} trial)")
    logging.info(f"Best parameters: {best_params}")
    if best_trial.state == optuna.trial.TrialState.PRUNED:
        logging.info(f"Best trial was pruned at epoch {list(best_trial.intermediate_values.keys())[-1]}")
    logging.info(f"Total trials: {len(study.trials)}, Completed: {len(completed_trials)}, " 
                f"Pruned: {len(pruned_trials)} ({len([t for t in pruned_trials if t.value is not None])} with valid results)")
    logging.info(f"Best configuration saved to {config_path}")
    logging.info(f"Search results saved to {results_path}")

def objective(trial: optuna.Trial, base_config: DictConfig) -> float:
    """Objective function for Optuna optimization."""
    try:
        # Create configuration for this trial
        trial_config = create_trial_config(trial, base_config)
        
        # Initialize model and trainer
        # mock model and trainer
        # model = MockModel(trial_config)
        # trainer = MockTrainer(trial_config)

        # Real model and trainer
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


@hydra.main(config_path="config", config_name="hyper_search_config", version_base="1.1")
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
    save_best_config(study, cfg)
    # plot_optimization_history(study, output_dir)

if __name__ == "__main__":
    main()