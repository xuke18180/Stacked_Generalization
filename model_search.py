from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
import json
import optuna
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from train import Trainer
from hyper_search import create_trial_config, OptunaPruningCallback
from models import create_model_from_config
import logging
import hydra
import seaborn as sns
import wandb
from torchsummary import summary
import time

from mock_model import MockModel, MockTrainer

COMPLETE = optuna.trial.TrialState.COMPLETE
PRUNED = optuna.trial.TrialState.PRUNED

def get_model_stats(model, input_size=(3, 32, 32)):
    """Get model parameters and summary using torchinfo"""
    try:
        # Get model summary
        model_summary = summary(
            model, 
            input_size=input_size,
            verbose=0,  # Don't print the summary
        )
        
        stats = {
            'total_params': model_summary.total_params,
            'trainable_params': model_summary.trainable_params
        }
        
        return stats
        
    except Exception as e:
        print(f"Error generating model summary: {str(e)}")
        return None

@dataclass
class ExperimentMetadata:
    """Structured config for experiment metadata"""
    experiment_type: str
    variant_name: str

class ExperimentManager:
    def __init__(self, cfg: DictConfig, output_dir: str):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.results = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_experiment_config(self, experiment_type: str, variant: dict) -> Dict:
        """Create specific experiment configuration with metadata tracking"""
        # Create a new config that includes metadata
        base_config = OmegaConf.to_container(self.cfg, resolve=True)
        
        # Create metadata
        metadata = ExperimentMetadata(
            experiment_type=experiment_type,
            variant_name=variant["name"]
        )
        
        # Add metadata to config
        base_config["experiment_metadata"] = OmegaConf.structured(metadata)
        
        # Convert back to DictConfig
        config = OmegaConf.create(base_config)
        
        # Automatically update all fields present in the variant
        def update_config_recursive(config_section, variant_section):
            for key, value in variant_section.items():
                if isinstance(value, dict) and key in config_section:
                    # Recursively handle nested configurations
                    update_config_recursive(config_section[key], value)
                else:
                    # Update the value directly
                    config_section[key] = value
        
        # Handle special case where variant defines complete base_learners list
        if "base_learners" in variant:
            config.model.base_learners = variant["base_learners"]
        
        # Handle all other fields by recursively updating the config
        for key, value in variant.items():
            if key != "name":  # Skip the name field
                if key in config.model:
                    if isinstance(value, dict):
                        update_config_recursive(config.model[key], value)
                    else:
                        config.model[key] = value
        
        return config

    def validate_experiment_config(self, experiment_type: str, variant: dict) -> List[str]:
        """Validate experiment configuration and return any warnings"""
        warnings = []
        
        # Check for required fields
        required_fields = ["name"]
        missing_fields = [field for field in required_fields if field not in variant]
        if missing_fields:
            warnings.append(f"Missing required fields: {missing_fields}")
        
        # Validate base_learners configuration if present
        if "base_learners" in variant:
            for i, learner in enumerate(variant["base_learners"]):
                required_learner_fields = ["architecture"]
                missing_learner_fields = [
                    field for field in required_learner_fields 
                    if field not in learner
                ]
                if missing_learner_fields:
                    warnings.append(
                        f"Base learner {i} missing required fields: {missing_learner_fields}"
                    )
        
        # Validate meta_learner configuration if present
        if "meta_learner" in variant:
            if "hidden_dims" not in variant["meta_learner"]:
                warnings.append("Meta learner configuration missing hidden_dims")
        
        return warnings
    
    def run_trials(self,
                  config_updates: DictConfig,
                  experiment_type: str,
                  variant_name: str) -> Dict:
        """Run hyperparameter optimization trials for a configuration"""
        trial_accuracies = []
        trial_times = []
        best_history = None
        best_trial_params = None
        best_acc = 0
        model_stats = None
        
        # Create study dir for this experiment
        study_dir = self.output_dir / experiment_type / variant_name
        study_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.cfg.optuna.n_startup_trials,
                n_warmup_steps=self.cfg.optuna.n_warmup_steps
            )
        )
        
        # Define objective with history tracking
        histories = {}
        def objective(trial: optuna.Trial) -> float:
            try:
                # Create trial config
                trial_config = create_trial_config(trial, config_updates)
                
                # Track training history
                history = {"train_acc": [], "val_acc": [], "epochs": []}
                
                # Create combined callback for both pruning and history tracking
                def combined_callback(epoch: int, acc: float):
                    history["epochs"].append(epoch)
                    history["val_acc"].append(acc)
                    trial.report(acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                # Train model
                start_time = time.time()
                # mock code
                # model = MockModel(trial_config)
                # trainer = MockTrainer(trial_config)

                # real code
                model = create_model_from_config(trial_config)
                nonlocal model_stats
                if model_stats is None:  # Only need to do this once per configuration
                    model_stats = get_model_stats(
                        model, 
                        input_size=(3, self.cfg.dataset.input_size, self.cfg.dataset.input_size)
                    )

                trainer = Trainer(trial_config)
                val_acc = trainer.train(
                    model,
                    callback=combined_callback
                )
                train_time = time.time() - start_time
                
                # Store results
                trial_accuracies.append(val_acc)
                trial_times.append(train_time)
                histories[trial.number] = history
                
                # Update best results
                nonlocal best_acc, best_history, best_trial_params
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_history = history
                    best_trial_params = trial.params
                
                return val_acc
                
            except optuna.TrialPruned:
                histories[trial.number] = {
                    **history,
                    "pruned_at_epoch": len(history["epochs"])
                }
                if wandb.run is not None:
                    wandb.finish()
                raise
            
            except Exception as e:
                logging.error(f"Trial failed with error: {str(e)}")
                raise
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.cfg.optuna.n_trials,
            timeout=self.cfg.optuna.timeout
        )
        
        # Save study results
        study_results = {
            "best_params": best_trial_params,
            "trial_accuracies": trial_accuracies,
            "trial_times": trial_times,
            "best_history": best_history,
            "all_trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "state": t.state,
                    "history": histories.get(t.number, {})
                }
                for t in study.trials
            ],
            "experiment_type": experiment_type,
            "variant_name": variant_name,
            "model_stats": model_stats,
        }
        
        with open(study_dir / "study_results.json", "w") as f:
            json.dump(study_results, f, indent=2)
        
        return study_results
    
    def run_specific_experiment(self, experiment_type: str):
        """Run a specific experiment type"""
        if experiment_type not in self.cfg.experiments:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        exp_config = self.cfg.experiments[experiment_type]
        logging.info(f"Running {experiment_type} experiment")
        
        for variant in exp_config.variants:
            config = self.get_experiment_config(experiment_type, variant)
            warnings = self.validate_experiment_config(experiment_type, variant)
            result = self.run_trials(
                config,
                experiment_type,
                variant["name"]
            )
            self.results.append(result)
    
    def run_all_experiments(self):
        """Run all experiments defined in config"""
        for exp_type in self.cfg.experiments:
            self.run_specific_experiment(exp_type)

    def plot_hyperparameters(self, study_dir: Path):
        """Analyze hyperparameters across all trials in a study"""
        with open(study_dir / "study_results.json", "r") as f:
            study_data = json.load(f)
        
        # Extract hyperparameters from all trials
        trial_params = [t["params"] for t in study_data["all_trials"] if t["state"] == COMPLETE]
        trial_values = [t["value"] for t in study_data["all_trials"] if t["state"] == COMPLETE]
        
        # Create DataFrame of hyperparameters
        params_df = pd.DataFrame(trial_params)
        params_df["accuracy"] = trial_values

        # Identify numerical and categorical columns
        numerical_cols = params_df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = params_df.select_dtypes(include=['object', 'bool']).columns
        
        # Calculate statistics for each hyperparameter
        stats = {}
        for column in params_df.columns:
            if column == "accuracy":
                continue
                
            if column in numerical_cols:
                # Handle numerical parameters
                stats[column] = {
                    "type": "numerical",
                    "mean": params_df[column].mean(),
                    "std": params_df[column].std(),
                    "best_trial_value": study_data["best_params"][column],
                    "correlation_with_accuracy": params_df[column].corr(params_df["accuracy"])
                }
            else:
                # Handle categorical parameters
                value_counts = params_df[column].value_counts()
                stats[column] = {
                    "type": "categorical",
                    "value_counts": value_counts.to_dict(),
                    "best_trial_value": study_data["best_params"][column],
                    "mode": value_counts.index[0]
                }
        
        # Create separate plots for numerical and categorical parameters
        n_numerical = len([c for c in params_df.columns if c in numerical_cols and c != "accuracy"])
        n_categorical = len([c for c in params_df.columns if c in categorical_cols])
        
        if n_numerical > 0:
            plt.figure(figsize=(12, n_numerical * 3))
            plot_num = 1
            for column in numerical_cols:
                if column != "accuracy":
                    plt.subplot(n_numerical, 1, plot_num)
                    sns.kdeplot(params_df[column], fill=True)
                    plt.axvline(study_data["best_params"][column], color='r', 
                            linestyle='--', label='Best Trial')
                    plt.title(f'{column} Distribution (corr with acc: {stats[column]["correlation_with_accuracy"]:.3f})')
                    plt.legend()
                    plot_num += 1
            plt.tight_layout()
            plt.savefig(study_dir / "numerical_hyperparameters.png")
            plt.close()
        
        if n_categorical > 0:
            plt.figure(figsize=(12, n_categorical * 3))
            plot_num = 1
            for column in categorical_cols:
                plt.subplot(n_categorical, 1, plot_num)
                value_counts = params_df[column].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.axvline(value_counts.index.get_loc(study_data["best_params"][column]), 
                        color='r', linestyle='--', label='Best Trial')
                plt.title(f'{column} Distribution')
                plt.xticks(rotation=45)
                plt.legend()
                plot_num += 1
            plt.tight_layout()
            plt.savefig(study_dir / "categorical_hyperparameters.png")
            plt.close()
        
        return stats
    
    def plot_results(self):
        """Generate plots for all experimental results"""
        for result in self.results:
            study_dir = self.output_dir / result["experiment_type"] / result["variant_name"]
            
            # Plot training histories
            plt.figure(figsize=(12, 6))
            for trial in result["all_trials"]:
                history = trial["history"]
                if "val_acc" in history:
                    epochs = history["epochs"]
                    acc = history["val_acc"]
                    if trial["state"] == PRUNED:
                        plt.plot(epochs, acc, 'r--', alpha=0.3)
                    elif trial["state"] == COMPLETE:
                        plt.plot(epochs, acc, 'b-', alpha=0.3)
            
            plt.plot([], [], 'b-', label='Completed')
            plt.plot([], [], 'r--', label='Pruned')
            
            plt.title(f"{result['experiment_type']} - {result['variant_name']}\nTraining Histories")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(study_dir / "training_histories.png")
            plt.close()

            hyper_stats = self.plot_hyperparameters(study_dir)

            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            with open(study_dir / "hyperparameter_stats.json", "w") as f:
                json.dump(hyper_stats, f, indent=2, default=convert_to_serializable)

@hydra.main(config_path="config", config_name="model_search_config", version_base="1.1")
def main(cfg: DictConfig):
    # Get experiment type from command line
    experiment_type = cfg.get("run_experiment", None)

    def test_experiment_configs():
        manager = ExperimentManager(cfg, "test_results")
        for exp_type in cfg.experiments:
            for variant in cfg.experiments[exp_type].variants:
                # Should not raise any exceptions
                warnings = manager.validate_experiment_config(exp_type, variant)
                
                variant_name = variant["name"] if "name" in variant else "Unnamed"
                if warnings:
                    logging.warning(
                        f"Configuration warnings for {experiment_type}/{variant_name}:\n" +
                        "\n".join(f"- {w}" for w in warnings)
                    )
                config = manager.get_experiment_config(exp_type, variant)
                logging.info(f"Generated model configuration for {exp_type}/{variant_name}:\n{OmegaConf.to_yaml(config.model)}")

    
    manager = ExperimentManager(cfg, "experiment_results")

    # to test the experiment configurations
    # test_experiment_configs()

    
    if experiment_type:
        # Run specific experiment
        manager.run_specific_experiment(experiment_type)
    else:
        # Run all experiments
        manager.run_all_experiments()

    manager.plot_results()


if __name__ == "__main__":
    main()