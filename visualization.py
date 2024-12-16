# visualization.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

class VisualizationProcessor:
    def __init__(self, run_id: str = None):
        """Initialize VisualizationProcessor with specific run ID
        Args:
            run_id: String in format "YYYY-MM-DD/HH-MM-SS" or None for latest run
        """
        self.project_root = Path("outputs/model_search")
        
        if run_id:
            self.base_dir = self.project_root / run_id / "experiment_results"
            if not self.base_dir.exists():
                raise FileNotFoundError(f"Results directory not found: {self.base_dir}")
        else:
            self.base_dir = self.find_latest_run()
            
        print(f"Processing results from: {self.base_dir}")
        
    def find_latest_run(self):
        """Find the most recent experiment run directory"""
        date_dirs = sorted(self.project_root.glob("*-*-*"))  # matches YYYY-MM-DD
        if not date_dirs:
            raise FileNotFoundError("No experiment dates found")
            
        latest_date_dir = date_dirs[-1]
        
        time_dirs = sorted(latest_date_dir.glob("*-*-*"))  # matches HH-MM-SS
        if not time_dirs:
            raise FileNotFoundError(f"No experiment times found in {latest_date_dir}")
            
        return time_dirs[-1] / "experiment_results"
    
    def load_experiment_results(self, experiment_type: str):
        """Load all results for a specific experiment type"""
        exp_dir = self.base_dir / experiment_type
        results = {}
        if not exp_dir.exists():
            raise FileNotFoundError(f"No results directory found for experiment type '{experiment_type}' at {exp_dir}")
            
        # Check if directory is empty
        variant_dirs = list(exp_dir.glob("*"))
        if not variant_dirs:
            raise FileNotFoundError(f"No variant directories found for experiment type '{experiment_type}'")

        
        for variant_dir in exp_dir.glob("*"):
            if variant_dir.is_dir():
                try:
                    with open(variant_dir / "study_results.json", "r") as f:
                        results[variant_dir.name] = json.load(f)
                except FileNotFoundError:
                    print(f"No results found for {variant_dir}")
                    
                    
        return results
    
    def create_visualization(self, experiment_type: str, output_dir: Path):
        """Create visualization for a specific experiment type"""
        results = self.load_experiment_results(experiment_type)
        if not results:
            print(f"No results found for {experiment_type}")
            return
        # Sort variants based on numeric value in name
        def extract_number(variant_name):
            try:
                # For alpha_study: "alpha_0.3" -> 0.3
                if experiment_type == "alpha_study":
                    return float(variant_name.split('_')[-1])
                # For image_features_dim_study: "dim_128" -> 128
                elif experiment_type == "image_features_dim_study":
                    return int(variant_name.split('_')[-1])
                # For meta_learner_study: Extract number of layers from hidden_dims
                elif experiment_type == "meta_learner_study":
                    if "single" in variant_name:
                        return 1
                    elif "two" in variant_name:
                        return 2
                    elif "three" in variant_name:
                        return 3
                    return 0
                # For homogeneous_scale_study: "1x_resnet9" -> 1
                elif experiment_type == "homogeneous_scale_study":
                    return int(variant_name.split('x_')[0])
                return variant_name  # fallback to alphabetical sorting
            except:
                return variant_name

        # Sort variants
        sorted_variants = sorted(results.items(), key=lambda x: extract_number(x[0]))
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color palette for different variants
        colors = sns.color_palette("husl", len(results))
        
        # Track metrics for summary statistics
        best_accuracies = {}
        convergence_speeds = {}
        
        # Plot training histories and collect statistics
        for idx, (variant_name, result) in enumerate(sorted_variants):
            color = colors[idx]
            
            # Plot best trial history
            if 'best_history' in result and result['best_history']:
                epochs = result['best_history']['epochs']
                accuracies = result['best_history']['val_acc']
                ax.plot(epochs, accuracies, '-', color=color, label=variant_name, linewidth=2)
                
                # Calculate convergence speed (epochs to reach 90% of max accuracy)
                # max_acc = max(accuracies)
                # threshold = 0.9 * max_acc
                # convergence_epoch = next((i for i, acc in enumerate(accuracies) if acc >= threshold), len(epochs))
                # convergence_speeds[variant_name] = convergence_epoch
                # best_accuracies[variant_name] = max_acc
        
        # Customize plot
        ax.set_title(f'{experiment_type}: Validation Accuracy for CIFAR100', pad=20)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add summary statistics as text
        # summary_text = "Summary Statistics:\n\n"
        # for variant in best_accuracies.keys():
        #     summary_text += f"{variant}:\n"
        #     summary_text += f"  Best Accuracy: {best_accuracies[variant]:.2f}%\n"
        #     summary_text += f"  Convergence Epoch: {convergence_speeds[variant]}\n\n"
        
        # plt.figtext(1.3, 0.5, summary_text, fontsize=10, family='monospace')
        
        # Adjust layout and save
        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        
        # Save plot
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{experiment_type}_visualization.png", 
                    bbox_inches='tight', dpi=300)
        plt.close()
    
    def create_all_visualizations(self, output_dir: Path = Path("visualizations")):
        """Create visualizations for all experiments"""
        experiment_types = [
            "meta_learner_study",
            "alpha_study",
            "init_study",
            "homogeneous_scale_study",
            "fixed_param_scale_study",
            "image_features_dim_study",
        ]
        
        for exp_type in experiment_types:
            try:
                self.create_visualization(exp_type, output_dir)
                print(f"Created visualization for {exp_type}")
            except Exception as e:
                print(f"Failed to create visualization for {exp_type}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from experiment runs')
    parser.add_argument('run_id', nargs='?', type=str, 
                       help='Run ID in format "YYYY-MM-DD/HH-MM-SS". If not provided, uses latest run.')
    parser.add_argument('--output-dir', type=str, default="visualizations",
                       help='Directory to save visualizations (default: visualizations)')
    
    args = parser.parse_args()
    
    try:
        processor = VisualizationProcessor(args.run_id)
        processor.create_all_visualizations(Path(args.output_dir))
        print(f"Visualizations saved to {args.output_dir}")
    except Exception as e:
        print(f"Error processing results: {str(e)}")

if __name__ == "__main__":
    main()