import json
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

class ResultsProcessor:
    def __init__(self, run_id: str = None):
        """Initialize ResultsProcessor with specific run ID
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
        self.project_root = Path(".")
        model_search_dir = self.project_root / "outputs" / "model_search"
        
        # Find all date directories
        date_dirs = sorted(model_search_dir.glob("*-*-*"))  # matches YYYY-MM-DD
        if not date_dirs:
            raise FileNotFoundError("No experiment dates found")
            
        latest_date_dir = date_dirs[-1]
        
        # Find all time directories within the latest date
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
    
    def create_scaling_table(self, experiment_type: str):
        """Create table for scaling studies"""
        results = self.load_experiment_results(experiment_type)
        
        # Extract relevant metrics
        data = []
        for variant_name, result in results.items():
            # Get parameter count
            total_params = result.get('model_stats', {}).get('total_params', 0)
            # Format parameter count in millions
            if total_params > 0:
                param_count = f"{total_params/1e6:.2f}M"
            else:
                param_count = 'N/A'
            data.append({
                'Configuration': variant_name,
                'Total Parameters': param_count,
                'Training Time (s)': f"{np.mean(result['trial_times']):.1f} ± {np.std(result['trial_times']):.1f}",
                'Final Accuracy (%)': f"{np.mean(result['trial_accuracies']):.2f} ± {np.std(result['trial_accuracies']):.2f}"
            })
        
        # Create DataFrame and sort by configuration name
        df = pd.DataFrame(data)
        df = df.sort_values('Configuration')
        
        # Generate markdown table
        markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
        return markdown_table
    
    def create_efficiency_table(self, experiment_type: str):
        """Create efficiency metrics table"""
        results = self.load_experiment_results(experiment_type)
        
        data = []
        for variant_name, result in results.items():
            params = result.get('model_stats', {}).get('total_params', 0)
            acc = np.mean(result['trial_accuracies'])
            time = np.mean(result['trial_times'])
            
            data.append({
                'Configuration': variant_name,
                'Accuracy/Parameter (×10⁶)': f"{acc/params*1e6:.3f}" if params else 'N/A',
                'Accuracy/Second': f"{acc/time:.3f}" if time else 'N/A'
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Configuration')
        
        markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
        return markdown_table
    
    def create_model_summaries(self, experiment_type: str):
        """Create formatted model architecture summaries"""
        results = self.load_experiment_results(experiment_type)
        
        summaries = {}
        for variant_name, result in results.items():
            if 'model_stats' in result and 'model_summary' in result['model_stats']:
                summaries[variant_name] = result['model_stats']['model_summary']
                
        return summaries
    
    def generate_markdown_report(self, experiment_type: str):
        """Generate complete markdown report for an experiment"""
        is_scaling_study = any(x in experiment_type.lower() 
                             for x in ['scale', 'scaling', 'budget'])
        
        report = [
            f"# {experiment_type} Results\n",
            "## Configuration Comparison\n",
            self.create_scaling_table(experiment_type),
            "\n"
        ]
        
        if is_scaling_study:
            report.extend([
                "\n## Efficiency Metrics\n",
                self.create_efficiency_table(experiment_type),
                "\n"
            ])
            
        report.extend([
            "\n## Model Architecture Details\n",
            "```\n"
        ])
        
        summaries = self.create_model_summaries(experiment_type)
        for variant, summary in summaries.items():
            report.extend([
                f"\n### {variant}\n",
                summary,
                "\n"
            ])
            
        report.append("```\n")
        
        return "\n".join(report)
    
    def save_report(self, experiment_type: str, output_dir: str = "reports"):
        """Save markdown report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report = self.generate_markdown_report(experiment_type)
        
        with open(output_path / f"{experiment_type}_report.md", "w") as f:
            f.write(report)

def main():
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate result tables from experiment runs')
    parser.add_argument('run_id', nargs='?', type=str, help='Run ID in format "YYYY-MM-DD/HH-MM-SS". If not provided, uses latest run.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize processor with provided run_id or None for latest
    processor = ResultsProcessor(args.run_id)
    
    # Process all experiment types
    experiment_types = [
        "homogeneous_scale_study",
        "fixed_param_scale_study", 
        "meta_learner_study",
        "alpha_study",
        "init_study"
    ]
    
    for exp_type in experiment_types:
        try:
            processor.save_report(exp_type)
            print(f"Generated report for {exp_type}")
        except FileNotFoundError:
            print(f"No results found for {exp_type}")

if __name__ == "__main__":
    main()