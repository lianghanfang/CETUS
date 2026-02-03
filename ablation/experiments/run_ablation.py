"""
Enhanced ablation experiment runner with selective component control
First-order ablation: vary either spatial or temporal while keeping the other fixed
Allows selection of specific encoders/models to test
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Add project root to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import core modules
from core.training.train import Trainer
from core.inference.inference import InferenceEngine
from ablation.models.ablation_model import AblationEventModel  # Add explicit import
from evaluation import evaluate_pixel_level  # Adjust path as needed


class EnhancedAblationRunner:
    """Enhanced ablation experiment runner with selective component control"""
    
    def __init__(self, base_config_path: str, output_dir: str):
        self.base_config = self.load_config(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment configurations
        self.setup_experiments()
    
    def load_config(self, path: str) -> Dict:
        """Load base configuration"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def setup_experiments(self):
        """Setup first-order ablation experiments"""
        
        # Fixed hyperparameters for fairness
        self.fixed_params = {
            'hidden_dim': 128,  # Same for all models
            'batch_size': 8,
            'learning_rate': 1e-4,
            'num_epochs': 50,
            'early_stopping_patience': 5,
            'window_size': 4,
            'dropout_rate': 0.1
        }
        
        # Baseline configuration
        self.baseline_spatial = {
            'spatial_encoder_type': 'knn',
            # 'knn': {
            #     'k_neighbors': 16,
            #     'spatial_radius': 1.0,
            #     'aggregation': 'attention',
            #     'causal': True
            # }
            "knn": {
                "k_neighbors": 8,
                "spatial_radius": 1.0,
                "aggregation": "attention",
                "causal": True,
                "use_adaptive_temp": True,
                "spatial_weight": 1.0,
                "temporal_weight": 0.3,

                "use_rings": True,
                "k_total": 24,
                "ring_sizes": [8,8,8],
                "gate_use_time": True,
                "gate_hidden": 32
            }
        }
        
        self.baseline_temporal = {
            'temporal_type': 'mamba',
            'num_blocks': 1,
            'mamba': {
                'd_state': 32,
                'd_conv': 4,
                'expand': 2
            }
        }
        
        # Available spatial encoders
        self.available_spatial = {
            'knn': self.baseline_spatial,
            'knn_base': {
                'spatial_encoder_type': 'knn_base',
                'k_neighbors': 16,
                'spatial_radius': 1.0,
                'aggregation': 'attention',
                'causal': True
            },
            'grid': {
                'spatial_encoder_type': 'grid',
                'grid_size': (1, 1),
                'neighborhood': 1
            },
            'voxel': {
                'spatial_encoder_type': 'voxel',
                'voxel_size': (0.5, 0.5, 0.005),
                'neighborhood_size': 0.1
            }
        }
        
        # Available temporal models
        self.available_temporal = {
            'mamba': self.baseline_temporal,
            'gru': {
                'temporal_type': 'gru',
                'num_layers': 2
            },
            'lstm': {
                'temporal_type': 'lstm',
                'num_layers': 2
            },
            'transformer': {
                'temporal_type': 'transformer',
                'num_layers': 2,
                'num_heads': 4
            },
            'none': {
                'temporal_type': 'none'
            }
        }
    
    def list_available_components(self):
        """List all available components"""
        print("\nAvailable Spatial Encoders:")
        for name in self.available_spatial.keys():
            print(f"  - {name}")
        
        print("\nAvailable Temporal Models:")
        for name in self.available_temporal.keys():
            print(f"  - {name}")
    
    def create_experiment_config(
        self,
        exp_name: str,
        spatial_config: Dict,
        temporal_config: Dict
    ) -> Dict:
        """Create complete experiment configuration"""
        config = self.base_config.copy()
        
        # Set model type for ablation
        config['model']['type'] = 'AblationEventModel'
        
        # Apply fixed hyperparameters
        config['model']['hidden_dim'] = self.fixed_params['hidden_dim']
        config['model']['window_size'] = self.fixed_params['window_size']
        config['model']['dropout_rate'] = self.fixed_params['dropout_rate']
        config['training']['batch_size'] = self.fixed_params['batch_size']
        config['training']['learning_rate'] = self.fixed_params['learning_rate']
        config['training']['num_epochs'] = self.fixed_params['num_epochs']
        config['training']['early_stopping_patience'] = self.fixed_params['early_stopping_patience']
        
        # Set spatial and temporal configurations
        config['model']['spatial_config'] = spatial_config
        config['model']['temporal_config'] = temporal_config
        
        # Set experiment directory
        config['exp_dir'] = str(self.output_dir / exp_name)
        
        return config
    
    def run_single_experiment(self, config: Dict, exp_name: str) -> Dict:
        """Run a single experiment with one seed"""
        print(f"\n{'='*60}")
        print(f"Running: {exp_name}")
        print(f"Spatial: {config['model']['spatial_config']['spatial_encoder_type']}")
        print(f"Temporal: {config['model']['temporal_config']['temporal_type']}")
        print(f"{'='*60}")
        
        results = {}
        
        try:
            # Training
            trainer = Trainer(config, device='cuda')
            best_metrics = trainer.train()
            checkpoint_path = str(Path(config['exp_dir']) / 'best_model.pth')
            
            # Inference
            engine = InferenceEngine(checkpoint_path=checkpoint_path, device='cuda')
            output_path = str(self.output_dir / f"{exp_name}_predictions.txt")
            
            engine.process_batch(
                test_dir=config['dataset']['test_dir'],
                output_path=output_path,
                feature_config=engine.config['features'],
                dataset_config=engine.config['dataset']
            )
            
            # Evaluation
            eval_df = evaluate_pixel_level(output_path)
            best_f1_idx = eval_df['F1'].idxmax()
            best_metrics = eval_df.iloc[best_f1_idx].to_dict()
            
            # Get model info
            model = trainer.model
            model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
            
            results = {
                'exp_name': exp_name,
                'checkpoint': checkpoint_path,
                'predictions': output_path,
                'best_metrics': best_metrics,
                'model_params': model_info.get('total_params', 0),
                'training_time': trainer.training_time if hasattr(trainer, 'training_time') else None
            }
            
            print(f"✓ {exp_name} completed")
            print(f"  Best F1: {best_metrics['F1']:.4f}")
            print(f"  Pd: {best_metrics['Pd']:.4f}, Fa: {best_metrics['Fa']:.6f}")
            
        except Exception as e:
            print(f"✗ Error in {exp_name}: {e}")
            results = {'exp_name': exp_name, 'error': str(e)}
        
        return results
    
    def run_spatial_ablation(self, selected_encoders: List[str] = None) -> List[Dict]:
        """Run spatial encoder ablation (fixed temporal = mamba)"""
        
        # Use all available encoders if none specified
        if selected_encoders is None:
            selected_encoders = list(self.available_spatial.keys())
        
        # Validate selected encoders
        invalid_encoders = set(selected_encoders) - set(self.available_spatial.keys())
        if invalid_encoders:
            raise ValueError(f"Invalid spatial encoders: {invalid_encoders}. Available encoders: {list(self.available_spatial.keys())}")

        # Print available encoders for reference
        print(f"Available spatial encoders: {list(self.available_spatial.keys())}")
        
        print("\n" + "="*70)
        print("SPATIAL ENCODER ABLATION")
        print("Fixed: Temporal = Mamba")
        print(f"Selected encoders: {', '.join(selected_encoders)}")
        print("="*70)
        
        results = []
        
        for encoder_name in selected_encoders:
            exp_name = f"spatial_{encoder_name}"
            spatial_config = self.available_spatial[encoder_name]
            
            config = self.create_experiment_config(
                exp_name,
                spatial_config,
                self.baseline_temporal
            )
            result = self.run_single_experiment(config, exp_name)
            results.append(result)
        
        return results
    
    def run_temporal_ablation(self, selected_models: List[str] = None) -> List[Dict]:
        """Run temporal model ablation (fixed spatial = knn)"""
        
        # Use all available models if none specified
        if selected_models is None:
            selected_models = list(self.available_temporal.keys())
        
        # Validate selected models
        invalid_models = set(selected_models) - set(self.available_temporal.keys())
        if invalid_models:
            raise ValueError(f"Invalid temporal models: {invalid_models}")
        
        print("\n" + "="*70)
        print("TEMPORAL MODEL ABLATION")
        print("Fixed: Spatial = KNN")
        print(f"Selected models: {', '.join(selected_models)}")
        print("="*70)
        
        results = []
        
        for model_name in selected_models:
            exp_name = f"temporal_{model_name}"
            temporal_config = self.available_temporal[model_name]
            
            config = self.create_experiment_config(
                exp_name,
                self.baseline_spatial,
                temporal_config
            )
            result = self.run_single_experiment(config, exp_name)
            results.append(result)
        
        return results
    
    def run_custom_experiments(self, custom_configs: List[Dict]) -> List[Dict]:
        """Run custom experiment combinations"""
        print("\n" + "="*70)
        print("CUSTOM EXPERIMENTS")
        print("="*70)
        
        results = []
        
        for custom_config in custom_configs:
            spatial_name = custom_config['spatial']
            temporal_name = custom_config['temporal']
            exp_name = custom_config.get('name', f"custom_{spatial_name}_{temporal_name}")
            
            if spatial_name not in self.available_spatial:
                print(f"Warning: Unknown spatial encoder '{spatial_name}', skipping")
                continue
            
            if temporal_name not in self.available_temporal:
                print(f"Warning: Unknown temporal model '{temporal_name}', skipping")
                continue
            
            spatial_config = self.available_spatial[spatial_name]
            temporal_config = self.available_temporal[temporal_name]
            
            config = self.create_experiment_config(
                exp_name,
                spatial_config,
                temporal_config
            )
            result = self.run_single_experiment(config, exp_name)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], name: str):
        """Save results to JSON and CSV"""
        # JSON format
        json_path = self.output_dir / f"{name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # CSV format
        if results and not any('error' in r for r in results):
            metrics_data = []
            for r in results:
                if 'best_metrics' in r:
                    row = {
                        'experiment': r['exp_name'],
                        'params': r.get('model_params', 0),
                        **r['best_metrics']
                    }
                    metrics_data.append(row)
            
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                csv_path = self.output_dir / f"{name}_metrics.csv"
                df.to_csv(csv_path, index=False)
                print(f"Results saved to {csv_path}")
    
    def generate_report(self, spatial_results: List[Dict], temporal_results: List[Dict], 
                       custom_results: List[Dict] = None):
        """Generate final ablation report"""
        report = []
        report.append("# Ablation Study Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Spatial ablation summary
        if spatial_results:
            report.append("## Spatial Encoder Ablation\n")
            report.append("(Fixed: Temporal = Mamba)\n\n")
            report.append("| Encoder | Params | F1 | Pd | Fa |\n")
            report.append("|---------|--------|----|----|----|\n")
            
            for r in spatial_results:
                if 'best_metrics' in r:
                    m = r['best_metrics']
                    encoder_name = r['exp_name'].replace('spatial_', '')
                    report.append(
                        f"| {encoder_name} | "
                        f"{r.get('model_params', 0):,} | "
                        f"{m['F1']:.4f} | {m['Pd']:.4f} | {m['Fa']:.6f} |\n"
                    )
            
            report.append("\n")
        
        # Temporal ablation summary
        if temporal_results:
            report.append("## Temporal Model Ablation\n")
            report.append("(Fixed: Spatial = KNN)\n\n")
            report.append("| Model | Params | F1 | Pd | Fa |\n")
            report.append("|-------|--------|----|----|----|\n")
            
            for r in temporal_results:
                if 'best_metrics' in r:
                    m = r['best_metrics']
                    model_name = r['exp_name'].replace('temporal_', '')
                    report.append(
                        f"| {model_name} | "
                        f"{r.get('model_params', 0):,} | "
                        f"{m['F1']:.4f} | {m['Pd']:.4f} | {m['Fa']:.6f} |\n"
                    )
            
            report.append("\n")
        
        # Custom experiments summary
        if custom_results:
            report.append("## Custom Experiments\n\n")
            report.append("| Experiment | Params | F1 | Pd | Fa |\n")
            report.append("|------------|--------|----|----|----|\n")
            
            for r in custom_results:
                if 'best_metrics' in r:
                    m = r['best_metrics']
                    report.append(
                        f"| {r['exp_name']} | "
                        f"{r.get('model_params', 0):,} | "
                        f"{m['F1']:.4f} | {m['Pd']:.4f} | {m['Fa']:.6f} |\n"
                    )
            
            report.append("\n")
        
        # Key findings
        report.append("## Key Findings\n\n")
        
        # Find best spatial encoder
        if spatial_results:
            best_spatial = max(
                [r for r in spatial_results if 'best_metrics' in r],
                key=lambda x: x['best_metrics']['F1'],
                default=None
            )
            if best_spatial:
                report.append(f"- **Best Spatial Encoder**: {best_spatial['exp_name']} "
                            f"(F1={best_spatial['best_metrics']['F1']:.4f})\n")
        
        # Find best temporal model
        if temporal_results:
            best_temporal = max(
                [r for r in temporal_results if 'best_metrics' in r],
                key=lambda x: x['best_metrics']['F1'],
                default=None
            )
            if best_temporal:
                report.append(f"- **Best Temporal Model**: {best_temporal['exp_name']} "
                            f"(F1={best_temporal['best_metrics']['F1']:.4f})\n")
        
        # Find overall best
        all_results = []
        if spatial_results:
            all_results.extend(spatial_results)
        if temporal_results:
            all_results.extend(temporal_results)
        if custom_results:
            all_results.extend(custom_results)
        
        if all_results:
            best_overall = max(
                [r for r in all_results if 'best_metrics' in r],
                key=lambda x: x['best_metrics']['F1'],
                default=None
            )
            if best_overall:
                report.append(f"- **Overall Best**: {best_overall['exp_name']} "
                            f"(F1={best_overall['best_metrics']['F1']:.4f})\n")
        
        # Save report
        report_path = self.output_dir / "ablation_report.md"
        with open(report_path, 'w') as f:
            f.writelines(report)
        
        print(f"\nReport saved to {report_path}")


def parse_component_list(component_str: str) -> List[str]:
    """Parse comma-separated component list"""
    if not component_str:
        return []
    return [c.strip() for c in component_str.split(',') if c.strip()]


def parse_custom_config(config_str: str) -> List[Dict]:
    """Parse custom experiment configuration
    Format: spatial1:temporal1,spatial2:temporal2 or JSON file path
    """
    if not config_str:
        return []
    
    # If it's a file path, load JSON
    if config_str.endswith('.json'):
        with open(config_str, 'r') as f:
            return json.load(f)
    
    # Parse inline format
    configs = []
    for pair in config_str.split(','):
        if ':' in pair:
            spatial, temporal = pair.strip().split(':', 1)
            configs.append({
                'spatial': spatial.strip(),
                'temporal': temporal.strip()
            })
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='Run enhanced ablation experiments with selective components')
    parser.add_argument('--config', type=str, required=True,
                       help='Base configuration file path')
    parser.add_argument('--output', type=str, default='ablation_results',
                       help='Output directory')
    parser.add_argument('--experiment', type=str,
                       choices=['spatial', 'temporal', 'custom', 'all'],
                       default='all',
                       help='Which ablation to run')
    
    # Selective component arguments
    parser.add_argument('--spatial-encoders', type=str,
                       help='Comma-separated list of spatial encoders to test (e.g., "knn,grid,voxel")')
    parser.add_argument('--temporal-models', type=str,
                       help='Comma-separated list of temporal models to test (e.g., "mamba,gru,lstm")')
    parser.add_argument('--custom-configs', type=str,
                       help='Custom experiments: "spatial1:temporal1,spatial2:temporal2" or JSON file path')
    
    # Utility arguments
    parser.add_argument('--list-components', action='store_true',
                       help='List all available components and exit')
    
    args = parser.parse_args()
    
    runner = EnhancedAblationRunner(args.config, args.output)
    
    # List components if requested
    if args.list_components:
        runner.list_available_components()
        return
    
    # Parse component selections
    selected_spatial = parse_component_list(args.spatial_encoders)
    selected_temporal = parse_component_list(args.temporal_models)
    custom_configs = parse_custom_config(args.custom_configs) if args.custom_configs else []
    
    # Run experiments
    spatial_results = []
    temporal_results = []
    custom_results = []
    
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("STARTING ENHANCED ABLATION STUDY")
    print(f"Output directory: {runner.output_dir}")
    if selected_spatial:
        print(f"Selected spatial encoders: {selected_spatial}")
    if selected_temporal:
        print(f"Selected temporal models: {selected_temporal}")
    if custom_configs:
        print(f"Custom configurations: {len(custom_configs)} experiments")
    print("="*70)
    
    if args.experiment in ['spatial', 'all']:
        spatial_results = runner.run_spatial_ablation(selected_spatial)
        runner.save_results(spatial_results, 'spatial')
    
    if args.experiment in ['temporal', 'all']:
        temporal_results = runner.run_temporal_ablation(selected_temporal)
        runner.save_results(temporal_results, 'temporal')
    
    if args.experiment in ['custom', 'all'] and custom_configs:
        custom_results = runner.run_custom_experiments(custom_configs)
        runner.save_results(custom_results, 'custom')
    
    # Generate report
    runner.generate_report(spatial_results, temporal_results, custom_results)
    
    duration = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"ENHANCED ABLATION STUDY COMPLETED")
    print(f"Total duration: {duration}")
    print(f"Results saved to: {runner.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()