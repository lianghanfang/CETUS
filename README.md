# 初始版本 新版本整理完成后上传

python .\ablation\experiments\run_ablation.py --config .\configs\base_config.json --output ablation_results --experiment spatial --spatial-encoders "knn_base"



# Spatial Event Mamba - Restructured Project

A clean, modular implementation of spatial-aware event processing using Mamba architecture with comprehensive ablation studies.

## Project Structure

```
spatial_event_mamba/
├── configs/                           # Configuration management
│   ├── config.py                      # Unified config manager
│   └── base_config.json              # Base configuration
│
├── core/                              # Main implementation
│   ├── models/                        # Core models
│   │   ├── spatial_mamba_model.py     # Main Mamba model
│   │   ├── spatial_encoder.py         # KNN spatial encoder
│   │   └── mamba_blocks.py           # Mamba building blocks
│   ├── data/
│   │   └── spatial_event_dataset.py  # Dataset implementation
│   ├── training/
│   │   ├── train.py                   # Training logic
│   │   └── losses.py                  # Loss functions
│   └── inference/
│       └── inference.py              # Inference engine
│
├── ablation/                          # Ablation experiments
│   ├── models/
│   │   ├── ablation_model.py          # Unified ablation model
│   │   ├── spatial_variants/          # Spatial encoder variants
│   │   │   ├── knn_variants.py        # KNN aggregation methods
│   │   │   ├── voxel_encoder.py      # Micro-voxel encoder
│   │   │   └── grid_encoder.py       # Grid-based encoder
│   │   └── temporal_variants/         # Temporal model variants
│   │       ├── rnn_variants.py        # RNN/LSTM/GRU
│   │       ├── transformer_variants.py # Causal Transformer
│   │       └── tcn_variants.py        # Temporal CNN
│   └── experiments/
│       └── run_ablation.py           # Ablation experiment runner
│
├── evaluation/                        # Evaluation tools
│   ├── pixel_based_eval.py           # Point-level metrics
│   ├── eval_like_paper.py            # Paper-style evaluation
│   └── norm_threshold.py             # Threshold analysis
│
├── scripts/                           # Entry points
│   ├── train.py                       # Training script
│   └── inference.py                  # Inference script
│
└── utils/                             # Utilities
    └── export_pointclouds.py         # Point cloud export
```

## Key Features

### Clean Separation
- **Core module**: Clean, efficient main implementation
- **Ablation module**: Isolated experiments without polluting core code
- **Clear interfaces**: Well-defined APIs between components

### Spatial Encoding Variants
1. **KNN Aggregation Methods**:
   - Attention-based (default)
   - Mean pooling
   - Max pooling
   - Weighted mean
   - Gated fusion

2. **Alternative Spatial Encoders**:
   - Micro-voxel encoder
   - Grid-based encoder
   - Causal vs non-causal variants

### Temporal Model Variants
1. **Mamba** (default): State-space model with linear complexity
2. **RNN variants**: GRU, LSTM with different layer configs
3. **Transformer**: Causal attention with RoPE
4. **TCN**: Temporal convolutional networks with dilations
5. **Baseline**: No temporal modeling (spatial only)

### Causal History Ablation
- History length: 0, 200, 400 samples
- Impact on spatial context and performance

## Quick Start

### Basic Training
```bash
# Train main model
python scripts/train.py --config configs/base_config.json

# Train ablation model
python scripts/train.py --model_type ablation \
    --spatial_type knn --temporal_type mamba --history_len 200
```

### Inference
```bash
# Basic inference
python scripts/inference.py --checkpoint best_model.pth \
    --test_dir ./data/test --output predictions.txt

# Streaming inference
python scripts/inference.py --checkpoint best_model.pth \
    --test_dir ./data/test --use_streaming --step_k 4
```

### Ablation Experiments
```bash
# Run all ablation studies
python ablation/experiments/run_ablation.py --config configs/base_config.json

# Run specific ablation group
python ablation/experiments/run_ablation.py --config configs/base_config.json \
    --group spatial  # or temporal, causal
```

## Usage Examples

### Custom Spatial Encoder
```python
from ablation.models.spatial_variants import create_spatial_encoder

config = {
    'spatial_encoder_type': 'knn_weighted_mean',
    'input_dim': 5,
    'output_dim': 64,
    'knn': {
        'k_neighbors': 16,
        'aggregation': 'weighted_mean',
        'causal': True
    }
}
encoder = create_spatial_encoder(config)
```

### Custom Temporal Model
```python
from ablation.models.temporal_variants import create_temporal_model

config = {
    'temporal_type': 'tcn',
    'd_model': 64,
    'temporal_params': {
        'num_layers': 4,
        'kernel_size': 3,
        'dilations': [1, 2, 4, 8]
    }
}
temporal_model = create_temporal_model(config)
```

### Ablation Model Creation
```python
from ablation.models.ablation_model import AblationEventModel

model = AblationEventModel(
    input_dim=5,
    num_classes=2,
    hidden_dim=64,
    spatial_config={
        'spatial_encoder_type': 'knn',
        'knn': {'k_neighbors': 16, 'aggregation': 'attention'}
    },
    temporal_config={
        'temporal_type': 'mamba',
        'num_blocks': 1
    }
)
```

## Configuration System

The project uses a hierarchical configuration system:

1. **Base configs** in `configs/base_config.json`
2. **Model-specific configs** through `ConfigManager`
3. **Experiment configs** generated dynamically

### Config Structure
```json
{
  "dataset": { "train_dir": "...", "causal_history_len": 200 },
  "model": {
    "type": "AblationEventModel",
    "spatial_config": { "spatial_encoder_type": "knn" },
    "temporal_config": { "temporal_type": "mamba" }
  },
  "training": { "batch_size": 8, "num_epochs": 50 },
  "inference": { "use_streaming": false, "chunk_size": 2000 }
}
```

## Import Paths

### Core Modules
```python
from core.models import SpatialAwareEventMamba, SpatialKNNEncoder
from core.data import SpatialEventDataset
from core.training import Trainer
from core.inference import InferenceEngine
```

### Ablation Modules
```python
from ablation.models import AblationEventModel, create_spatial_encoder
from ablation.experiments import AblationExperimentRunner
```

### Configuration
```python
from configs.config import get_config, ConfigManager, create_experiment_config
```

## Evaluation

Multiple evaluation metrics supported:
- **Point-level**: Pd, Fa, IoU, Accuracy, F1
- **Temporal windows**: Detection with different time windows
- **Paper-style**: Connected components analysis
- **Threshold analysis**: Performance across thresholds

```bash
# Point-level evaluation
python evaluation/pixel_based_eval.py predictions.txt

# Paper-style evaluation
python evaluation/eval_like_paper.py --input predictions.txt

# Comprehensive analysis
python evaluation/norm_threshold.py --input predictions.txt \
    --outdir analysis_results
```

## Key Design Principles

1. **Separation of Concerns**: Core functionality separated from experiments
2. **Factory Pattern**: Configurable model/encoder creation
3. **Unified Interface**: Consistent APIs across variants
4. **Extensibility**: Easy to add new variants
5. **Reproducibility**: Seed control and config versioning

## Ablation Study Results Structure

```
ablation_results/
├── spatial_ablation_results.csv       # Spatial encoder comparison
├── temporal_ablation_results.csv      # Temporal model comparison  
├── causal_ablation_results.csv        # History length analysis
├── core_ablation_report.md            # Summary report
└── individual_experiments/             # Detailed per-experiment results
```

## Performance Considerations

- **Memory**: Efficient chunking for large sequences
- **Speed**: Causal KNN for streaming inference
- **Scalability**: Configurable batch sizes and history lengths
- **Hardware**: CUDA acceleration with fallback to CPU

This restructured codebase maintains clean separation between core functionality and experimental variants while providing comprehensive ablation study capabilities.
