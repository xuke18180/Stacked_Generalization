import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from omegaconf import DictConfig
from torchvision.models import vit_b_16
from enum import Enum
from pathlib import Path

@dataclass
class BaseModelConfig:
    """Configuration for base learners"""
    architecture: str
    pretrained: bool = False
    init_method: Optional[str] = None
    init_params: Optional[Dict] = None

@dataclass
class MetaLearnerConfig:
    """Configuration for meta learner"""
    hidden_dims: List[int]
    dropout_rate: float = 0.5
    
class allcnn_t(nn.Module):
    def __init__(self, c1=96, c2=192):
        super().__init__()
        def convbn(ci, co, ksz, s=1, pz=0): 
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz, stride=s, padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co))
        d = 0.3 # Dropout rate
        n_classes = 10 # 10 for CIFAR-10
        pool_size = 8  # Pooling kernel size
        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3, c1, 3, 1, 1),
            convbn(c1, c1, 3, 1, 1),
            convbn(c1, c1, 3, 2, 1),
            nn.Dropout(d),
            convbn(c1, c2, 3, 1, 1),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, c2, 3, 2, 1),
            nn.Dropout(d),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, n_classes, 1, 1),
            nn.AvgPool2d(pool_size),  # Average pooling over 8x8
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.m(x)

class BaseLearner(nn.Module):
    """Wrapper for base learners with flexible architecture"""
    def __init__(self, 
                 config: BaseModelConfig,
                 num_classes: int,
                 in_channels: int = 3):
        super().__init__()
        self.config = config
         
        if config.architecture == "allcnn_t":
            self.model = allcnn_t(
                c1=config.init_params.get('c1', 48),  # Default to 48
                c2=config.init_params.get('c2', 96)   # Default to 96
            )       
        elif config.architecture == "vit":
            self.upsample = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)  # Upsample layer
            base_model = vit_b_16(pretrained=config.pretrained)
            base_model.heads.head = nn.Linear(
              base_model.heads.head.in_features, num_classes  
            )
            self.model = base_model 
        # Initialize base architecture
        elif hasattr(models, config.architecture):
            weights = 'DEFAULT' if config.pretrained else None
            base_model = getattr(models, config.architecture)(
                weights=weights,
                num_classes=num_classes
            )
            
            # Adjust input channels if needed
            if in_channels != 3:
                first_conv = list(base_model.modules())[1]
                if isinstance(first_conv, nn.Conv2d):
                    new_conv = nn.Conv2d(
                        in_channels, 
                        first_conv.out_channels,
                        first_conv.kernel_size, 
                        first_conv.stride,
                        first_conv.padding,
                        bias=False
                    )
                    base_model.conv1 = new_conv
            self.model = base_model
        else:
            raise ValueError(f"Architecture {config.architecture} not found")
            
        #self.model = base_model
        
        # Apply custom initialization if specified
        if config.init_method:
            self._initialize_weights()
    
    def _initialize_weights(self):
        params = self.config.init_params or {}
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self.config.init_method == 'normal':
                    nn.init.normal_(m.weight, **params)
                elif self.config.init_method == 'uniform':
                    nn.init.uniform_(m.weight, **params)
                elif self.config.init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, **params)
    
    def forward(self, x):
        if self.config.architecture == "vit":
            x = self.upsample(x)
        return self.model(x)

class ImageFeatureExtractor(nn.Module):
    """CNN to reduce image dimensions before meta-learning"""
    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Linear(256, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.features(x)

class SpectralNormLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.utils.spectral_norm(
            nn.Linear(in_features, out_features)
        )
    
    def forward(self, x):
        return self.linear(x)

# In MetaLearner, replacing nn.Linear with SpectralNormLinear

class MetaLearner(nn.Module):
    """MLP-based meta-learner with configurable architecture"""
    def __init__(self, 
                 logits_dim: int,
                 image_features_dim: int,
                 num_classes: int,
                 config: MetaLearnerConfig):
        super().__init__()
        self.combined_dim = logits_dim + image_features_dim
        
        # Build MLP layers dynamically based on config
        layers = []
        input_dim = self.combined_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Final layer
        layers.extend([
            nn.Linear(input_dim, num_classes)
        ])
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, logits_features, image_features):
        combined = torch.cat([logits_features, image_features], dim=1)
        return self.mlp(combined)

class StackedModel(nn.Module):
    """Complete stacked generalization model"""
    def __init__(self,
                 base_configs: List[BaseModelConfig],
                 meta_config: MetaLearnerConfig,
                 num_classes: int,
                 criterion: Optional[nn.Module] = nn.CrossEntropyLoss(),
                 image_features_dim: int = 256,
                 alpha: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Initialize base learners
        self.base_learners = nn.ModuleList([
            BaseLearner(config, num_classes)
            for config in base_configs
        ])
        
        # Initialize image feature extractor
        self.image_extractor = ImageFeatureExtractor(
            in_channels=3,
            out_features=image_features_dim
        )
        
        # Initialize meta-learner
        total_logits_dim = num_classes * len(base_configs)
        self.meta_learner = MetaLearner(
            logits_dim=total_logits_dim,
            image_features_dim=image_features_dim,
            num_classes=num_classes,
            config=meta_config
        )
        
        self.criterion = criterion
    
    def forward(self, x):
        # Get predictions from base learners
        base_outputs = []
        base_losses = []
        
        for learner in self.base_learners:
            output = learner(x)
            base_outputs.append(output)
            
        # Stack base learner outputs
        stacked_logits = torch.cat(base_outputs, dim=1)
        # print(f"Base Learner Output Shape: {stacked_logits.shape}")
        # Get image features
        image_features = self.image_extractor(x)
        
        # Get meta-learner prediction
        meta_output = self.meta_learner(stacked_logits, image_features)
        
        return {
            'base_outputs': base_outputs,
            'meta_output': meta_output
        }
    
    def compute_losses(self, outputs: Dict, targets: torch.Tensor) -> Dict:
        base_losses = [
            self.criterion(output, targets)
            for output in outputs['base_outputs']
        ]
        
        L2 = sum(base_losses)
        L1 = self.criterion(outputs['meta_output'], targets)
        
        total_loss = L1 + self.alpha * L2
        
        return {
            'total_loss': total_loss,
            'meta_loss': L1,
            'base_loss': L2,
            'individual_base_losses': base_losses
        }

def create_model_from_config(cfg: DictConfig) -> StackedModel:
    """Create model from Hydra config"""
    base_configs = [
        BaseModelConfig(
            architecture=model_cfg.architecture,
            pretrained=model_cfg.pretrained,
            init_method=model_cfg.get('init_method'),
            init_params=model_cfg.get('init_params')
        )
        for model_cfg in cfg.model.base_learners
    ]
    
    meta_config = MetaLearnerConfig(
        hidden_dims=cfg.model.meta_learner.hidden_dims,
        dropout_rate=cfg.model.meta_learner.get('dropout_rate', 0.5)
    )
    
    return StackedModel(
        base_configs=base_configs,
        meta_config=meta_config,
        num_classes=cfg.dataset.num_classes,
        image_features_dim=cfg.model.image_features_dim,
        alpha=cfg.model.alpha
    )

class TestTimeStrategy(Enum):
    BEST_FOLD = "best_fold"      # Use only the best performing fold's models
    ALL_FOLDS = "all_folds"      # Average predictions across all folds
    WEIGHTED = "weighted"        # Weight predictions by fold performance

class StackedEnsemble(nn.Module):
    """Wrapper for inference with trained stacked models"""
    def __init__(
        self,
        fold_models: Dict[int, List[nn.Module]],  # Dict[model_idx, List[fold_models]]
        level1_model: nn.Module,
        fold_accuracies: Dict[str, List[float]],
        strategy: str = "all_folds",
        device: str = "cuda"
    ):
        super().__init__()
        self.strategy = TestTimeStrategy(strategy)
        self.fold_models = fold_models
        self.level1_model = level1_model
        self.fold_accuracies = fold_accuracies
        self.device = device
        
        # Move models to device
        self.level1_model.to(device)
        for model_idx, models in self.fold_models.items():
            for model in models:
                model.to(device)
                
        # Compute weights for weighted strategy
        if self.strategy == TestTimeStrategy.WEIGHTED:
            self._compute_fold_weights()
    
    def _compute_fold_weights(self):
        """Compute weights for each fold based on validation accuracy"""
        self.weights = {}
        for model_idx, accuracies in self.fold_accuracies.items():
            # Convert accuracies to weights using softmax
            weights = torch.tensor(accuracies)
            weights = torch.softmax(weights / 0.1, dim=0)  # Temperature of 0.1
            self.weights[model_idx] = weights.to(self.device)
    
    def _predict_best_fold(self, x: torch.Tensor) -> torch.Tensor:
        """Use only the best performing fold's models"""
        best_models = {}
        for model_idx, accuracies in self.fold_accuracies.items():
            best_fold_idx = max(range(len(accuracies)), key=lambda i: accuracies[i])
            best_models[model_idx] = self.fold_models[model_idx][best_fold_idx]
        
        # Get predictions from best models
        predictions = []
        for model in best_models.values():
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Concatenate predictions for level 1
        stacked_preds = torch.cat(predictions, dim=1)
        return self.level1_model(stacked_preds)
    
    def _predict_all_folds(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions across all folds"""
        all_level1_preds = []
        
        # For each fold
        n_folds = len(next(iter(self.fold_models.values())))
        for fold_idx in range(n_folds):
            # Get predictions from each model type for this fold
            fold_predictions = []
            for model_idx in self.fold_models:
                with torch.no_grad():
                    pred = self.fold_models[model_idx][fold_idx](x)
                    fold_predictions.append(pred)
            
            # Get level 1 prediction for this fold
            stacked_preds = torch.cat(fold_predictions, dim=1)
            level1_pred = self.level1_model(stacked_preds)
            all_level1_preds.append(level1_pred)
        
        # Average all fold predictions
        return torch.stack(all_level1_preds).mean(dim=0)
    
    def _predict_weighted(self, x: torch.Tensor) -> torch.Tensor:
        """Weight predictions by fold performance"""
        all_level1_preds = []
        all_weights = []
        
        n_folds = len(next(iter(self.fold_models.values())))
        for fold_idx in range(n_folds):
            # Get predictions from each model type for this fold
            fold_predictions = []
            fold_weight = 1.0
            
            for model_idx in self.fold_models:
                with torch.no_grad():
                    pred = self.fold_models[model_idx][fold_idx](x)
                    fold_predictions.append(pred)
                fold_weight *= self.weights[f'model_{model_idx}'][fold_idx]
            
            # Get level 1 prediction for this fold
            stacked_preds = torch.cat(fold_predictions, dim=1)
            level1_pred = self.level1_model(stacked_preds)
            
            all_level1_preds.append(level1_pred)
            all_weights.append(fold_weight)
        
        # Weight and sum predictions
        weighted_preds = torch.stack([
            pred * weight for pred, weight in zip(all_level1_preds, all_weights)
        ])
        return weighted_preds.sum(dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        
        if self.strategy == TestTimeStrategy.BEST_FOLD:
            return self._predict_best_fold(x)
        elif self.strategy == TestTimeStrategy.ALL_FOLDS:
            return self._predict_all_folds(x)
        else:  # WEIGHTED
            return self._predict_weighted(x)
    
    @classmethod
    def load(cls, 
            path: Union[str, Path], 
            strategy: str = "all_folds",
            device: str = "cuda") -> "StackedEnsemble":
        """Load a saved StackedEnsemble model"""
        path = Path(path)
        
        # Load metadata
        metadata = torch.load(path / "metadata.pt")
        fold_accuracies = metadata["fold_accuracies"]
        
        # Load level 1 model
        level1_model = torch.load(path / "level1_model.pt")
        
        # Load fold models
        fold_models = {}
        for model_idx in metadata["model_indices"]:
            fold_models[model_idx] = []
            for fold_idx in range(metadata["n_folds"]):
                model_path = path / f"model_{model_idx}_fold_{fold_idx}.pt"
                model = torch.load(model_path)
                fold_models[model_idx].append(model)
        
        return cls(fold_models, level1_model, fold_accuracies, strategy, device)
    
    def save(self, path: Union[str, Path]):
        """Save the StackedEnsemble model"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "fold_accuracies": self.fold_accuracies,
            "model_indices": list(self.fold_models.keys()),
            "n_folds": len(next(iter(self.fold_models.values())))
        }
        torch.save(metadata, path / "metadata.pt")
        
        # Save level 1 model
        torch.save(self.level1_model, path / "level1_model.pt")
        
        # Save fold models
        for model_idx, fold_models in self.fold_models.items():
            for fold_idx, model in enumerate(fold_models):
                model_path = path / f"model_{model_idx}_fold_{fold_idx}.pt"
                torch.save(model, model_path)