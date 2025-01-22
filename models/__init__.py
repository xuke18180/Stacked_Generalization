import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from omegaconf import DictConfig
from .resnet9 import ResNet9
from torchvision.models import ViT_B_16_Weights, vit_b_16
from enum import Enum
from pathlib import Path

@dataclass
class BaseModelConfig:
    """Configuration for base learners"""
    architecture: str
    pretrained: bool = False
    init_method: Optional[str] = None
    init_params: Optional[Dict] = None
    freeze_layers: int = 0  # New parameter for layer freezing

def get_base_model(architecture, pretrained, num_classes):
    """Create base model with updated ViT support"""
    # Handle ResNet9
    if architecture == "resnet9":
        return ResNet9(num_classes=num_classes)
    
    # Handle ViT
    if architecture == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        if num_classes != 1000:  # If not using ImageNet classes
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    
    # Handle ResNet family and other torchvision models
    if hasattr(models, architecture):
        weights = 'DEFAULT' if pretrained else None
        model = getattr(models, architecture)(weights=weights)
        
        # Modify final layer based on architecture type
        if num_classes != 1000:  # If not using ImageNet classes
            if architecture.startswith('resnet'):
                # ResNet family
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif architecture.startswith('densenet'):
                # DenseNet family
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif architecture.startswith('efficientnet'):
                # EfficientNet family
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            elif architecture.startswith('convnext'):
                # ConvNeXt family
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            else:
                # Generic approach for other architectures
                try:
                    if hasattr(model, 'fc'):
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                    elif hasattr(model, 'classifier'):
                        if isinstance(model.classifier, nn.Linear):
                            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                        else:
                            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
                except AttributeError as e:
                    raise ValueError(f"Unable to modify classification head for architecture {architecture}: {str(e)}")
        
        return model
        
    raise ValueError(f"Unknown architecture: {architecture}")

def freeze_layers(model, num_layers):
    """Freeze specified number of transformer layers in ViT"""
    if num_layers == -1:  # Freeze everything except classifier
        for name, param in model.named_parameters():
            if 'head' not in name:  # 'head' is classifier in ViT
                param.requires_grad = False
    elif num_layers > 0:
        # Freeze encoder blocks up to num_layers
        for i in range(num_layers):
            for param in model.encoder.layers[i].parameters():
                param.requires_grad = False

@dataclass
class MetaLearnerConfig:
    """Configuration for meta learner"""
    hidden_dims: List[int]
    dropout_rate: float = 0.5
    image_features_dim: int = 256  # Added this parameter

class BaseLearner(nn.Module):
    """Wrapper for base learners with flexible architecture"""
    def __init__(self, 
                 config: BaseModelConfig,
                 num_classes: int,
                 in_channels: int = 3):
        super().__init__()
        self.config = config
        
        # Load base model
        base_model = get_base_model(config.architecture, config.pretrained, num_classes)
            
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

        self.is_vit = config.architecture.startswith("vit")
        if config.pretrained and config.freeze_layers > 0:
            freeze_layers(base_model, config.freeze_layers)
            
        self.model = base_model
        
        # Apply custom initialization if specified
        if config.init_method and not config.pretrained:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self.config.init_method == 'kaiming_fan_out':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.config.init_method == 'kaiming_fan_in':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif self.config.init_method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=1.0)
                elif self.config.init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight) 
                elif self.config.init_method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                    
        

    
    def forward(self, x):
        if self.is_vit:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear',align_corners=False)
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

class CombinedMetaLearner(nn.Module):
    """Meta-learner that combines base model predictions with image features"""
    def __init__(self, 
                 logits_dim: int,
                 num_classes: int,
                 config: MetaLearnerConfig,
                 in_channels: int = 3):
        super().__init__()
        
        # Image feature extractor
        self.image_features = nn.Sequential(
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
            nn.Linear(256, config.image_features_dim),
            nn.BatchNorm1d(config.image_features_dim),
            nn.ReLU(inplace=True)
        )
        
        # Combined MLP
        layers = []
        input_dim = logits_dim + config.image_features_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, logits_features, images):
        # Extract image features
        image_features = self.image_features(images)

        # temporarily turn off feature extraction
        # image_features = torch.zeros_like(image_features)
        # temporarily turn off logits
        # logits_features = torch.zeros_like(logits_features)
        
        # Combine with logits
        combined = torch.cat([logits_features, image_features], dim=1)
        
        # Pass through MLP
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
        self.meta_learner = CombinedMetaLearner(
            logits_dim=total_logits_dim,
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
            init_params=model_cfg.get('init_params'),
            freeze_layers=model_cfg.get('freeze_layers', 0)
        )
        for model_cfg in cfg.model.base_learners
    ]
    
    meta_config = MetaLearnerConfig(
        hidden_dims=cfg.model.meta_learner.hidden_dims,
        dropout_rate=cfg.model.meta_learner.get('dropout_rate', 0.5),
        image_features_dim=cfg.model.image_features_dim
    )
    
    return StackedModel(
        base_configs=base_configs,
        meta_config=meta_config,
        num_classes=cfg.dataset.num_classes,
        image_features_dim=cfg.model.image_features_dim,
        alpha=cfg.model.alpha
    )

def create_base_models_from_config(cfg: DictConfig) -> List[BaseLearner]:
    """Create separate base learners and meta learner from config"""
    # Create base learners
    base_learners = [
        BaseLearner(
            config=BaseModelConfig(
                architecture=model_cfg.architecture,
                pretrained=model_cfg.pretrained,
                init_method=model_cfg.get('init_method'),
                init_params=model_cfg.get('init_params'),
                freeze_layers=model_cfg.get('freeze_layers', 0)
            ),
            num_classes=cfg.dataset.num_classes
        )
        for model_cfg in cfg.model.base_learners
    ]
    return base_learners

def create_meta_learner_from_config(cfg: DictConfig) -> CombinedMetaLearner:
    # Create meta learner
    total_logits_dim = cfg.dataset.num_classes * len(cfg.model.base_learners)
    meta_learner = CombinedMetaLearner(
        logits_dim=total_logits_dim,
        num_classes=cfg.dataset.num_classes,
        config=MetaLearnerConfig(
            hidden_dims=cfg.model.meta_learner.hidden_dims,
            dropout_rate=cfg.model.meta_learner.get('dropout_rate', 0.5),
            image_features_dim=cfg.model.image_features_dim
        )
    )
    
    return meta_learner

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
        return self.level1_model(stacked_preds, x)
    
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
            level1_pred = self.level1_model(stacked_preds, x)
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
            level1_pred = self.level1_model(stacked_preds, x)
            
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