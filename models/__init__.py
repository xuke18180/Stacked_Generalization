import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from omegaconf import DictConfig
from .resnet9 import ResNet9
from torchvision.models import ViT_B_16_Weights, vit_b_16

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
        dropout_rate=cfg.model.meta_learner.get('dropout_rate', 0.5)
    )
    
    return StackedModel(
        base_configs=base_configs,
        meta_config=meta_config,
        num_classes=cfg.dataset.num_classes,
        image_features_dim=cfg.model.image_features_dim,
        alpha=cfg.model.alpha
    )