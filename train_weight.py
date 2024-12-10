import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from omegaconf import DictConfig
import numpy as np

@dataclass
class BaseModelConfig:
    """Configuration for base learners"""
    architecture: str
    pretrained: bool = False
    init_method: Optional[str] = None
    init_params: Optional[Dict] = None

class BaseLearner(nn.Module):
    """Wrapper for base learners with flexible architecture"""
    def __init__(self, 
                 config: BaseModelConfig,
                 num_classes: int,
                 in_channels: int = 3,
                 previous_weights: List[torch.Tensor] = None):
        super().__init__()
        self.config = config
        
        # Initialize base architecture
        if hasattr(models, config.architecture):
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
        else:
            raise ValueError(f"Architecture {config.architecture} not found")
        
        self.model = base_model
        
        # Apply custom initialization or orthogonalization if specified
        if config.init_method:
            self._initialize_weights()
        elif previous_weights:
            self._initialize_orthogonal_weights(previous_weights)
    
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
    
    def _initialize_orthogonal_weights(self, previous_weights):
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    weight = m.weight.view(-1)
                    new_weight = weight.clone().detach()
                    for prev_weight in previous_weights:
                        proj = torch.dot(new_weight, prev_weight) / torch.dot(prev_weight, prev_weight)
                        new_weight -= proj * prev_weight
                    new_weight /= new_weight.norm()
                    m.weight.copy_(new_weight.view_as(m.weight))
    
    def forward(self, x):
        return self.model(x)

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
        self.base_learners = nn.ModuleList()
        previous_weights = []
        
        for i, config in enumerate(base_configs):
            if i == 0:
                learner = BaseLearner(config, num_classes)
            else:
                learner = BaseLearner(config, num_classes, previous_weights=previous_weights)
            self.base_learners.append(learner)
            with torch.no_grad():
                previous_weights.append(learner.model.fc.weight.view(-1).clone())
        
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
