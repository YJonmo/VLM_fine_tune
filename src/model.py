import torch.nn as nn
import torch 
from typing import Dict, Union, Callable
from .utils import PneumoniaDict, CLIPClassifierOutput, CLIPOutput, clip_loss

class PneumoniaClassifier(nn.Module):
    """
    Neural network model for pneumonia classification using a backbone vision-language model.
    
    Args:
        backbone_model: Pre-trained vision-language model (BiomedCLIP or CLIP) to use as feature extractor
        hidden_size (int): Size of the hidden layer in the classifier head. Default: 1024
        num_classes (int): Number of output classes. Default: 2 
        freeze_backbone (bool): Whether to freeze backbone model parameters. Default: True
        input_ids (Union[Dict, PneumonidaDict, None]): Dictionary mapping labels to IDs. Default: None
    
    The model architecture consists of:
    1. A frozen backbone vision-language model for feature extraction
    2. A classifier head with:
        - Linear layer mapping backbone features to hidden_size 
        - ReLU activation
        - Dropout (p=0.2)
        - Linear layer mapping to num_classes
    """
    def __init__(self, backbone_model, 
                 hidden_size:int=1024, 
                 num_classes=2, 
                 freeze_backbone:bool=True,
                 label_dict:Union[Dict, PneumoniaDict]=None,
                 criterion:Callable=None
                 ):
        super().__init__()
        self.criterion = criterion
        self.label_dict = label_dict
        self.backbone = backbone_model.vision_model  # BiomedCLIP or CLIP image encoder
        # Freeze backbone parameters
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "post_layernorm" in name or "final_layer_norm" in name or "_projection" in name:
                    continue
                param.requires_grad = False

        backbone_hidden_size = backbone_model.vision_model.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(backbone_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, inputs:Dict, groundtruth)->CLIPClassifierOutput:
        features = self.backbone(pixel_values=inputs['pixel_values'],
                                #  input_ids=inputs['input_ids'],
                                #  attention_mask=inputs['attention_mask'],
                                 return_dict=True)
        features = features.last_hidden_state[:,0,:]
        pred = self.classifier(features)
        loss = self.criterion(pred, groundtruth.to(pred.device))
        return CLIPClassifierOutput(pred, loss)




class CustomCLIP(nn.Module):
    """
    Custom CLIP model that wraps the original CLIP model components.
    Args:
        clip_model: Original CLIP model to extract components from
    The model contains:
        - vision_model: CLIP vision encoder
        - text_model: CLIP text encoder 
        - text_projection: Projection layer for text features
        - visual_projection: Projection layer for image features
        - logit_scale: Learned temperature parameter
    """
    def __init__(self, clip_model):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        self.visual_projection = clip_model.visual_projection
        
    def forward(self, inputs:Dict[str, torch.Tensor], _, return_loss:bool=True)->CLIPOutput:
        vision_outputs = self.vision_model(
            pixel_values=inputs['pixel_values'],
            return_dict=True,
        )
        text_outputs = self.text_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True,
        )
        image_feature = vision_outputs[1]
        image_feature = self.visual_projection(image_feature)

        text_feature = text_outputs[1]
        text_feature = self.text_projection(text_feature)

        # normalized features
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = text_feature @ image_feature.t()
        logits_per_text = logits_per_text * self.logit_scale.exp()

        logits_per_image = logits_per_text.t()

        if return_loss:
            loss = clip_loss(logits_per_text)
        else:
            loss = torch.tensor([0.0], device=text_feature.device)

        return  CLIPOutput(
                    logits_per_image=logits_per_image,
                    logits_per_text=logits_per_text,
                    loss = loss,
                    text_feature=text_feature,
                    image_feature=image_feature
                )

