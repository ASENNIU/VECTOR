import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.models import load_model, load_model_and_preprocess

from config.base_config import Config

logger = logging.getLogger(__name__)

class VICTOR(nn.Module):
    def __init__(self, config: Config) -> None:
        super(VICTOR, self).__init__()
        self.config = config
        self.device = config.device
        # self.blip2, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=self.config.device, is_eval=True)
        self.blip2 = load_model("blip2_image_text_matching", "pretrain", is_eval=True, device=self.config.device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, data):
        batch_size = data["video"].shape[0]
        text_input = data["text"]
        video_input = data["video"]
        
        video_input = video_input.reshape(-1, 3, self.config.input_res, self.config.input_res)

        
        video_features = self.blip2.extract_features({"image": video_input}, mode="image").image_embeds
        text_features = self.blip2.extract_features({"text_input": text_input}, mode="text").text_embeds
        
        # Assuming the first token is [CLS] or a special token for the entire sequence
        video_features = video_features[:, 0, :]
        text_features = text_features[:, 0, :]
        
        # Reshape video features
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
        
        # Mean pooling over frames
        video_features = torch.mean(video_features, dim=1)
        
        return video_features, text_features
      
      
      
        