import itertools
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import ujson as json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config.base_config import Config
from datasets.video_capture import VideoCapture
from modules.basic_utils import load_json


class MSRVTTDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type='train', vis_processor=None, txt_processor=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.vis_processor = vis_processor
        self.txt_processor =txt_processor
        self.split_type = split_type
    
        
        db_file = 'data/MSRVTT/MSRVTT_data.json'
        test_csv = 'data/MSRVTT/MSRVTT_JSFUSION_test.csv'

        if config.msrvtt_train_file == '7k':
            train_csv = 'data/MSRVTT/MSRVTT_train.7k.csv'
        else:
            train_csv = 'data/MSRVTT/MSRVTT_train.9k.csv'

        self.db = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(test_csv)

            
    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)

        # 处理每一帧
        processed_imgs = []
        for img in imgs:
             # 确保图像是正确的格式 (H, W, C)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)  # 从 (C, H, W) 转换到 (H, W, C)
            
            # 将 NumPy 数组转换为 PIL Image
            img_pil = Image.fromarray(img.astype(np.uint8))
                
            # 应用视觉处理器
            if self.split_type == "train":
                processed_img = self.vis_processor["train"](img_pil)
                processed_imgs.append(processed_img)
            else:
                processed_img = self.vis_processor["eval"](img_pil)
                processed_imgs.append(processed_img)

        # 将处理后的帧堆叠成一个张量
        processed_imgs = torch.stack(processed_imgs)

        # 处理文本
        if self.split_type == "train":
            processed_caption = self.txt_processor["train"](caption)
        else:
            processed_caption = self.txt_processor["eval"](caption)

        return {
            'video_id': video_id,
            'video': processed_imgs,
            'text': processed_caption,
        }

    
    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence

        return video_path, caption, vid

    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])

            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)
