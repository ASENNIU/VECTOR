import os
import random
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


import numpy as np
import torch
from lavis.common.registry import registry
from lavis.models import load_preprocess
from omegaconf import OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter

from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.basic_utils import get_logger
from modules.loss import LossFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.optimization import AdamW, get_cosine_schedule_with_warmup
from trainer.trainer import Trainer


def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    # torch.autograd.set_detect_anomaly(True)
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    global logger
    log_path = os.path.join(config.model_path, "log.txt")
    logger = get_logger(log_path)
    
    logger.info("#################### EXPERIMENTS SETUP ####################")
    logger.info(f"exp_name: {config.exp_name}, model_path: {config.model_path}")
    logger.info(f"loss: {config.loss}")
    logger.info(f"device: {config.device}")
    logger.info(f"load epoch: {config.load_epoch}")
    logger.info(f"epoch nums: {config.num_epochs}")

    # get livas processor
    model_cls = registry.get_model_class("blip2_image_text_matching")
    cfg = OmegaConf.load(model_cls.default_config_path("pretrain"))
    preprocess_cfg = cfg.preprocess
    vis_processor, txt_processor = load_preprocess(preprocess_cfg)
    
    # prepare dataloader
    train_data_loader = DataFactory.get_data_loader(config, split_type='train', shuffle=True, 
                                                    vis_processor=vis_processor, txt_processor=txt_processor)
    valid_data_loader = DataFactory.get_data_loader(config, split_type='test', shuffle=False, 
                                                    vis_processor=vis_processor, txt_processor=txt_processor)
    
    model = ModelFactory.get_model(config)
    
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented
      
    params = model.parameters()
    optimizer = AdamW(params, lr=config.noclip_lr, weight_decay=config.weight_decay)
    
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    
    loss = LossFactory.get_loss(config)
    
    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer
                      )
    if config.load_epoch is not None:
        logger.info("load_epoch: %s", config.load_epoch)
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.finetuning_from_load_checkpoint("model_best.pth")

    if config.mode == "train":
        trainer.train()
    elif config.mode == "test":
        trainer.validate()


if __name__ == '__main__':
    main()