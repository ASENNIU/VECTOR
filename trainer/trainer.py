import logging
import os
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.base_config import Config
from modules.metrics import (
    _compute_dsl_metrics,
    _compute_metrics,
    sim_matrix,
    sim_matrix_training,
    v2t_metrics,
)
from trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    """
    Trianer class
    
    """
    
    def __init__(self, model, loss, metrics, optimizer, config: Config, 
                 train_data_loader, valid_data_loader, lr_scheduler=None, writer=None):
        super().__init__(model, loss, metrics, optimizer, config, writer)
        
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        
        self.pooling_type = config.pooling_type
        
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        
        self.config = config
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
  
        
        self.model.cuda(self.device)
        self.model.train()
        
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            data["text"] = data["text"]
            data["video"] = data["video"].to(self.device)
            
            video_embeds, text_embeds = self.mode(data)
           
            output_sim = sim_matrix_training(text_embeds, video_embeds, "avg")
            
            loss = self.loss(output_sim, self.model.logit_scale)
            
            loss.backward()
            
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
            torch.clamp_(self.model.logit_scale.data, max=np.log(100))
            
            self.global_step += 1
        
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                logger.info('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps - 1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps - 1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']
                    # TODO checkpoint和load时，需要处理哪些模块
                    self._save_checkpoint(epoch, save_best=True)
                logger.info("\t--------------------------- Best Modeling ----------------------------------")
                logger.info(" Current Best Window Average R@1 is {}".format(self.best_window))
                logger.info(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train': total_loss / num_steps
        }

        return res
    
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        
        self.model.cuda(self.device)
        self.model.eval()
        
        total_val_loss = 0.0
        text_embeds_arr = []
        vis_embeds_arr = []
        all_vis_ids = []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                data["text"] = data["text"]
                data["video"] = data["video"].to(self.device)
                video_embeds, text_embeds = self.model(data)
                text_embeds_arr.append(text_embeds)
                vis_embeds_arr.append(video_embeds)
                
                sims_batch = sim_matrix_training(text_embeds, video_embeds, "avg")
                
                curr_loss = self.loss(sims_batch, self.model.logit_scale)
                
                total_val_loss += curr_loss.item()
                
                for v_id in data['video_id']:
                    all_vis_ids.append(v_id)

            text_embeds = torch.cat(text_embeds_arr)
            vid_embeds = torch.cat(vis_embeds_arr)   
            
            
            sims = sim_matrix(text_embeds, vid_embeds)
            logger.info(f"sims shape: {sims.shape}")


            sim_matrix_state = {
                "sims": sims,
            }

            matrix_path = os.path.join(self.checkpoint_dir, 'matrix.pth')
            torch.save(sim_matrix_state, matrix_path)
            logger.info(f"Saving matrix state: {matrix_path} ...")



            total_val_loss = total_val_loss / len(self.valid_data_loader)

            res = _compute_metrics(sims)
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            logger.info(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}, loss: {total_val_loss}-----\n")
            logger.info(f"\t>>> Text -> Video:")
            logger.info(f"\t>>> R@1: {res['R1']}, R@5: {res['R5']}, R@10: {res['R10']}, R@50: {res['R50']}, MedR: {res['MedR']}, MeanR: {res['MeanR']}")
            logger.info(f"\t>>> window:")
            logger.info(f"\t>>> R@1: {res['R1-window']}, R@5: {res['R5-window']}, R@10: {res['R10-window']}, R@50: {res['R50-window']}, MedR: {res['MedR-window']}, MeanR: {res['MeanR-window']}")

            # res_v2t = v2t_metrics(sims)
            res_v2t = _compute_metrics(sims.T)
            logger.info(f"\t>>> Video -> Text:")
            logger.info(
                f"\t>>> R@1: {res_v2t['R1']}, R@5: {res_v2t['R5']}, R@10: {res_v2t['R10']}, R@50: {res_v2t['R50']}, MedR: {res_v2t['MedR']}, MeanR: {res_v2t['MeanR']}\n")
            logger.info("\t--------------------------- Metrics after DSL ----------------------------------")
            # res_v2t_dsl = v2t_metrics_dsl(sims)
            res_v2t_dsl = _compute_dsl_metrics(sims.T)
            # res_t2v_dsl = t2v_metrics_dsl(sims)
            res_t2v_dsl = _compute_dsl_metrics(sims)
            logger.info(f"\t>>> Text -> Video DSL:")
            logger.info(
                f"\t>>> R@1: {res_t2v_dsl['R1']}, R@5: {res_t2v_dsl['R5']}, R@10: {res_t2v_dsl['R10']}, R@50: {res_t2v_dsl['R50']}, MedR: {res_t2v_dsl['MedR']}, MeanR: {res_t2v_dsl['MeanR']}")
            logger.info(f"\t>>> Video -> Text DSL:")
            logger.info(
                f"\t>>> R@1: {res_v2t_dsl['R1']}, R@5: {res_v2t_dsl['R5']}, R@10: {res_v2t_dsl['R10']}, R@50: {res_v2t_dsl['R50']}, MedR: {res_v2t_dsl['MedR']}, MeanR: {res_v2t_dsl['MeanR']}\n")


            res['loss_val'] = total_val_loss

            logger.info(f"\t>>> Cur Evaluation Loss: {total_val_loss}")

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res
