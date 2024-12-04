from torch.utils.data import DataLoader

from config.base_config import Config
from datasets.model_transforms import init_transform_dict

# from datasets.msrvtt_dataset import MSRVTTDataset
from datasets.msrvtt_dataset import MSRVTTDataset


class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type="train", shuffle=True, txt_processor=None, vis_processor=None):

        if config.dataset_name == "MSRVTT":
            if split_type == "train":
                dataset = MSRVTTDataset(config, split_type, vis_processor, txt_processor)
                return DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=shuffle,
                    num_workers=config.num_workers,
                )
            else:
                dataset = MSRVTTDataset(config, split_type, vis_processor, txt_processor)
                return DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=shuffle,
                    num_workers=config.num_workers,
                )

        else:
            raise NotImplementedError
