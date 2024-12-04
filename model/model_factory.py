from config.base_config import Config

from .model_victor import VICTOR


class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == "victor":
            return VICTOR(config)
