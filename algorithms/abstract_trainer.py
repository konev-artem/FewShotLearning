from .base_trainer import BaseTrainer

class AbstractTrainer(BaseTrainer):
    def train(self, *args, **kwargs):
        pass
