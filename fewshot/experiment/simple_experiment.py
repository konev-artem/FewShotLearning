from .experiment import Experiment, Report
from ..backbones import ConvNet
from ..data_provider import Dataset


# just for example


class BackboneTrainer:
    def train(self, backbone_model, dataset):
        ...


class FewShotModel:
    def __init__(self, backbone_model):
        ...


class Trainer:
    def train(self, fewshot_model, dataset):
        ...


class Tester:
    def test(self, fewshot_model):
        # do some stuff
        return 0


class SimpleExperiment(Experiment):
    """Basic experiment with metric learning approach"""

    def __init__(self, config):
        self.config = config

        self.dataset = None

        self.backbone_model = None
        self.fewshot_model = FewShotModel(self.backbone_model)

        self.backbone_trainer = BackboneTrainer()
        self.fewshot_trainer = Trainer()
        self.tester = Tester()

    def run(self):

        # train backbone
        self.backbone_trainer.train(self.backbone_model, self.dataset)

        # train fewshot
        self.fewshot_trainer.train(self.fewshot_model, self.dataset)

        # test
        metric = self.tester.test(self.fewshot_model)

        # create report
        report = Report(metric=metric)
        return report
