from .experiment import Experiment, Report
from ..backbones import ConvNet, Resnet12
from ..data_provider import Dataset

from enum import Enum


class SimpleExperiment(Experiment):
    """Basic experiment with metric learning approach"""

    class BackboneType(str, Enum):
        CONVNET = "convnet"
        RESNET12 = "resnet12"

    def __init__(self, config):
        self.config = config

        self.dataset = Dataset(config["dataset"]["dataset_dir"])

        width = height = config["backbone"]["input_size"]
        input_size = (width, height, 3)

        if config["backbone"]["type"] == self.BackboneType.CONVNET:
            self.backbone = ConvNet(input_size=input_size)
        elif config["backbone"]["type"] == self.BackboneType.RESNET12:
            self.backbone = Resnet12(input_size=input_size)

    @classmethod
    def get_availible_backbone_types(cls):
        return [backbone_type.value for backbone_type in cls.BackboneType]

    def run(self):
        metric = 0
        # create report
        report = Report(metric=metric)
        return report
