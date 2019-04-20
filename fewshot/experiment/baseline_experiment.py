from .experiment import Experiment, Report
from ..backbones import ConvNet, Resnet12
from ..data_provider import Dataset

from enum import Enum


class BaselineExperiment(Experiment):
    """Basic experiment with metric learning approach"""

    class BackboneType(str, Enum):
        CONVNET = "convnet"
        RESNET12 = "resnet12"

    def __init__(self, config):
        self.config = config

        # prepare datasets
        self.dataset = Dataset(config["dataset"]["dataset_dir"])

        self.backbone_dataset, self.fewshot_dataset = self.dataset.split_by_classes(
            train_size=config["dataset"]["backbone_dataset_size"]
        )

        self.backbone_train_dataset, self.backbone_val_dataset = (
            self.backbone_dataset.split_by_objects(
                train_size=config["dataset"]["backbone_train_size"]))

        # prepare backbone
        width = height = config["backbone"]["input_size"]
        input_size = (width, height, 3)

        if config["backbone"]["type"] == self.BackboneType.CONVNET:
            self.backbone = ConvNet(input_size=input_size)
        elif config["backbone"]["type"] == self.BackboneType.RESNET12:
            self.backbone = Resnet12(input_size=input_size)

    @classmethod
    def get_availible_backbone_types(cls):
        return [backbone_type.value for backbone_type in cls.BackboneType]

    def save(self, path):
        ...

    @staticmethod
    def load(cls, path):
        ...

    def run(self):
        # .. train backbone ...

        # .. train fewshot ...

        # .. test fewshot ...

        # create report
        report = Report(metric=0.0)
        return report
