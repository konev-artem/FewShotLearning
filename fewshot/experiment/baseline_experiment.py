from .experiment import Experiment, Report
from ..backbones import ConvNet, Resnet12
from ..data_provider import Dataset

from ..algorithms.fewshot_models import BaselineFewShotModel
from ..algorithms.backbone_pretrain import (
    simple_one_layer_cross_entropy_train
)

import numpy as np
import scipy.stats as st
from enum import Enum
import tqdm


class BaselineExperiment(Experiment):
    """Basic experiment with metric learning approach"""

    class BackboneType(str, Enum):
        CONVNET = "convnet"
        RESNET12 = "resnet12"

        def __str__(self):
            return self.value

    def __init__(self, config):
        self.config = config

        img_width = config["dataset"]["img_width"]
        img_height = config["dataset"]["img_height"]

        # prepare datasets
        self.dataset = Dataset(config["dataset"]["dataset_dir"],
                               csv_name=config["dataset"]["csv_name"])

        self.backbone_dataset, self.val_fewshot_dataset = self.dataset.split_by_classes(
            train_size=config["dataset"]["base_num_classes"],
            random_state=config["seed"]
        )

        self.fewshot_dataset, _ = (self.val_fewshot_dataset.split_by_classes(
            train_size=config["dataset"]["novel_num_classes"],
            random_state=config["seed"])
        )

        # prepare backbone
        input_size = (img_width, img_height, 3)
        if config["backbone"]["type"] == self.BackboneType.CONVNET:
            self.backbone = ConvNet(input_size=input_size)
        elif config["backbone"]["type"] == self.BackboneType.RESNET12:
            self.backbone = Resnet12(input_size=input_size)
        else:
            raise ValueError("Not supported backbone type")

    @classmethod
    def get_availible_backbone_types(cls):
        return [backbone_type.value for backbone_type in cls.BackboneType]

    def save(self, path):
        ...

    @staticmethod
    def load(cls, path):
        ...

    def train_backbone(self):
        simple_one_layer_cross_entropy_train(
            self.backbone,
            self.backbone_dataset.get_batch_generator(
                batch_size=self.config["backbone"]["batch_size"],
                shuffle=True
            ),
            n_epochs=self.config["backbone"]["num_epoch"]
        )

    def test_fewshot(self):
        accuracies = []
        for episode in tqdm.tqdm_notebook(range(self.config["fewshot"]["n_episodes"])):
            fewshot_model = BaselineFewShotModel(self.backbone, self.config["fewshot"]["n_way"])

            # not very simple way to prepare eposide support and query sets
            # FIXME: in future replace it with @bobbythehiver
            #  .few_shot_episode_generator() implementation
            # and @schoooler tester version
            episode_dataset, _ = self.fewshot_dataset.split_by_classes(
                train_size=self.config["fewshot"]["n_way"],
                random_state=self.config["seed"] + episode)

            # subset support and query datasets
            episode_support, episode_left = episode_dataset.split_by_objects(
                train_size=self.config["fewshot"]["k_shot"],
                random_state=self.config["seed"] + episode)
            episode_query, _ = episode_left.split_by_objects(
                train_size=self.config["fewshot"]["query_samples_per_class"],
                random_state=self.config["seed"] + episode)

            fewshot_model.fit(episode_support.get_batch_generator(
                batch_size=self.config["fewshot"]["batch_size"],
                shuffle=True),
                n_epochs=self.config["fewshot"]["batches_per_episode"]
            )

            out = fewshot_model.predict(episode_query.get_batch_generator(
                batch_size=self.config["fewshot"]["batch_size"],
                shuffle=False,
            ))

            # TODO not proper way to calc accuracy
            classes = np.array(episode_query.classes)
            acc = np.mean(classes[np.argmax(out, axis=1)] ==
                          episode_query.dataframe["class"].values)

            accuracies.append(acc)

        return accuracies

    def run(self):
        # ... train bckbone ...
        self.train_backbone()

        # ... fit fewshot ...
        accuracies = self.test_fewshot()

        # ... prepare report ...
        mean_accuracy = np.mean(accuracies)
        conf_interval = st.t.interval(0.95, len(accuracies) - 1,
                                      loc=mean_accuracy,
                                      scale=st.sem(accuracies))
        report = Report(mean_accuracy, conf_interval)
        return report
