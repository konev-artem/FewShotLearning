from .experiment import Experiment, Report
from ..backbones import ConvNet, Resnet12
from ..data_provider import Dataset

from ..algorithms.fewshot_models import BaselineFewShotModel
from ..algorithms.backbone_pretrain import (
    build_one_layer_classifier,
    cross_entropy_train
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

        # prepare datasets
        self.dataset = Dataset(config["dataset"]["dataset_dir"])

        self.backbone_dataset, self.fewshot_dataset = self.dataset.split_by_classes(
            train_size=config["backbone"]["num_classes"],
            random_state=config["seed"]
        )

        self.backbone_train_dataset, self.backbone_val_dataset = (
            self.backbone_dataset.split_by_objects(
                train_size=config["backbone"]["train_size"],
                random_state=config["seed"])
        )

        # prepare backbone
        img_width = config["dataset"]["img_width"]
        img_height = config["dataset"]["img_height"]
        input_size = (img_width, img_height, 3)

        if config["backbone"]["type"] == self.BackboneType.CONVNET:
            self.backbone = ConvNet(input_size=input_size)
        elif config["backbone"]["type"] == self.BackboneType.RESNET12:
            self.backbone = Resnet12(input_size=input_size)
        else:
            raise ValueError("Not supported backbone type")

        self.backbone_classifier = build_one_layer_classifier(self.backbone,
                                                              self.backbone_train_dataset.n_classes)

    @classmethod
    def get_availible_backbone_types(cls):
        return [backbone_type.value for backbone_type in cls.BackboneType]

    def save(self, path):
        ...

    @staticmethod
    def load(cls, path):
        ...

    def train_backbone(self):
        cross_entropy_train(
            self.backbone_classifier,
            self.backbone_train_dataset.get_batch_generator(
                batch_size=self.config["backbone"]["batch_size"],
                target_size=(self.config["dataset"]["img_width"],
                             self.config["dataset"]["img_height"]),  # FIXME: redundant
                shuffle=True
            ),
            validation_dataset=self.backbone_val_dataset.get_batch_generator(
                batch_size=self.config["backbone"]["batch_size"],
                target_size=(self.config["dataset"]["img_width"],
                             self.config["dataset"]["img_height"]),  # FIXME: redundant
                shuffle=False),
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
                target_size=(self.config["dataset"]["img_width"],
                             self.config["dataset"]["img_height"]),  # FIXME: redundant
                shuffle=True),
                n_epochs=self.config["fewshot"]["batches_per_episode"]
            )

            out = fewshot_model.predict(episode_query.get_batch_generator(
                batch_size=self.config["fewshot"]["batch_size"],
                target_size=(self.config["dataset"]["img_width"],
                             self.config["dataset"]["img_height"]),  # FIXME: redundant
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
