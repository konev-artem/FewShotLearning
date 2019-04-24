from abc import ABC, abstractmethod


class Experiment(ABC):
    """Basic class for all experiments.
    Every new experiment design we should
    inherit from this one.
    """

    @abstractmethod
    def run(self):
        """Main method of experiment.
        Do all stuff (train, test) calc metrics,
        create and return report.
        """
        ...

    @abstractmethod
    def save(self, path):
        """Save experiment"""
        ...

    @staticmethod
    @abstractmethod
    def load(cls, path):
        """Load experiment"""
        ...


class Report:
    """Class to represent experiment results"""

    def __init__(self, mean_accuracy, conf_interval):
        self.mean_accuracy = mean_accuracy
        self.conf_interval = conf_interval

    def __repr__(self):
        return "Metric: {:.2f}% +- {:.2f}%".format(
            self.mean_accuracy * 100,
            (self.mean_accuracy - self.conf_interval[0]) * 100
        )
