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


class Report:
    """Class to represent experiment results"""

    def __init__(self, metric):
        self.metric = metric

    def __repr__(self):
        return "Metric: {:.3f}".format(self.metric)
