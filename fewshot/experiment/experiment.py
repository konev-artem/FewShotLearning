from abc import ABC, abstractmethod


class Experiment(ABC):
    """Basic class for all experiments.
    Every new experiment design we should
    inherit from this one.
    """

    @abstractmethod
    def setup(self, config):
        ...

    @abstractmethod
    def run(self):
        """Main method of experiment.
        Do all stuff (train, test) calc metrics,
        create reports.
        """
        ...
