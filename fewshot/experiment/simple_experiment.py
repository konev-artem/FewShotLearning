from .experiment import Experiment
from ..datasets import OmniglotPreparer


class SimpleExperiment(Experiment):
    """Basic experiment with metric learning approach"""

    def setup(self, config):
        ...

    def run(self):
        ...
