from .experiment import Experiment, Report
from ..datasets import OmniglotPreparer


class SimpleExperiment(Experiment):
    """Basic experiment with metric learning approach"""

    def setup(self, config):
        ...

    def run(self):
        report = Report(metric = 0.5)
        return report
