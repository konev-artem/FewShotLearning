import unittest

from fewshot.experiment import BasicExperiment


class TestExperiment(unittest.TestCase):

    def test_experiment(self):
        # just for example
        experiment = BasicExperiment()
        self.assertTrue(True)
