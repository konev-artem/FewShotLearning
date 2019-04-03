import unittest

from fewshot.experiment import SimpleExperiment


class TestExperiment(unittest.TestCase):

    def test_simple_experiment(self):
        # just for example
        experiment = SimpleExperiment()
        self.assertTrue(True)
