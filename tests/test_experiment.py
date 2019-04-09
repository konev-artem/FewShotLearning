import unittest

from fewshot.experiment import SimpleExperiment


class TestExperiment(unittest.TestCase):

    def test_simple_experiment(self):

        # just for example
        config = {}
        experiment = SimpleExperiment(config)
        experiment.run()

        self.assertTrue(True)
