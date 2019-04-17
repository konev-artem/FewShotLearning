import unittest

from fewshot.experiment import BaselineExperiment


class TestExperiment(unittest.TestCase):

    @unittest.skip("Not implemented yet")
    def test_simple_experiment(self):
        experiment = BaselineExperiment(None)
        experiment.run()

        # just for example
        self.assertTrue(True)
