import unittest

import numpy as np
from fewshot.algorithms.fewshot_test import bootstrap


class TesterTest(unittest.TestCase):

    def test_bootstrap(self):
        np.random.seed(42)
        accuracy = [np.random.rand() for i in range(1000)]
        right_answers = (0.4902565533201336, 0.291991256731072, 0.47250089680602275, 0.5074194264083435)
        answers = bootstrap(accuracy, sz = len(accuracy), verbose = False, seed = 42)
        self.assertTrue(np.array_equal(right_answers, answers))
