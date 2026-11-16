"""Unit tests only; running these must not start an experiment scheduler."""
import unittest
import numpy as np
from analyze_confirmation import sign,bootstrap_ci


class ConfirmationTests(unittest.TestCase):
    def test_neutral_boundaries(self):
        np.testing.assert_array_equal(sign([-2e-4,-1e-4,0,1e-4,2e-4]),[-1,0,0,0,1])

    def test_undefined_CI_not_zero(self):
        self.assertIsNone(bootstrap_ci([]))

    def test_constant_seed_bootstrap(self):
        self.assertEqual(bootstrap_ci([1.]*20),[1.,1.])


if __name__=='__main__':unittest.main()
