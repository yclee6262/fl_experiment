import unittest
import analyze_study


class AnalysisTests(unittest.TestCase):
    def test_dead_zone_boundaries(self):
        self.assertEqual([analyze_study.direction(x,1e-4) for x in [-2e-4,-1e-4,0,1e-4,2e-4]],[-1,0,0,0,1])

    def test_bootstrap_constant_and_mixed(self):
        self.assertEqual(analyze_study.ci([1]*10,20261016),[1.,1.])
        low,high=analyze_study.ci([0,1]*5,20261016)
        self.assertLess(low,.5);self.assertGreater(high,.5)
        self.assertEqual([low,high],analyze_study.ci([0,1]*5,20261016))


if __name__=='__main__':unittest.main()
