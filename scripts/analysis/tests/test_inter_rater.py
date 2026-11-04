"""Numerical and alignment regression tests; no dataset mutation."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('analysis', Path(__file__).parents[1]/'analyze_inter_rater.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class MetricsTest(unittest.TestCase):
    def test_constants(self):
        equal = m.pair_metrics(np.array([1.,1.]), np.array([1.,1.]))
        self.assertTrue(np.isnan(equal['pcc']))
        self.assertTrue(np.isnan(equal['ccc']))
        self.assertEqual(equal['rmse'], 0)
        different = m.pair_metrics(np.array([1.,1.]), np.array([2.,2.]))
        self.assertEqual(different['ccc'], 0)
        self.assertEqual(different['rmse'], 1)

    def test_shift_and_missing(self):
        r = m.pair_metrics(np.array([0.,2.,np.nan]), np.array([1.,3.,9.]))
        self.assertEqual(r['n_common'], 2)
        self.assertAlmostEqual(r['pcc'], 1)
        self.assertAlmostEqual(r['ccc'], 2/3)
        self.assertAlmostEqual(r['rmse'], 1)
        one = m.pair_metrics(np.array([0.,np.nan]), np.array([2.,3.]))
        self.assertTrue(np.isnan(one['ccc']))
        self.assertTrue(np.isnan(one['pcc']))
        self.assertEqual(one['rmse'], 2)
        empty = m.pair_metrics(np.array([np.nan]), np.array([1.]))
        self.assertTrue(np.isnan(empty['rmse']))

    def test_population_sd_and_insufficient_raters(self):
        stats = m.time_statistics(pd.DataFrame([[0.,2.],[0.,4.],[1.,np.nan]]))
        self.assertEqual(stats.variance.mean(), 2.5)
        self.assertEqual(stats.sd.mean(), 1.5)
        self.assertTrue(np.isnan(stats.variance.iloc[2]))

    def test_exact_alignment_and_missing_song(self):
        df = pd.DataFrame({'song_id':[1]*4,'user_id':['a','a','b','b'],
                           'timestamp':[0.,1.,1.,2.], 'valence':[0.,1.,3.,4.],
                           'arousal':[0.,np.nan,3.,4.]})
        users = pd.DataFrame({'user_id':['a','b'], 'user_number':[1,2]})
        songs, pairs, summary, times, grids = m.analyze(df, [1,2], users)
        v = pairs[pairs.axis == 'valence'].iloc[0]
        self.assertEqual(v.n_common, 1)
        self.assertEqual(v.rmse, 2)
        self.assertTrue(np.isnan(v.pcc))
        self.assertEqual(songs.iloc[0].valence_n_valid_times, 1)
        self.assertEqual(songs.iloc[1].valence_n_valid_times, 0)
        self.assertTrue(np.isnan(songs.iloc[1].valence_mean_variance))
        self.assertEqual(summary.iloc[0].pcc_n_valid_songs, 0)

if __name__ == '__main__':
    unittest.main()
