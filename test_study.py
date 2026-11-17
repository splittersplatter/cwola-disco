import unittest
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from study import disco_squared, inject, partitions
from phase1 import extract

class ScientificChecks(unittest.TestCase):
    def test_distance_correlation(self):
        x = torch.linspace(-1, 1, 100, dtype=torch.float64, requires_grad=True)
        self.assertAlmostEqual(disco_squared(x,x).item(), 1., places=10)
        self.assertEqual(disco_squared(x*0,x).item(),0.)
        loss=disco_squared(x,x.square())
        self.assertGreater(loss.item(),.1)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertAlmostEqual(disco_squared(x.detach(),x.square().detach()).item(),
                               disco_squared(x.detach()*5+3,x.square().detach()*2).item(),places=8)
    def test_independence_and_constant_gradient(self):
        # Cartesian product has exactly independent empirical marginals.
        grid=torch.linspace(-1,1,10,dtype=torch.float64)
        x=grid.repeat_interleave(10)
        y=grid.repeat(10)
        self.assertAlmostEqual(disco_squared(x,y).item(),0.,places=12)
        constant=torch.ones(20,dtype=torch.float64,requires_grad=True)
        disco_squared(constant,torch.arange(20,dtype=torch.float64)).backward()
        self.assertTrue(torch.isfinite(constant.grad).all())

    def test_injection_and_splits(self):
        truth=np.tile([0,0,0,1],1000)
        mass=np.tile([3200,3400,3800,3500],1000)
        data=dict(truth=truth,mass=mass,row=np.arange(60000,64000))
        tr,va,te=partitions(data,11)
        self.assertFalse(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
        self.assertEqual(set(te),set(partitions(data,22)[2]))
        sr=((mass>=3300)&(mass<3700)).astype(int)
        zero=inject(tr,truth,sr,0,1)
        one=inject(tr,truth,sr,.01,1)
        two=inject(tr,truth,sr,.02,1)
        self.assertFalse(truth[zero].any())
        self.assertTrue(set(one)<=set(two))
        self.assertEqual(int(truth[one].sum()),round(.01/.99*sr[zero].sum()))

    def test_reserved_rows_cannot_enter_development(self):
        truth=np.tile([0,0,0,1],1000)
        mass=np.tile([3200,3400,3800,3500],1000)
        data=dict(truth=truth,mass=mass,row=np.arange(198000,202000))
        with self.assertRaisesRegex(ValueError, 'reserved'):
            partitions(data,11)

    def test_feature_cache_coverage_and_degree_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'features.npz'
            np.savez(path, x=np.ones((4, 3)), mass=np.full(4, 3400.),
                     truth=np.array([0, 1, 0, 1]), row=np.array([1, 3, 5, 7]),
                     jet_mass=np.ones((4, 2)),
                     metadata=json.dumps({'version': 2, 'degree': 4, 'events': 8}))
            args = argparse.Namespace(feature_file=path, degree=4, events=6)
            np.testing.assert_array_equal(extract(args)['row'], [1, 3, 5])
            args.events = 9
            with self.assertRaisesRegex(ValueError, 'coverage'):
                extract(args)
            args.events, args.degree = 6, 3
            with self.assertRaisesRegex(ValueError, 'degree'):
                extract(args)

if __name__=='__main__': unittest.main()
