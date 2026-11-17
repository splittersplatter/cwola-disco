"""Phase 1: background-only CWoLa closure test with per-jet EFPs.
Run python phase1.py; see README_phase1.md for interpretation and limitations.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import warnings
import h5py
import hdf5plugin  # Registers the dataset's Blosc filter.
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import fastjet as fj
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message=".*'fork' is not available.*")
    import energyflow as ef

ROOT = Path(__file__).resolve().parent


def extract(args):
    """Bounded raw-data memory; cache only compact observables and source row IDs."""
    if getattr(args, 'feature_file', None):
        with np.load(args.feature_file, allow_pickle=False) as saved:
            result = {k: saved[k] for k in ('x', 'mass', 'truth', 'row', 'jet_mass')}
            metadata = json.loads(str(saved['metadata']))
        if metadata.get('version') != 2 or metadata.get('degree') != args.degree or metadata.get('events', 0) < args.events:
            raise ValueError('Feature-cache version, EFP degree or raw-row coverage does not match the request')
        n = len(result['mass'])
        if any(len(value) != n for value in result.values()) or len(np.unique(result['row'])) != n:
            raise ValueError('Feature arrays must align and contain unique source row IDs')
        if result['x'].ndim != 2 or result['jet_mass'].shape != (n, 2) or not np.issubdtype(result['row'].dtype, np.integer) or np.any(result['row'] < 0) or np.any(result['row'] >= metadata['events']):
            raise ValueError('Invalid feature, jet-mass or source-row layout')
        if not all(np.isfinite(result[k]).all() for k in ('x', 'mass', 'jet_mass')):
            raise ValueError('Feature cache contains non-finite values')
        if not np.isin(result['truth'], [0, 1]).all():
            raise ValueError('Feature cache contains non-binary truth labels')
        keep = result['row'] < args.events
        return {k: v[keep] for k, v in result.items()}
    metadata = dict(version=2, source=str(args.data.resolve()), size=args.data.stat().st_size,
                    mtime=args.data.stat().st_mtime_ns, events=args.events, degree=args.degree,
                    source_sha256=getattr(args, 'data_sha256', None))
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    cache = ROOT / 'feature_cache' / f'jets_{key}.npz'
    if cache.exists():
        print(f'Loading cached features: {cache}', flush=True)
        with np.load(cache) as data:
            return {k: data[k] for k in ('x', 'mass', 'truth', 'row', 'jet_mass')}
    calculator = ef.EFPSet(f'd<={args.degree}', measure='hadr', beta=1., normed=True)
    definition = fj.JetDefinition(fj.antikt_algorithm, 1.0)
    result = {k: [] for k in ('x', 'mass', 'truth', 'row', 'jet_mass')}
    with h5py.File(args.data, 'r') as handle:
        values = handle['df/block0_values']
        if values.shape[1] != 2101 or not np.array_equal(handle['df/block0_items'][:], np.arange(2101)):
            raise ValueError('Unexpected HDF5 layout: expected 700 (pt,eta,phi) triplets plus label.')
        if args.events > len(values):
            raise ValueError('Requested raw-row count exceeds the HDF5 source')
        count = args.events
        for start in range(0, count, args.chunk_size):
            block = values[start:min(start + args.chunk_size, count)]
            for offset, row in enumerate(block):
                if row[-1] not in (0, 1):
                    raise ValueError('Non-binary truth label')
                p = row[:2100].reshape(700, 3)
                if not np.isfinite(p).all() or (p[:, 0] < 0).any():
                    raise ValueError(f'Invalid particles at row {start + offset}')
                p = p[p[:, 0] > 0]
                pt, eta, phi = p.T
                particles = [fj.PseudoJet(float(px), float(py), float(pz), float(e))
                             for px, py, pz, e in zip(pt*np.cos(phi), pt*np.sin(phi),
                                                      pt*np.sinh(eta), pt*np.cosh(eta))]
                sequence = fj.ClusterSequence(particles, definition)
                jets = fj.sorted_by_pt(sequence.inclusive_jets())
                if len(jets) < 2 or jets[0].pt() <= 1200:
                    continue
                mass = (jets[0] + jets[1]).m()
                if not 3100 <= mass < 3900:
                    continue
                jets = sorted(jets[:2], key=lambda j: j.m())
                features = []
                for jet in jets:
                    constituents = np.array([[c.pt(), c.rap(), c.phi()] for c in jet.constituents()])
                    features.extend(calculator.compute(constituents))
                result['jet_mass'].append([jets[0].m(), jets[1].m()-jets[0].m()])
                result['x'].append(features)
                result['mass'].append(mass)
                result['truth'].append(row[-1])
                result['row'].append(start + offset)
            if start // args.chunk_size % 40 == 0 or start + len(block) == count:
                print(f'Processed {start + len(block)}/{count}; retained {len(result["mass"])}', flush=True)
    result = {k: np.asarray(v) for k, v in result.items()}
    if not len(result['mass']) or not np.isfinite(result['x']).all():
        raise ValueError('No selected events or invalid features')
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **result, metadata=json.dumps(metadata))
    return result


class BaselineEFPNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32),
                                 nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Logits: numerically stable BCEWithLogitsLoss.


def region(mass):
    return ((mass >= 3300) & (mass < 3700)).astype(np.int64)


def plot_results(mass, score, metrics, output):
    """Plot observed k/n efficiencies with asymmetric Wilson intervals."""
    bins = np.linspace(3100, 3900, 33)
    centers = (bins[1:]+bins[:-1])/2
    n, _ = np.histogram(mass, bins)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].hist(mass, bins=bins, density=True, histtype='step', color='black', label='All held-out QCD')
    for efficiency in (.2, .05):
        threshold = metrics['cuts'][str(efficiency)]['logit_threshold']
        selected = score > threshold
        k, _ = np.histogram(mass[selected], bins)
        if selected.any():
            axes[0].hist(mass[selected], bins=bins, density=True, histtype='step', label=f'Validation top {efficiency:.0%}')
        good = n > 0
        rate = k[good]/n[good]
        # Wilson 68% intervals (z=1), valid even at zero selected counts.
        center = (rate + .5/n[good])/(1+1/n[good])
        half = np.sqrt(rate*(1-rate)/n[good]+.25/n[good]**2)/(1+1/n[good])
        lower, upper = np.clip(center-half, 0, 1), np.clip(center+half, 0, 1)
        errors = np.maximum(0, np.vstack((rate-lower, upper-rate)))
        axes[1].errorbar(centers[good], rate, yerr=errors, fmt='.', label=f'Target {efficiency:.0%}')
        axes[1].axhline(efficiency, ls=':', alpha=.5)
    for ax in axes:
        ax.axvspan(3300, 3700, color='gray', alpha=.12)
        ax.legend()
        ax.grid(alpha=.2)
    axes[0].set(title='Background-only CWoLa closure test', ylabel='Normalized density / GeV')
    axes[1].set(xlabel='Dijet invariant mass [GeV]', ylabel='QCD selection efficiency')
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/'mass_sculpting.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'events_anomalydetection.h5')
    parser.add_argument('--events', type=int, default=50000, help='Number of raw rows to process')
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=ROOT/'results_phase1')
    parser.add_argument('--plot-only', action='store_true', help='Regenerate plot from saved output without training')
    args = parser.parse_args()
    if args.plot_only:
        metrics = json.loads((args.output/'metrics.json').read_text())
        with np.load(args.output/'predictions.npz') as saved:
            plot_results(saved['mass'], saved['score'], metrics, args.output)
        print(f'Updated plot in {args.output}')
        return
    if min(args.events, args.chunk_size, args.degree, args.epochs) <= 0:
        parser.error('Counts and degree must be positive')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = extract(args)
    # Truth is used ONLY to define the simulated zero-signal control sample.
    keep = data['truth'] == 0
    x, mass, rows = data['x'][keep], data['mass'][keep], data['row'][keep]
    target = region(mass)  # CWoLa label, NOT signal truth.
    if min(np.bincount(target, minlength=2)) < 30:
        raise ValueError('Too few background SR/SB events; increase --events.')
    indices = np.arange(len(x))
    train, holdout = train_test_split(indices, test_size=.4, stratify=target, random_state=args.seed)
    valid, test = train_test_split(holdout, test_size=.5, stratify=target[holdout], random_state=args.seed)
    # Positive EFPs span many orders of magnitude. Fit all preprocessing on training only.
    x = np.log(np.maximum(x, 1e-12))
    mean, scale = x[train].mean(0), x[train].std(0)
    active = scale > 1e-8
    x = ((x[:, active] - mean[active]) / scale[active]).astype('float32')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaselineEFPNetwork(x.shape[1]).to(device)
    counts = np.bincount(target[train], minlength=2)
    weights = len(train)/(2 * counts)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train]),
                        torch.tensor(target[train], dtype=torch.float32)),
                        batch_size=256, shuffle=True, num_workers=0)

    def evaluate(ids):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(torch.from_numpy(x[part]).to(device)).cpu().numpy()
                                   for part in np.array_split(ids, max(1, (len(ids)+511)//512))])

    best_loss, history, best_state = float('inf'), [], None
    print(f'Training on {device}: {len(train)} train / {len(valid)} validation / {len(test)} test; {x.shape[1]} features')
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            w = torch.as_tensor(weights, dtype=torch.float32, device=device)[by.long()]
            loss = (criterion(model(bx), by) * w).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()*len(bx)
        logits = evaluate(valid)
        val_loss = float(np.mean((np.logaddexp(0, logits)-target[valid]*logits)*weights[target[valid]]))
        history.append([epoch+1, total/len(train), val_loss])
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch+1}: train {total/len(train):.4f}, validation {val_loss:.4f}', flush=True)
    model.load_state_dict(best_state)
    # Logits preserve score ordering and avoid saturation-induced threshold ties.
    validation_score, score = evaluate(valid), evaluate(test)
    metrics = dict(test_sr_vs_sb_auc=float(roc_auc_score(target[test], score)),
                   train_events=len(train), validation_events=len(valid), test_events=len(test), cuts={})
    for efficiency in (.2, .05):
        threshold = float(np.quantile(validation_score, 1-efficiency))
        selected = score > threshold
        sr = target[test] == 1
        metrics['cuts'][str(efficiency)] = dict(logit_threshold=threshold, selected=int(selected.sum()),
            sr_efficiency=float(selected[sr].mean()), sb_efficiency=float(selected[~sr].mean()))
    plot_results(mass[test], score, metrics, args.output)
    np.savez_compressed(args.output/'predictions.npz', mass=mass[test], score=score,
                        test_rows=rows[test], train_rows=rows[train], validation_rows=rows[valid])
    torch.save(dict(state_dict={k:v.cpu() for k,v in best_state.items()}, mean=mean,
                    scale=scale, active=active), args.output/'model.pt')
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2))
    (args.output/'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    np.savetxt(args.output/'training.csv', history, delimiter=',', header='epoch,train_bce,validation_bce')
    print(json.dumps(metrics, indent=2))
    print(f'Results saved in {args.output.resolve()}')


if __name__ == '__main__':
    main()
