"""Controlled CWoLa/DisCo study. See README_study.md before interpreting results."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
import csv
import json
import time
import hashlib
import platform
import importlib.metadata
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase1 import extract, region, BaselineEFPNetwork, ROOT


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def disco_squared(score, mass):
    """Biased empirical squared distance correlation, uniform event weights.

    A_ij = |s_i-s_j| - row_mean - column_mean + grand_mean; likewise B.
    dCor^2 = mean(A*B)/sqrt(mean(A^2)*mean(B^2)). O(n^2).
    Constant scores have zero penalty; sensitivity tests detect collapse.
    """
    if len(score) < 4:
        return score.sum()*0
    def centered(v):
        distance = torch.abs(v[:, None]-v[None, :])
        return distance-distance.mean(0)-distance.mean(1)[:, None]+distance.mean()
    a, b = centered(score), centered(mass)
    denominator = torch.sqrt((a.square().mean()*b.square().mean()).clamp_min(1e-20))
    return (a*b).mean().clamp_min(0)/denominator


def partitions(data, seed):
    """Fixed exploratory test; all source rows must belong to the explored prefix."""
    if np.any(data['row'] >= 200000):
        raise ValueError('This development study cannot use reserved rows >=200000')
    ids = np.arange(len(data['mass']))
    strata = 2*data['truth'].astype(int)+region(data['mass'])
    eligible = ids[data['row'] >= 50000]
    if len(eligible) < 1000:  # Smoke tests only; recorded in config/report.
        eligible = ids
    _, test = train_test_split(eligible, test_size=.25, stratify=strata[eligible], random_state=9173)
    dev = np.setdiff1d(ids, test)
    train, valid = train_test_split(dev, test_size=.25, stratify=strata[dev], random_state=seed)
    return train, valid, test


def inject(ids, truth, sr, fraction, seed):
    """Fixed background; nested random SR signal subsets. f=S/(B_SR+S)."""
    background = ids[truth[ids] == 0]
    signal = ids[(truth[ids] == 1) & (sr[ids] == 1)]
    rng = np.random.default_rng(seed)
    signal = rng.permutation(signal)
    count = int(round(fraction/(1-fraction)*np.sum(sr[background])))
    if count > len(signal):
        raise ValueError('Not enough disjoint signal events for requested injection')
    return np.concatenate((background, signal[:count]))


def prepare(data, train, include_mass):
    raw = np.log(np.maximum(data['x'], 1e-12))
    if include_mass:
        raw = np.column_stack((raw, np.log1p(np.maximum(data['jet_mass'], 0))))
    mean, scale = raw[train].mean(0), raw[train].std(0)
    active = scale > 1e-8
    return ((raw[:, active]-mean[active])/scale[active]).astype('float32'), dict(mean=mean, scale=scale, active=active)


def train_mlp(x, data, train, valid, seed, mode, strength, args):
    torch.manual_seed(seed)
    model = BaselineEFPNetwork(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    target = region(data['mass'])
    counts = np.bincount(target[train], minlength=2)
    weights = torch.tensor(len(train)/(2*counts), dtype=torch.float32)
    tx = torch.from_numpy(x)
    ty = torch.tensor(target, dtype=torch.float32)
    tm = torch.tensor(data['mass']/1000, dtype=torch.float32)
    allowed = data['truth'] == 0 if mode == 'oracle' else target == 0
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    generator = torch.Generator().manual_seed(seed+100)
    best, state, stale, history = float('inf'), None, 0, []
    rng = np.random.default_rng(seed+200)
    penalty_valid = valid[allowed[valid]]
    penalty_valid = rng.choice(penalty_valid, min(512, len(penalty_valid)), replace=False)
    def predict(ids):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(tx[part]) for part in np.array_split(ids, max(1, (len(ids)+1023)//1024))])
    for epoch in range(args.epochs):
        model.train()
        total_bce, total_penalty, batches = 0., 0., 0
        order = train[torch.randperm(len(train), generator=generator).numpy()]
        for begin in range(0, len(order), 512):
            ids = order[begin:begin+512]
            logits = model(tx[ids])
            bce = (criterion(logits, ty[ids])*weights[ty[ids].long()]).mean()
            penalty = logits.sum()*0
            if strength:
                # Deterministic shuffled batch order gives a random bounded subset.
                local = np.flatnonzero(allowed[ids])[:128]
                penalty = disco_squared(torch.sigmoid(logits[local]), tm[ids[local]])
            loss = bce+strength*penalty
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_bce += bce.item(); total_penalty += penalty.item(); batches += 1
        logits = predict(valid)
        val_bce = (criterion(logits, ty[valid])*weights[ty[valid].long()]).mean().item()
        with torch.no_grad():
            val_penalty = disco_squared(torch.sigmoid(predict(penalty_valid)), tm[penalty_valid]).item() if strength else 0.
        objective = val_bce+strength*val_penalty
        history.append([epoch+1, total_bce/batches, total_penalty/batches, val_bce, val_penalty, objective])
        if objective < best-1e-6:
            best, state, stale = objective, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(state)
    return model, history


def score_model(model, x):
    if isinstance(model, HistGradientBoostingClassifier):
        return model.predict_proba(x)[:, 1]
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(part))).numpy()
                               for part in np.array_split(x, max(1, (len(x)+1023)//1024))])


def js_stat(mass, selected, bins):
    base = np.histogram(mass, bins)[0]
    accepted = np.histogram(mass[selected], bins)[0]
    if not accepted.sum():
        return float('nan')
    return float(jensenshannon(base, accepted, base=2)**2)


def metrics_for(data, ids, score, threshold, seed, permutations):
    truth, mass = data['truth'][ids], data['mass'][ids]
    sr = region(mass).astype(bool)
    background, signal = truth == 0, (truth == 1) & sr
    selected = score > threshold
    bsr = background & sr
    eps_b, eps_s = selected[bsr].mean(), selected[signal].mean()
    bins = np.linspace(3100, 3900, 17)
    bm, bs = mass[background], selected[background]
    observed = js_stat(bm, bs, bins)
    rng = np.random.default_rng(seed)
    null = np.array([js_stat(bm, rng.permutation(bs), bins) for _ in range(permutations)])
    pick = rng.choice(np.flatnonzero(background), min(1000, background.sum()), replace=False)
    dc = disco_squared(torch.tensor(score[pick], dtype=torch.float64),
                       torch.tensor(mass[pick]/1000, dtype=torch.float64)).item()
    both = sr
    return dict(signal_auc_sr=float(roc_auc_score(truth[both], score[both])),
        bkg_region_auc=float(roc_auc_score(sr[background], score[background])),
        signal_efficiency=float(eps_s), background_efficiency_sr=float(eps_b),
        background_efficiency_sb=float(selected[background & ~sr].mean()),
        sic=float(eps_s/np.sqrt(eps_b)) if eps_b else None,
        bkg_selected_sr=int(selected[bsr].sum()), signal_selected_sr=int(selected[signal].sum()),
        bkg_test_sr=int(bsr.sum()), signal_test_sr=int(signal.sum()),
        js_divergence=observed, js_null_median=float(np.median(null)),
        js_permutation_p=float((1+np.sum(null >= observed))/(permutations+1)),
        score_mass_dcor_squared=dc, score_std=float(score[background].std()))


def summarize(records, output):
    keys = sorted(set().union(*(r.keys() for r in records)))
    with (output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(records)
    fractions = sorted({r['fraction'] for r in records})
    fig, axes = plt.subplots(1, len(fractions), figsize=(6*len(fractions), 5), squeeze=False)
    for fraction, ax in zip(fractions, axes[0]):
        group = [r for r in records if r['fraction'] == fraction and r['cut'] == .05 and r['split'] == 'test']
        for name in sorted({r['model'] for r in group}):
            subset = [r for r in group if r['model'] == name and r['sic'] is not None]
            if not subset: continue
            xx = np.array([r['js_divergence']-r['js_null_median'] for r in subset])
            yy = np.array([r['sic'] for r in subset])
            ax.errorbar(xx.mean(), yy.mean(), xerr=xx.std(), yerr=yy.std(), fmt='o', label=name, capsize=3)
        ax.axhline(1, color='gray', ls=':')
        ax.axvline(0, color='gray', ls=':')
        ax.set(title=f'Training SR signal fraction {fraction:.1%}', xlabel='JS divergence minus permutation median', ylabel='SR SIC at validation 5% SB cut')
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=7)
    fig.suptitle('Fixed test set; error bars show seed standard deviation, not independent-data uncertainty')
    fig.tight_layout(); fig.savefig(output/'tradeoff.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'events_anomalydetection.h5')
    parser.add_argument('--feature-file', type=Path, help='Previously extracted feature cache; treated as exploratory input')
    parser.add_argument('--events', type=int, default=200000)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11,22,33,44,55])
    parser.add_argument('--fractions', type=float, nargs='+', default=[0., .005, .01])
    parser.add_argument('--lambdas', type=float, nargs='+', default=[.001, .003, .01, .03])
    parser.add_argument('--permutations', type=int, default=199)
    parser.add_argument('--output', type=Path, default=ROOT/'results/development')
    args = parser.parse_args()
    if min(args.events,args.chunk_size,args.degree,args.epochs,args.permutations) <= 0 or any(not 0 <= f < 1 for f in args.fractions) or any(l <= 0 for l in args.lambdas):
        parser.error('Invalid counts, fractions or lambda strengths')
    if args.events > 200000:
        parser.error('Development is restricted to the previously explored first 200000 raw rows; later rows are reserved')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    source = args.feature_file if args.feature_file else args.data
    source_sha256 = sha256_file(source)
    args.data_sha256 = source_sha256
    config['protocol_version'] = 2
    config['source_sha256'] = source_sha256
    config['source_kind'] = 'precomputed_features' if args.feature_file else 'raw_hdf5'
    config['code_sha256'] = {p.name: sha256_file(p) for p in (Path(__file__), ROOT/'phase1.py')}
    config['python_version'] = platform.python_version()
    config['packages'] = {name: importlib.metadata.version(name) for name in
                          ('numpy', 'torch', 'scikit-learn', 'scipy', 'h5py', 'fastjet', 'energyflow')}
    config['torch_version'] = torch.__version__
    serialized = json.dumps(config, default=str, sort_keys=True, indent=2)
    config_path = args.output/'config.json'
    if config_path.exists() and config_path.read_text() != serialized:
        raise ValueError('Output already contains a different experiment; choose a new --output')
    config_path.write_text(serialized)
    data = extract(args)
    sr, truth = region(data['mass']), data['truth']
    records = []
    specifications = [('efp_mlp', False, 'none', 0.), ('mass_mlp', True, 'none', 0.), ('mass_gbdt', True, 'none', 0.)]
    specifications += [(f'{mode}_{lam:g}', True, mode, lam) for mode in ('oracle','sideband') for lam in args.lambdas]
    for seed in args.seeds:
        train_pool, valid_pool, test_pool = partitions(data, seed)
        test = test_pool[(truth[test_pool] == 0) | (sr[test_pool] == 1)]
        for fraction in args.fractions:
            train = inject(train_pool, truth, sr, fraction, seed)
            valid = inject(valid_pool, truth, sr, fraction, seed+1)
            # Independent validation signal pool is used for evaluation only, never checkpointing.
            val_eval = valid_pool[(truth[valid_pool] == 0) | (sr[valid_pool] == 1)]
            for name, include_mass, mode, strength in specifications:
                directory = args.output/f'seed{seed}_f{fraction:g}_{name}'
                done = directory/'metrics.json'
                if done.exists():
                    records.extend(json.loads(done.read_text())); continue
                directory.mkdir(exist_ok=True)
                started = time.monotonic()
                print(f'Train seed={seed}, fraction={fraction}, model={name}', flush=True)
                x, preprocessing = prepare(data, train, include_mass)
                if name == 'mass_gbdt':
                    target = sr[train]
                    counts = np.bincount(target, minlength=2)
                    model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=1., early_stopping=False, random_state=seed)
                    model.fit(x[train], target, sample_weight=len(train)/(2*counts[target]))
                    joblib.dump(model, directory/'model.joblib')
                else:
                    model, history = train_mlp(x, data, train, valid, seed, mode, strength, args)
                    torch.save(model.state_dict(), directory/'model.pt')
                    np.savetxt(directory/'training.csv', history, delimiter=',', header='epoch,train_bce,train_dcor2,val_bce,val_dcor2,val_objective')
                np.savez_compressed(directory/'preprocessing.npz', **preprocessing)
                test_score, val_score = score_model(model, x[test]), score_model(model, x[val_eval])
                calibration = sr[val_eval] == 0  # Signal injection is SR-only; SB threshold is data-driven.
                run_records = []
                for cut in (.2, .05):
                    threshold = float(np.quantile(val_score[calibration], 1-cut))
                    for split, ids, scores in [('validation',val_eval,val_score),('test',test,test_score)]:
                        row = dict(seed=seed, fraction=fraction, model=name, cut=cut, split=split, threshold=threshold,
                                   train_signal=int(truth[train].sum()), validation_signal=int(truth[valid].sum()))
                        row.update(metrics_for(data, ids, scores, threshold, seed+300, args.permutations))
                        run_records.append(row)
                np.savez_compressed(directory/'predictions.npz', test_rows=data['row'][test], test_mass=data['mass'][test],
                    test_truth=truth[test], test_score=test_score, validation_rows=data['row'][val_eval],
                    validation_mass=data['mass'][val_eval], validation_truth=truth[val_eval], validation_score=val_score,
                    train_rows=data['row'][train], checkpoint_validation_rows=data['row'][valid])
                done.write_text(json.dumps(run_records, indent=2, allow_nan=False))
                records.extend(run_records)
                print(f'Finished in {time.monotonic()-started:.1f}s', flush=True)
                summarize(records, args.output)
    summarize(records,args.output)
    print(f'Completed study: {args.output}', flush=True)


if __name__ == '__main__':
    main()
