"""Generate auditable study report and held-out mass-efficiency comparisons."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from study import ROOT, extract, partitions, region


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results_study')
    args=parser.parse_args()
    config=json.loads((args.output/'config.json').read_text())
    rows=[]
    for path in sorted(args.output.glob('seed*/metrics.json')):
        rows.extend(json.loads(path.read_text()))
    expected=len(config['seeds'])*len(config['fractions'])*(3+2*len(config['lambdas']))
    completed=len(list(args.output.glob('seed*/metrics.json')))
    lines=['# CWoLa / DisCo experimental report', '', f'Completed fits: {completed}/{expected}.',
           '','Values below are means +/- seed standard deviations on a shared test set.',
           'Signal fractions refer to injected training SR mixtures; test signal is held out.',
           'The 5% cut is fixed by validation sidebands, not test data.', '',
           '| Training fraction | Model | SR SIC | SR signal efficiency | SR background efficiency | JS | JS null median |',
           '|---|---|---|---|---|---|---|']
    for fraction in config['fractions']:
        subset=[r for r in rows if r['fraction']==fraction and r['split']=='test' and r['cut']==.05]
        for name in sorted({r['model'] for r in subset}):
            group=[r for r in subset if r['model']==name]
            def display(key):
                values=np.array([r[key] for r in group if r[key] is not None])
                return f'{values.mean():.4f} +/- {values.std():.4f}' if len(values) else 'undefined'
            lines.append(f'| {fraction:.1%} | {name} | '+ ' | '.join(display(k) for k in ['sic','signal_efficiency','background_efficiency_sr','js_divergence','js_null_median'])+' |')
    lines += ['', '## Interpretation limits', '',
        '- Predefined configurations are all reported; no winner is chosen using the test set.',
        '- Zero-signal training still has held-out signal evaluation; this is not a signal injection into the null training sample.',
        '- Oracle regularization uses truth-background labels; sideband regularization does not.',
        '- A lower distortion metric alone is insufficient: inspect signal efficiency and score variation.',
        '- Seed spreads share a fixed test sample and do not quantify independent-data uncertainty.',
        '- SIC is not a bump significance. No background fit or discovery p-value is provided.',
        '- This is a reduced degree-4 EFP study, not a reproduction of the full paper.']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    paired=[]
    for row in rows:
        if row['split']!='test' or not row['model'].startswith(('oracle','sideband')): continue
        reference=next((r for r in rows if r['split']=='test' and r['model']=='mass_mlp'
            and r['seed']==row['seed'] and r['fraction']==row['fraction'] and r['cut']==row['cut']),None)
        if reference is None: continue
        paired.append(dict(seed=row['seed'],fraction=row['fraction'],cut=row['cut'],model=row['model'],
            delta_sic=row['sic']-reference['sic'] if row['sic'] is not None and reference['sic'] is not None else None,
            delta_js=row['js_divergence']-reference['js_divergence'],
            delta_dcor_squared=row['score_mass_dcor_squared']-reference['score_mass_dcor_squared']))
    if paired:
        with (args.output/'paired_differences.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)

    # Representative seed is fixed in advance as the first configured seed.
    seed=config['seeds'][0]
    for fraction in config['fractions']:
        fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True)
        bins=np.linspace(3100,3900,17); centers=(bins[1:]+bins[:-1])/2
        baseline_drawn=False
        for path in sorted(args.output.glob(f'seed{seed}_f{fraction:g}_*/predictions.npz')):
            model=path.parent.name.split(f'_f{fraction:g}_')[1]
            displayed = ('efp_mlp', 'mass_mlp', 'mass_gbdt',
                         f"oracle_{config['lambdas'][0]:g}", f"sideband_{config['lambdas'][0]:g}")
            if model not in displayed: continue
            records=json.loads((path.parent/'metrics.json').read_text())
            threshold=next(r['threshold'] for r in records if r['cut']==.05 and r['split']=='test')
            with np.load(path) as d:
                background=d['test_truth']==0
                mass=d['test_mass'][background]; selected=d['test_score'][background]>threshold
            n=np.histogram(mass,bins)[0]; k=np.histogram(mass[selected],bins)[0]
            if not baseline_drawn:
                axes[0].hist(mass,bins=bins,density=True,histtype='step',color='black',label='All QCD',lw=2)
                baseline_drawn=True
            if selected.any(): axes[0].hist(mass[selected],bins=bins,density=True,histtype='step',label=model)
            good=n>0; rate=k[good]/n[good]
            center=(rate+.5/n[good])/(1+1/n[good]); half=np.sqrt(rate*(1-rate)/n[good]+.25/n[good]**2)/(1+1/n[good])
            errors=np.maximum(0,np.vstack((rate-np.clip(center-half,0,1),np.clip(center+half,0,1)-rate)))
            axes[1].errorbar(centers[good],rate,yerr=errors,fmt='.-',alpha=.7,label=model)
        for ax in axes:
            ax.axvspan(3300,3700,color='gray',alpha=.1); ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[0].set(title=f'Seed {seed}; training SR injection {fraction:.1%}; validation SB 5% cut',ylabel='Normalized density / GeV')
        axes[1].axhline(.05,color='black',ls=':')
        axes[1].set(xlabel='Dijet mass [GeV]',ylabel='QCD selection efficiency')
        fig.tight_layout(); fig.savefig(args.output/f'closure_f{fraction:g}.png',dpi=160); plt.close(fig)
    for fraction in config['fractions']:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for path in sorted(args.output.glob(f'seed{seed}_f{fraction:g}_*/predictions.npz')):
            model=path.parent.name.split(f'_f{fraction:g}_')[1]
            with np.load(path) as d:
                sr=region(d['test_mass'])==1
                truth=d['test_truth'][sr]; score=d['test_score'][sr]
                fpr,tpr,_=roc_curve(truth,score)
                # Omit unstable extreme tails with fewer than 20 expected selected backgrounds.
                good=(fpr*(truth==0).sum()>=20)&(fpr>0)
                axes[0].plot(fpr,tpr,label=model)
                axes[1].plot(fpr[good],tpr[good]/np.sqrt(fpr[good]),label=model)
        axes[0].plot([0,1],[0,1],'k:',label='Random')
        axes[0].set(xlabel='SR background efficiency',ylabel='SR signal efficiency',title='Held-out SR ROC')
        axes[1].axhline(1,color='black',ls=':')
        axes[1].set(xlabel='SR background efficiency',ylabel='SIC',title='Diagnostic curve; not used to choose cuts')
        for ax in axes: ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(f'Predetermined representative seed {seed}; training injection {fraction:.1%}')
        fig.tight_layout();fig.savefig(args.output/f'roc_sic_f{fraction:g}.png',dpi=160);plt.close(fig)

    # Feature information is diagnosed on development events, never on final test.
    feature_file=Path(config['feature_file']) if config.get('feature_file') else None
    extract_args=argparse.Namespace(data=Path(config['data']),feature_file=feature_file,events=config['events'],degree=config['degree'],chunk_size=config['chunk_size'],data_sha256=config.get('source_sha256'))
    data=extract(extract_args)
    train,_,_=partitions(data,seed)
    train=train[data['truth'][train]==0]
    raw=np.column_stack((data['x'],data['jet_mass']))
    names=[f'efp_{i}' for i in range(data['x'].shape[1])]+['lighter_jet_mass','jet_mass_difference']
    with (args.output/'feature_diagnostics.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['feature','background_SR_vs_SB_univariate_AUC','abs_AUC_minus_half'])
        for i,name in enumerate(names):
            auc=roc_auc_score(region(data['mass'][train]),raw[train,i])
            writer.writerow([name,auc,abs(auc-.5)])
    # Additional simulation benchmark: calibrate a fixed SR background efficiency
    # using validation truth only. This is separate from the data-driven SB cut.
    fixed_rows=[]
    for path in args.output.glob('seed*/predictions.npz'):
        run=json.loads((path.parent/'metrics.json').read_text())[0]
        with np.load(path) as d:
            vb=(d['validation_truth']==0)&(region(d['validation_mass'])==1)
            tb=(d['test_truth']==0)&(region(d['test_mass'])==1)
            ts=(d['test_truth']==1)&(region(d['test_mass'])==1)
            for target in (.2,.05):
                threshold=np.quantile(d['validation_score'][vb],1-target)
                selected=d['test_score']>threshold
                eb,es=float(selected[tb].mean()),float(selected[ts].mean())
                fixed_rows.append(dict(seed=run['seed'],fraction=run['fraction'],model=run['model'],
                    target_sr_background_efficiency=target,test_background_efficiency=eb,
                    test_signal_efficiency=es,test_sic=es/np.sqrt(eb) if eb else None))
    with (args.output/'fixed_background_efficiency.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(fixed_rows[0]));writer.writeheader();writer.writerows(fixed_rows)

    # Audit disjointness and fixed test across all completed fits.
    reference=None
    for path in args.output.glob('seed*/predictions.npz'):
        with np.load(path) as d:
            tr,va,te=(set(d[k].tolist()) for k in ['train_rows','validation_rows','test_rows'])
            assert not(tr&va or tr&te or va&te), str(path)
            if reference is None: reference=te
            assert te==reference, str(path)
            assert np.isfinite(d['test_score']).all()
    print(f'Report, diagnostics and split audits completed: {completed}/{expected} fits')

if __name__=='__main__': main()
