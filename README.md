# Understanding and Mitigating Client Fragmentation in Federated Domain Generalisation

This is the code and log repository for my project. The project studies what happens to FedPall, a prototype-based federated domain generalisation method, when a single domain's data is split across more than one client -- a situation we call client fragmentation -- and proposes a hierarchical redesign (referred to as RA, the Refined Approach, in the text) to address it.

The repository has two parts:

- `exps/` -- the full code: FedPall's original framework plus everything modified or added for this project.
- `logs/` -- raw per-seed training logs for every result reported in the project, plus the exploratory experiments summarised only briefly in the appendix.

The code has had its comments and docstrings stripped for this submission, so this README is really the only map you get. Read it alongside the write-up, not instead of it.

## What's in `exps/`

Six files carry the actual mechanism:

| File | What it does |
|---|---|
| `federated_main.py` | The training entry point. `ours()` is the main federated training loop; the switches for Layer 1, Layer 2, the domain-indexed discriminator, and StrongLocalAug are all wired in here. |
| `update.py` | Local client update logic (`update_weights_ours_debug`), including the prototype-alignment loss terms. |
| `util.py` | The aggregation functions -- `get_mean_domain_keyed` is Layer 1 (client-to-domain), `get_cross_domain_gpcl` is Layer 2 (domain-to-global). |
| `option.py` | Command-line arguments. |
| `models.py` | Model architecture (ResNet-50 backbone, discriminator, classifiers). |
| `data_utils.py` | Dataset and DataLoader construction for Office-10 and PACS. |

Everything else in `exps/` is a driver script (`run_*.py`) or a diagnostic script (`diag_*.py`, `calib_*.py`, `probe_*.py`, `eval_*.py`, `check_*.py`, `analyze_*.py`). Each `run_*.py` file sets up `args` and calls `ours()` (or occasionally `federated_main.py`'s baseline methods) with one specific combination of settings. A few of the important ones aren't exposed as command-line flags at all -- `lambda_G` (the Layer 2 loss weight), for instance, is a plain Python argument to `ours()`, so the only way to see exactly what a given configuration used is to open the driver script itself.

To reproduce a specific number from the write-up, the fastest route is: find the matching subfolder under `logs/` (they're named after the write-up's RQ sections), then find the driver script whose name matches -- for example, `logs/rq2_RA_office/` was produced by `run_layer2_consumption_office.py`, and `logs/rq2_C_office/` by `run_discriminator_domain_keyed_office.py`. The exploratory work under `logs/appendix_exploratory/` maps the same way to scripts like `run_d_bnaffine_office_dslr.py`, `run_d_featurebank_office_dslr.py`, and so on.

## What's in `logs/`

Each subfolder holds the raw `*_acc.csv` files (one row per communication round per domain, with train loss, KL loss, and accuracy columns where logged) for one configuration, across however many seeds that configuration was run with in the project. The folders are organised by RQ:

- `rq1_ceiling_*`, `rq1_baseline_*` -- the unfragmented reference and the fragmented baseline (RQ1).
- `rq2_discriminator_fix_alone_*`, `rq2_C_*`, `rq2_RA_*`, plus `rq2_flat_gpcl_office` and `rq2_domain_keyed_proto_alone_office` -- the ablation ladder and its two appendix controls (RQ2).
- `rq3_severity_sweep_dslr` -- the M=1/2/4 fragmentation-severity sweep on DSLR (RQ3).
- `rq4_pathway_decomposition_dslr` -- the RA control, the diagnostic expanded-support condition, and Path A/Path B (RQ4).
- `rq5_stronglocalaug_*`, `rq5_baseline_stronglocalaug_office` -- RA_matched, RA+StrongAug, and the baseline+StrongAug attribution control (RQ5).
- `appendix_exploratory/` -- everything mentioned only in passing in Appendix A.9: the BN-pooling and BN-affine diagnostics, FeatureBank, the K-support oracle, the component-interaction matrix, the KL/CE balance intervention, the domain-swap control, and the FiLM-based prototype-composer experiments that were tried and abandoned.

A couple of these folders only have one seed rather than three -- that matches what's reported in the write-up; RQ3 in particular is explicitly a single-seed diagnostic throughout.

## Datasets

Not included here -- Office-10 (Office-Caltech-10) and PACS are standard, publicly available domain generalisation benchmarks, not something specific to this project. `data_utils.py` expects them in the same directory layout FedPall's own code uses; see the FedPall repository below for where to get them.

## Environment

Python 3, PyTorch with torchvision, plus numpy, pandas, matplotlib, seaborn, Pillow, tqdm, and tensorboardX. No specific version pins were recorded; the code was run on both Google Colab and Kaggle notebooks over the course of the project, so it's tolerant of fairly recent versions of all of these.

## Relation to FedPall

This work builds directly on FedPall (Zhang et al., ICCV 2025). The original implementation is at <https://github.com/DistriAI/FedPall>; this project modifies its discriminator indexing and prototype-aggregation logic, and adds the StrongLocalAug intervention, while leaving the rest of the training pipeline as-is.
