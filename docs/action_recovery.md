# Action-only model and parameter recovery

`run_split_action_recovery.py` is the primary recovery analysis. It generates complete synthetic action paths on every actual task design and replays the frozen empirical participant split: training fits parameters for every candidate, validation selects a cognitive setting within each family, and test scores compare those frozen family choices with exact uniform legal-child movement. For the current 13,965-trial design the split contains 8,733 training, 2,459 validation, and 2,773 test trials. Recovery is conditional on a homogeneous generating model across participants.

The generating scenarios are the empirically selected main, unbounded, and undirected choices at their fitted lapse rates. Generative distributions use separately sampled finalist confirmation caches; fitting distributions pool all baseline histories, preserving empirical MC precision. RNG preflight checks every used generating checkpoint against every fitting checkpoint and working-memory law on each actual `(pid, trial_index)`. Same-problem shared seeds are rejected. Coincident uint32 seeds on different problems are unchecked and imply a weak-dependence approximation.

For ordinary learned candidates the primary analysis uses the same 101-point lapse profile and bounded refinement around every local maximum as empirical fitting. For each undirected working-memory law it first fits lapse for the 30 coarse Gamma settings, selects the top three training starts, and applies the same continuous Gamma shape/mean/lapse optimizer and neighboring starts as empirical fitting. It then profiles lapse at the fitted Gamma law, selects working memory on validation, and freezes test predictions. The coarse Gamma grid initializes continuous stopping optimization; the trained cognitive/working-memory grid remains finite.

Pass baseline learned shards with `--cache`, baseline trajectory shards with `--undirected-cache`, independent learned finalist shards with `--truth-cache`, independent trajectory finalist shards with `--truth-undirected-cache`, the frozen inventory with `--candidate-manifest`, empirical results with `--fit-results`, and the frozen split with `--split`. Outputs include hashes, full pooled-repeat precision, independent RNG audit, candidate selections, per-family frozen test scores, and synthetic responses. Existing output files are preserved.

Use five independent chunks with `--repetitions 20` per generating family and distinct `--seed` values to obtain 100 datasets per family. Each chunk reuses the same cached generating distributions and has independent synthetic action draws. Aggregate all five outputs using the report generator's `--protocol-recovery` inputs. Wilson intervals quantify uncertainty from finite synthetic replications conditional on those cached distributions; they do not include latent-MC uncertainty. Start with a benchmark chunk before committing the final allocation.

Primary outputs report family confusion with Wilson intervals, validation-selected cognitive/working-memory exact recovery and normalized distance, parameter bias and RMSE, lapse bias/RMSE, and empirical coverage of conditional training lapse support intervals. Continuous Gamma parameters are summarized by bias/RMSE rather than exact equality. Each dataset retains its generating-family test score minus the best competing family score, including random. The lapse interval holds Gamma and the selected cognitive candidate fixed; it is not a joint cognitive/Gamma confidence region, and empirical coverage includes validation-selection effects.

`--checkpoint-dir` saves completed working-memory-law optimizer results and resumes them only when synthetic responses, split, starts, profiles, cache manifests, and analysis sources match. At the full design, the 1,500-setting compact Gamma coefficient grid occupies about 10.7 GB per pooled copy. Streaming one pooled group temporarily retains two copies; one law's full raw trajectories and mean add about 3.6 GB during optimization. Request approximately 32 GB initially and verify actual memory. A Torch benchmark of one law on the actual 8,733 training trials took about three seconds for the continuous optimizer with six starts; total recovery also includes coarse profiles, scalar refinement, data loading, and frozen scoring. Keep job walltime below four hours when feasible and limit BLAS threads to allocated CPUs.

`action_recovery.py` provides a separate pooled identifiability diagnostic at 50 and 2,500 actual trial designs. It fits every synthetic dataset over the full finite cognitive/Gamma candidate grid and a fine lapse grid. Its 2,500-trial design approximates the validation/test sample size, whereas the primary protocol uses all 13,965 trials. Choose representative truths from fitted choices, nearest competitors, and cognitive extremes, with lapse levels 0.01, 0.1, 0.3, 0.6, and 1. Every fitted competitor remains in the analysis.

Pass `--undirected-cache` to include the complete supplied working-memory/Gamma stopping grid. The CLI selects the union of requested task designs before allocating coefficient arrays. Learned repeat 0 supplies fitting and repeat 1 supplies truth by default. Four undirected repeats are grouped into independent folds `[[0,1],[2,3]]` to match the two learned folds. Swap folds in a separate run to assess MC sensitivity. These split histories have half the empirical pooled latent precision; the primary independent-finalist analysis preserves full precision. The 2,500-trial secondary grid needs about 5.8 GB for pooled plus two fold copies during loading; request approximately 8 GB initially.

```sh
.venv/bin/python action_recovery.py results/diagnostics/cache_shards/*.json \
  --candidate-manifest results/diagnostics/action_fitting_20261011/candidate_manifest.json \
  --truth-ids main_d0p9375_n0p5_c0p001 \
  --trial-counts 50 2500 --repetitions 100 --truth-repeat 1 --fit-repeats 0 \
  --lapse-step 0.005 --output results/diagnostics/recovery_fold1.json
```

Choose actual candidate IDs from the frozen manifest and a new output name for each run. Synthetic seeds are derived separately for each truth/lapse/design cell, so changing the competitor grid preserves that cell's responses. Scoring uses a sparse matrix with one response indicator per trial and holds only one candidate's lapse-by-trial-by-path log probabilities at a time. More synthetic repetitions require no additional latent-planning simulation.

The secondary output includes family confusion, exact candidate recovery, normalized parameter distances, parameter-specific bias/RMSE/correlation, lapse bias/RMSE, empirical coverage and width of within-family profile support, and expected score gaps to competing models/settings. Its interval profiles over all cognitive settings in the true family and can span disconnected support regions. Exact numerical model ties choose families uniformly, then tied candidate/lapse cells uniformly within that family. Lapse 1 erases model/cognitive information and is a negative control; lapse below 1 alone does not establish identifiability.

All analyses are conditional on the finite trained cognitive grid and approximate MC-generating distributions. They do not establish continuous cognitive identification or recovery under participant heterogeneity. Independent truth caches/folds reveal MC sensitivity but still approximate the model's generative law.

`continuous_gamma_recovery.py` adds a separate diagnostic matching the continuous stopping optimizer used for empirical fits. It fixes one working-memory law, generates complete paths from an independent trajectory repetition, selects the resulting observed-path coefficients from the fitting repetition, and jointly optimizes Gamma shape, Gamma mean, and lapse. Three diverse Gamma-grid starts are screened for each dataset; the optimizer adds neighboring starts. The actual sampled path indices are retained for reuse.

```sh
.venv/bin/python continuous_gamma_recovery.py results/diagnostics/undirected_shards/*.json \
  --wm-decay 0.9375 --wm-neighbor-activation 0.5 \
  --truth-gamma 2 8 --truth-gamma 0.5 4 --truth-gamma 8 16 \
  --true-lapses 0.1 0.3 --trial-counts 50 2500 --repetitions 20 \
  --truth-repeat 1 --fit-repeats 0 --output results/diagnostics/continuous_gamma_recovery.json
```

Use the empirically selected working-memory law and its continuously fitted Gamma parameters for the main truth setting, plus contrasting shape/mean settings. Report shape, scale, mean, lapse, effective capped mean fixation count, and maximum-target mass. Output includes bias, RMSE, correlations across truth settings, optimizer success, boundary rates, and local-flatness diagnostics. This is continuous stopping/lapse recovery conditional on one working-memory law; the separate finite-grid analysis assesses joint model and working-memory recovery. Local curvature and multistart agreement are not calibrated parameter confidence intervals.
