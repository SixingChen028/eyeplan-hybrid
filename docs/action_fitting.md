# Action-only fitting

`fit_action_models.py` fits pooled movement lapse and selects among a frozen, finite set of cognitive parameter settings. Inputs contain complete problem graphs, rewards, starts, and root-to-leaf moves. Observed looks, durations, and fixation counts do not enter fitting, candidate selection, predictive scoring, or recovery.

Freeze the participant split before scoring candidates:

```sh
.venv/bin/python fit_action_models.py \
  --data results/diagnostics/action_likelihood_20261011/all_action_trials.json \
  --split results/diagnostics/action_fitting_20261011/participant_split.json \
  --freeze-split-only
```

The default split hash-ranks participant IDs using seed `20261011`, assigning 60/20/20 percent by the largest-remainder rule. All trials from a participant stay together. The complete 288-participant dataset gives 173 training, 58 validation, and 57 test participants. Existing split files must agree exactly with the requested input and method.

Fit from ordinary Bernstein cache shards and an explicit candidate inventory:

```sh
.venv/bin/python fit_action_models.py \
  --data results/diagnostics/action_likelihood_20261011/all_action_trials.json \
  --cache results/diagnostics/action_fitting_20261011/cache_*.json \
  --candidates results/diagnostics/action_fitting_20261011/candidate_manifest.json \
  --split results/diagnostics/action_fitting_20261011/participant_split.json \
  --output results/diagnostics/action_fitting_20261011/fit.json
```

For action-fitted undirected stopping, add `--undirected-cache gamma_cache_*.json`. These caches contain full uniform-fixation trajectories. The loader integrates an initial Gamma shape/mean grid without new simulation, then the default workflow jointly refines Gamma shape, Gamma mean, and movement lapse on training trials for every cached WM law. It compares those WM-specific training optima on validation trials. The report saves the coarse fits, all continuous refinement diagnostics, and the final refined stopping settings for recovery. `--coarse-gamma-only` is an explicit development option.

For the complete dataset, the 1,500-candidate initial Gamma grid with two independent repeats needs about 30 GiB for compact coefficients because trees can have depth seven. Request 48 GB for the initial fitting job. The coarse coefficients are released before continuous refinement; the 50 final refined laws need about 1 GiB. Keep trajectory manifests split by WM law and trial range, so re-reading one law for refinement remains inexpensive.

Run fitting and substantive simulation on Torch. Local verification uses only small synthetic fixtures. Cache filenames in the example are placeholders for actual manifests; the loader verifies each NPZ hash, trial identity, and frozen checkpoint hashes. It streams coefficient arrays, discards the per-sample covariance arrays, and retains only cognitive-candidate mixtures and their independent numerical repetitions.

Each candidate identifies a model family, cognitive parameters, and checkpoint runs. The loader averages independent latent samples within each checkpoint using sample-count weights, then averages checkpoint seed probabilities with equal checkpoint weight. It takes logs only after forming these mixtures. All inventory checkpoints must have every trial cached exactly once, including when supplied in trial shards.

For every candidate, pooled movement lapse is fitted on training trials with equal trial weight, constrained to `[0.01, 1]`. The likelihood profile checks every interior grid maximum and refines it with bounded optimization. The report records both the training maximum within each family and the candidate selected by validation likelihood at its training-fitted lapse. Test data evaluate that frozen validation choice without refitting. The random baseline chooses each legal child with probability one half; its path probability is `2**(-path_length)`.

Reports include:

- Every training fit and lapse likelihood profile; validation scores; selected cognitive parameters; finite-grid nuisance profiles and boundary flags.
- Held-out total and mean log scores, every paired family comparison, participant-cluster bootstrap intervals and the bootstrap fraction favoring each comparison. Equal-participant mean scores provide a weighting sensitivity.
- Held-out full-path expectations for reward in raw units, optimal-path rate, and path length, compared with observed values using participant-cluster uncertainty. Brier scores and top-path calibration accompany log scores.
- Aggregate training, validation, and test log likelihoods under each independent latent-MC repetition at fixed pooled fitted optima, approximate pooled numerical standard errors, and repetition-level model-comparison gaps. This numerical variation is distinct from participant sampling uncertainty and trained-policy-seed variation.

Likelihood-ratio support sets and cognitive boundary flags are finite-grid diagnostics. Their nominal asymptotic cutoffs do not establish continuous-parameter confidence intervals. Recovery measures empirical performance conditional on the supplied trained grid and independent cache repetitions; see `action_recovery.py` for its separate interface.

Continuous Gamma refinement uses `modules/action_gamma_fitting.py` and the training-only optimizer in `modules/undirected_likelihood_cache.py`. The Gamma shape bounds are `[0.1, 100]` and uncapped mean bounds are `[0.1, 1000]`; actual stopping retains the original 100-fixation task cap. Boundary flags, effective capped means, multistart agreement, and local curvature diagnose weak stopping identifiability. The final report's `refined_undirected_candidate_specs` exports each WM law's action-fitted stopping parameters for finite-grid recovery scenarios.

Reusable fitting functions are in `modules/action_fitting.py`. `Candidate` supports optional subset and observed-path evaluators, so recovery and likelihood optimization can score selected designs without expanding every full path distribution. `modules/action_cache_fitting.py` builds these callbacks from compact Bernstein coefficients.
