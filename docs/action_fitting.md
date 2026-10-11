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

Run fitting and substantive simulation on Torch. Local verification uses only small synthetic fixtures. Cache filenames in the example are placeholders for actual manifests; the loader verifies each NPZ hash, trial identity, and frozen checkpoint hashes. It streams coefficient arrays, discards the per-sample covariance arrays, and retains only cognitive-candidate mixtures and their independent numerical repetitions.

Each candidate identifies a model family, cognitive parameters, and checkpoint runs. The loader averages independent latent samples within each checkpoint using sample-count weights, then averages checkpoint seed probabilities with equal checkpoint weight. It takes logs only after forming these mixtures. All inventory checkpoints must have every trial cached exactly once, including when supplied in trial shards.

For every candidate, pooled movement lapse is fitted on training trials with equal trial weight, constrained to `[0.01, 1]`. The likelihood profile checks every interior grid maximum and refines it with bounded optimization. The report records both the training maximum within each family and the candidate selected by validation likelihood at its training-fitted lapse. Test data evaluate that frozen validation choice without refitting. The random baseline chooses each legal child with probability one half; its path probability is `2**(-path_length)`.

Reports include:

- Every training fit and lapse likelihood profile; validation scores; selected cognitive parameters; finite-grid nuisance profiles and boundary flags.
- Held-out total and mean log scores, every paired family comparison, participant-cluster bootstrap intervals and the bootstrap fraction favoring each comparison. Equal-participant mean scores provide a weighting sensitivity.
- Held-out full-path expectations for reward in raw units, optimal-path rate, and path length, compared with observed values using participant-cluster uncertainty. Brier scores and top-path calibration accompany log scores.
- Aggregate training, validation, and test log likelihoods under each independent latent-MC repetition at fixed pooled fitted optima, approximate pooled numerical standard errors, and repetition-level model-comparison gaps. This numerical variation is distinct from participant sampling uncertainty and trained-policy-seed variation.

Likelihood-ratio support sets and cognitive boundary flags are finite-grid diagnostics. Their nominal asymptotic cutoffs do not establish continuous-parameter confidence intervals. Recovery measures empirical performance conditional on the supplied trained grid and independent cache repetitions; see `action_recovery.py` for its separate interface.

Reusable fitting functions are in `modules/action_fitting.py`. `Candidate` supports optional subset and observed-path evaluators, so recovery and likelihood optimization can score selected designs without expanding every full path distribution. `modules/action_cache_fitting.py` builds these callbacks from compact Bernstein coefficients.
