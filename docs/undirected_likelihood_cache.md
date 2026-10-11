# Action-only undirected stopping fits

`undirected_likelihood_cache.py` simulates uniform legal fixation trajectories under each candidate working-memory parameter pair. It stores independent repetition means of conditional execution Bernstein coefficients at every possible stopping time. Gamma shape, Gamma scale, and movement lapse can then be fitted using the cache, without further simulation. Observed moves enter scoring after simulation; observed fixations, durations, and fixation counts do not enter either calibration or evaluation.

The ordinary undirected policy permits immediate fixation repetitions. The cache preserves that rule, the stochastic working-memory dynamics, and the cognitive architecture's legal fixation mask. Working-memory decay and neighbor activation require separate trajectory simulations because both influence memory and the available fixations. No trained neural policy is needed for this model.

For `X ~ Gamma(shape, scale)`, let `M = min(t_max, 100)`. The stopping target is `max(min(floor(X), M) - 1, 0)`, counting extra fixations after the initial root fixation. For `M > 1`, target zero has mass `P(X < 2)`, internal target `k` has mass `P(k + 1 <= X < k + 2)`, and target `M - 1` has mass `P(X >= M)`. For `M = 1`, target zero has probability one. These masses include the capped tail exactly. All shape and scale pairs use the same simulated trajectories, so changing the stopping parameters introduces no additional Monte Carlo noise.

```sh
.venv/bin/python undirected_likelihood_cache.py results/runs/EXPERIMENT/UNDIRECTED_RUN \
  --data action_trials.json --output results/diagnostics/undirected_wm08_04.json \
  --samples 256 --repeats 8 --batch-size 256 --seed 0 \
  --wm-decay 0.8 --wm-neighbor-activation 0.4
```

The run must have `lesion_policy = "random_search_gamma_stopping"` and current compatibility metadata. Optional working-memory argument lists form a Cartesian grid; omitting them uses the run's original values. The input schema and reward units match `path_likelihood.py`. The command creates a JSON manifest and a compressed NPZ file exclusively, recording data and array hashes, source metadata, parameter pairs, seeds, and evaluator Git revision. Prefer one working-memory pair and a bounded trial shard per job to bound host memory and permit reuse of completed work.

```python
from modules.undirected_likelihood_cache import load_undirected_cache_bundle

manifest, caches = load_undirected_cache_bundle("results/diagnostics/undirected_wm08_04.json")
result = caches[0][0].evaluate(shape=2, scale=8, lapse_rate=0.1)
independent_repeats = caches[0][0].evaluate_repeats(shape=2, scale=8, lapse_rate=0.1)
```

Each cache stores `[independent_repeat, stopping_target, path, Bernstein_coefficient]` means in float64. The standard error is the standard deviation of independent repetition probabilities divided by the square root of their number. This accounts for dependence across stopping snapshots within a trajectory without storing a large covariance matrix. At least two repetitions are required; four to eight or more are preferable for assessing uncertainty. Fitting, recovery truth generation, and confirmation should use explicitly separated repetition groups or independent cache seeds where independence is required.

A complete 15-node depth-three tree with horizon 100 requires 25,600 bytes per independent repeat and trial. Four repeats therefore need about 102 MB per 1,000 trials and working-memory pair before compression. A batch of 256 trajectories produces about 3.3 MB of float32 coefficient output. Compilation and runtime depend on CPU and problem shape; benchmark representative jobs on the target machine before allocating the full compute budget.

`load_undirected_candidates(manifest_path, candidate_specs=None, repeat_indices=None)` converts a bundle to compact fitting candidates. Its default stopping grid crosses Gamma shapes `0.5, 1, 2, 4, 8` with unbounded Gamma means `2, 4, 8, 16, 32, 64`; scales are mean divided by shape. Supplying `candidate_specs` with `gamma_shape` and `gamma_scale` evaluates any desired stopping parameters. `repeat_indices` selects an explicit independent repetition group. The resulting candidates retain Gamma-integrated Bernstein coefficients and provide efficient full-distribution, trial-subset, and observed-path evaluation.

For shortlisted working-memory candidates, `refine_gamma_coefficients(observed_coefficients, path_lengths, starts)` jointly refines Gamma shape, unbounded Gamma mean, and movement lapse. The coefficients have shape `[stopping_target, training_trial, Bernstein_coefficient]`, retaining only each training trial's observed path and averaging the intended independent repetition group before the call. Each starting dictionary contains `gamma_shape`, `gamma_scale` or `gamma_mean`, and `lapse_rate`. The helper adds coordinate neighbors around the best supplied initialization and fits in log shape and log mean, bounded by `[0.1, 100]` and `[0.1, 1000]`, with lapse bounded by `[0.01, 1]`.

Refinement uses training observations only. Apply the resulting fixed Gamma parameters to validation observations to select among working-memory candidates, then evaluate the selected fixed model on the untouched test observations. The helper reports optimization status, boundary flags, local likelihood curvature, agreement across starts, and the effective mean fixation count after rounding and capping. Curvature diagnostics identify flat or weakly separated fits; they are not calibrated confidence intervals. The unbounded Gamma mean can greatly exceed the maximum fixation count while predicting nearly identical capped stopping behavior, so interpret it together with the effective mean and maximum-target probability.

`load_undirected_trajectory_arrays(manifest_paths, wm_decay, wm_neighbor_activation, trial_keys=..., repeat_indices=...)` streams one working-memory law from hash-verified shards. It returns trial metadata, coefficients shaped `[repeat, stopping_target, trial, path, Bernstein_coefficient]`, and seeds shaped `[repeat, trial]`. Requested participant/trial keys preserve their supplied order. Use distinct seed-checked repetition groups for generating synthetic recovery actions and fitting them. Trial subsets are applied before retaining arrays, bounding memory for repeated recovery studies.
