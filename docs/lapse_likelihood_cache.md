# Reusable action-only lapse likelihoods

`lapse_likelihood_cache.py` samples each checkpoint's latent planning histories once and stores sufficient statistics for every execution lapse rate in `[0.01,1]`. Observed fixations and durations never enter this calculation. The planning and stopping laws are the same as `path_likelihood.py`; the new cache only avoids repeating planning Monte Carlo when fitting lapse.

```sh
.venv/bin/python lapse_likelihood_cache.py results/runs/EXPERIMENT/RUN \
  --data action_trials.json --output results/diagnostics/lapse_cache_000.json \
  --samples 1024 --batch-size 128 --repeats 2 --trial-start 0 --limit 128 --seed 0
```

The output is an exclusively created JSON manifest and companion `.npz` arrays. `--trial-start` and `--limit` select contiguous input records for independent jobs. The seed is the first four little-endian bytes of SHA-256 applied to `base:checkpoint_identity:global_record_index:repeat`; checkpoint identity concatenates the metadata and parameter file hashes. Each planning draw uses `fold_in(PRNGKey(seed), sample_index)`. Thus trial shards, batch sizes, run lists, and adding repetitions preserve draw identity. Distinct checkpoint identities and repetitions use independent RNG streams. Resume by verifying and skipping completed shard pairs, then submitting only missing shards. A manifest appears only after array writing succeeds; an orphan NPZ from an interrupted write is not a completed shard and must be retained or renamed before retrying that output path.

Each manifest records the complete data SHA-256, problem SHA-256 excluding behavioral fields, metadata and checkpoint hashes, source-file hashes, Git revision, JAX version/backend, seeds, timing, sample count, and latent-planning diagnostics. The array SHA-256 is checked when loading. All provenance refers to the frozen calculation; changes to source, metadata, or checkpoints require a new cache. JAX defaults to CPU unless `JAX_PLATFORMS` is set.

```python
from modules.path_likelihood_cache import load_cache_bundle

manifest, caches = load_cache_bundle("results/diagnostics/lapse_cache_000.json")
cache = caches[0][0][0]  # run, trial, repetition
result = cache.evaluate(0.15)
probabilities = result["probabilities"]
standard_errors = result["standard_errors"]
observed_index = manifest["trials"][0]["observed_index"]
```

The estimator API is `PathLikelihoodEstimator(...).estimate_lapse_cache(problem, policy_params, env_params, samples=..., batch_size=..., seed=...)`. Its `LapseLikelihoodCache` contains `paths`, `path_lengths`, `mean_coefficients`, `coefficient_m2`, `samples`, `seed`, `mean_latent_fixations`, and `horizon_fraction`. `weights(lapse_rate)` returns the Bernstein evaluation weights, and `evaluate(lapse_rate)` returns the entire normalized path distribution and MC standard errors. The manifest's `trials` records participant/trial identifiers, global input indices, paths, and observed indices. `runs` records checkpoint provenance and `estimates[trial][repeat]` array keys and diagnostics.

For a path with depth `d`, conditional execution probability is

```
prod_j ((1-lambda) * greedy_probability_j + lambda/2)
  = sum_k c_k * binom(d,k) * lambda**k * (1-lambda)**(d-k).
```

Each coefficient is nonnegative. A dynamic product computes the `d+1` coefficients without cancellation. Unequal-depth paths retain their own polynomial degrees. The cache stores each coefficient's sample mean and within-path centered cross-product matrix; the quadratic form in Bernstein weights gives the exact empirical MC variance of the probability estimate at any lapse rate. No cross-path covariance is stored. Independent repetitions remain separate for likelihood/ranking diagnostics and independent generation-versus-fitting recovery checks. Candidate checkpoint distributions should be averaged before taking behavioral log probabilities.

Storage per trial/repetition is `8 * paths * (D+1) * (D+2)` bytes for float64 mean and second-moment arrays, before compression and manifest overhead, where `D` is maximum path depth. Eight leaves and depth four require 1,920 bytes; depth seven requires 4,608 bytes. This is independent of the MC sample count. Transient coefficient memory scales with `batch_size * paths * (D+1)`, plus the sampled JAX cognitive states and compilation/runtime overhead. Planning remains the dominant cost. Building the statistics adds `O(samples * paths * D**2)` work for covariance; evaluating a new lapse requires `O(paths * D**2)` for means and standard errors, or `O(paths * D)` for means alone. The implementation retains arrays for one output shard in RAM; choose modest trial shards to bound memory and permit reuse of completed work.

Focused tests compare cached means and SEs against independently recomputed fixed-lapse MC, verify batch invariance and normalization, check unequal-depth degree-seven products, confirm the exact uniform-execution endpoint, and check shard RNG/provenance and corruption detection. The standard cautions about rare unobserved planning events and biased log probability estimates still apply.
