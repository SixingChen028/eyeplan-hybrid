# Action-only path likelihoods

`path_likelihood.py` estimates `p(moves | problem)` for frozen learned policies or the ordinary undirected Gamma-stopping baseline. It samples independent latent planning histories through their own stopping events, then integrates movement forgetting, greedy tie-breaking, and movement lapses analytically. No observed fixation information enters the estimator. Forced-stop outcomes at the planning horizon are included.

At each movement step, an independent lapse chooses uniformly between the two legal children. With probability `1 - lapse_rate`, the ordinary memory-dependent greedy rule applies. Thus each conditional branch probability is `(1 - lapse_rate) * p_greedy + lapse_rate / 2`. The same lapse rule applies to all model groups. `--lapse-rate` defaults to 0.01 and must be in `[0.01, 1]`; the Python estimator enforces the same bounds. Lapses affect movement execution only, and do not change planning, stopping, cognitive parameters, or trained checkpoints.

Input is a JSON list with zero-indexed `graph`, `start`, `rewards`, and `moves` fields. Optional `pid` and `trial_index` fields are retained in output. `graph` must be a connected full binary tree, the root reward must be zero, and `moves` must be a complete root-to-leaf path excluding the root. Rewards must use the checkpoint's raw point units (normally values from -8 to 8); convert experimental rewards before evaluation. Other record fields are ignored.

```sh
.venv/bin/python path_likelihood.py results/runs/EXPERIMENT/RUN \
  --data action_trials.json --output results/diagnostics/path_likelihood.json \
  --samples 4096 --batch-size 512 --repeats 4 --seed 0 --lapse-rate 0.01
```

Pass multiple run directories to evaluate each separately. Network parameters and cognitive parameters are loaded from run metadata with compatibility checks. The undirected baseline uses the existing `RandomSearchSimulator` stopping/fixation law. Output files are created exclusively, so use a new name for each run. JAX defaults to CPU unless `JAX_PLATFORMS` is already set.

Output contains the entire leaf/path distribution, independent-sample MC standard errors, counts of positive contributions, mean latent voluntary fixation count, and the fraction reaching the horizon. The lapse rate and rule are recorded alongside checkpoint provenance. Each repetition uses independent random draws; sample indexing preserves the same draws across batch sizes. File hashes, checkpoint metadata, evaluator Git revision, and timing are recorded. The first call includes compilation; subsequent calls of the same shape give steady-state timing.

Observed-path log probabilities and delta-method log standard errors are diagnostics. The probability estimate is unbiased under independent draws, but its logarithm is generally biased. Every legal depth-`d` path has conditional probability at least `(lapse_rate / 2)**d`, including under planning histories where greedy execution cannot take that path. This is a normalized behavioral response model, not a post-hoc probability floor or a mixture with uniform complete paths. A zero estimate, if numerical underflow occurs on a sufficiently deep tree, remains marked explicitly with a null log probability. Positive support does not establish MC accuracy: evaluate independent repetitions and increased budgets before using these scores for model ranking or optimization. Empirical MC standard errors can miss unsampled rare planning events.

For a node at depth `d` (root depth zero), its stopping Q survives `d-1` unprotected execution fixations. Its parent then refreshes it before clearing. The survival probability is the product of successively decayed activation strengths. Each branch probability sums the four possible survival outcomes of its child pair, then mixes in the lapse probability. Since execution skips Q backups, and child pairs at different depths have independent retention streams, multiplying branch probabilities gives the exact conditional path probability. This relies on downward-only tree traversal with independent node forgetting, independent movement lapses, and no execution backups.

Existing model calibration may have used looks even though evaluation ignores them. The command does not refit parameters, combine checkpoints into an ensemble, or perform cross-validation; those choices belong to the analysis using its output.
