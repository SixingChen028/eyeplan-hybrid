# Action-only path likelihoods

`path_likelihood.py` estimates `p(moves | problem)` for frozen learned policies or the ordinary undirected Gamma-stopping baseline. It samples independent latent planning histories through their own stopping events, then integrates movement forgetting and greedy tie-breaking analytically. No observed fixation information enters the estimator. Forced-stop outcomes at the planning horizon are included.

Input is a JSON list with zero-indexed `graph`, `start`, `rewards`, and `moves` fields. Optional `pid` and `trial_index` fields are retained in output. `graph` must be a connected full binary tree, the root reward must be zero, and `moves` must be a complete root-to-leaf path excluding the root. Rewards must use the checkpoint's raw point units (normally values from -8 to 8); convert experimental rewards before evaluation. Other record fields are ignored.

```sh
.venv/bin/python path_likelihood.py results/runs/EXPERIMENT/RUN \
  --data action_trials.json --output results/diagnostics/path_likelihood.json \
  --samples 4096 --batch-size 512 --repeats 4 --seed 0
```

Pass multiple run directories to evaluate each separately. Network parameters and cognitive parameters are loaded from run metadata with compatibility checks. The undirected baseline uses the existing `RandomSearchSimulator` stopping/fixation law. Output files are created exclusively, so use a new name for each run. JAX defaults to CPU unless `JAX_PLATFORMS` is already set.

Output contains the entire leaf/path distribution, independent-sample MC standard errors, counts of positive contributions, mean latent voluntary fixation count, and the fraction reaching the horizon. Each repetition uses independent random draws; sample indexing preserves the same draws across batch sizes. File hashes, checkpoint metadata, evaluator Git revision, and timing are recorded. The first call includes compilation; subsequent calls of the same shape give steady-state timing.

Observed-path log probabilities and delta-method log standard errors are diagnostics. The probability estimate is unbiased under independent draws, but its logarithm is generally biased. A zero probability estimate is marked explicitly and has a null log probability; it is not replaced by a floor. A zero MC standard error with zero positive contributions does not establish a structural zero. Evaluate repeated-run stability and rare-path coverage before using these scores for model ranking or optimization.

For a node at depth `d` (root depth zero), its stopping Q survives `d-1` unprotected execution fixations. Its parent then refreshes it before clearing. The survival probability is the product of successively decayed activation strengths. Each branch probability sums the four possible survival outcomes of its child pair. Since execution skips Q backups, and child pairs at different depths have independent retention streams, multiplying branch probabilities gives the exact conditional path probability. This relies on downward-only tree traversal with independent node forgetting and no execution backups.

Existing model calibration may have used looks even though evaluation ignores them. The command does not refit parameters, combine checkpoints into an ensemble, or perform cross-validation; those choices belong to the analysis using its output.
