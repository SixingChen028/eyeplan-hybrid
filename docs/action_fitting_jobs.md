# Frozen action-likelihood jobs on Torch

`action_fitting_jobs.py` creates a portable work manifest, executes one candidate and contiguous trial shard per CPU job, and validates completed output before reuse. Learned candidates average the selected training seeds downstream; every selected seed is cached separately within the same shard. Undirected candidates cache trajectories at one working-memory pair, reusing them over Gamma stopping parameters and movement lapse rates.

Freeze all likelihood code in Git and deploy that revision before generating a work manifest. Transfer the frozen candidate manifest, action-only trials, participant split, and required run metadata/checkpoints as data. Paths in work manifests are repository-relative; generate and run from the repository root.

```sh
RUN=results/diagnostics/action_fitting_20261011
.venv/bin/python action_fitting_jobs.py generate \
  --candidates "$RUN/candidate_manifest.json" \
  --data results/diagnostics/action_likelihood_20261011/all_action_trials.json \
  --split "$RUN/participant_split.json" --output "$RUN/benchmark_main_jobs.json" \
  --cache-dir "$RUN/benchmark_main_cache" --candidate-id main_d0p9375_n0p375_c0p005 \
  --limit 32 --shard-size 32 --samples 1024 --repeats 2 --batch-size 128 \
  --memory 8G --walltime 00:30:00 --concurrency 1

.venv/bin/python action_fitting_jobs.py run --manifest "$RUN/benchmark_main_jobs.json" --task-index 0
```

The same generator supports `--model main`, `--model unbounded`, or `--model undirected`, repeated `--candidate-id`, `--scope primary|sensitivity|all`, repeated `--training-seed`, and `--seed-count`. Without seed restrictions, all inventoried candidate seeds are included. `--trial-start` and `--limit` select the global input range; `--shard-size` partitions it without gaps or overlap. `--samples` and `--repeats` apply to learned models; `--undirected-samples` and `--undirected-repeats` apply to undirected trajectories. Samples are per checkpoint and repetition for learned models, and trajectories per repetition for undirected models. Independent seeds remain stable across changes to trial partitions. A new base `--seed` creates an independent cache for confirmation or recovery generation.

Each task key hashes the exact Git revision, likelihood source hashes, input hashes including the frozen participant split, checkpoint/metadata hashes, selected runs, cognitive parameters, sample count, repetition count, base seed, and global trial range. Moving data between machines preserves identity because absolute paths are excluded from task keys. Resource requests are stored separately so scheduling can change without changing the scientific calculation. Runtime checks require the frozen Git revision and source/input/checkpoint bytes before launching any calculation.

```sh
mkdir -p "$RUN/log_main"
.venv/bin/python action_fitting_jobs.py slurm --manifest "$RUN/benchmark_main_jobs.json" \
  --output "$RUN/benchmark_main.sbatch" --repo /ABSOLUTE/TORCH/nn-python \
  --log-dir "$RUN/log_main"
bash -n "$RUN/benchmark_main.sbatch"
sbatch --test-only "$RUN/benchmark_main.sbatch"
sbatch "$RUN/benchmark_main.sbatch"
```

The script requests one CPU, forces JAX CPU execution and one numerical-library thread, and allocates a unique temporary directory with cleanup on exit. The default array cap is 32 simultaneous tasks, memory is 8 GB, and wall time is `03:59:00`; generation of scripts with wall time at least four hours is rejected. Benchmark representative learned and undirected candidates on Torch before selecting shard sizes and final memory requests. The benchmark above requests 30 minutes and all available candidate seeds. Slurm log directories must exist before submission. Code deployment uses Git; generated manifests, arrays, receipts, logs, and run metadata are data under ignored `results/` and can be transferred separately.

```sh
.venv/bin/python action_fitting_jobs.py status --manifest "$RUN/benchmark_main_jobs.json"
# Restrict a retry array to missing indices by repeating --task-index:
.venv/bin/python action_fitting_jobs.py slurm --manifest "$RUN/benchmark_main_jobs.json" \
  --output "$RUN/retry_main.sbatch" --repo /ABSOLUTE/TORCH/nn-python \
  --log-dir "$RUN/log_main" --task-index 0
```

Before launching, a task exclusively creates a lock and a request sidecar containing its task key. Successful calculation writes the cache JSON and NPZ exclusively; the runner validates the cache configuration, trial/run counts, source/checkpoint/data identities, and NPZ SHA-256, then exclusively writes a completion receipt with both output hashes, requested resources, Slurm identifiers, elapsed time, and the exact command. Resume skips only a task with matching key and validated output hashes. A fully written cache with a matching request but no receipt can be validated and receipted without recomputing. A stale lock, orphan array, truncated JSON, changed input, or hash mismatch stops that task for inspection; preserve or rename the affected task files before retrying. No output is overwritten and no failed cache is silently accepted.

`status` validates completed output pairs and reports complete, pending, locked, receiptable, and invalid task indices. Its hash audit reads all completed NPZ files, so use it at checkpoints rather than repeatedly polling large result sets. Keep the frozen checkout available while an array runs; new code revisions require a separately generated work manifest.
