# Action-only model and parameter recovery

`action_recovery.py` samples independent complete paths on actual cached task designs and fits every synthetic dataset over a finite cognitive candidate grid and a fine movement-lapse grid. It averages checkpoint seed probabilities before computing log likelihoods, using the same candidate definitions as the empirical fitting pipeline. Generative and fitting latent-planning MC repetitions must be disjoint; the CLI checks both repetition indices and recorded RNG seeds. Synthetic response sampling is inexpensive and reuses cached full path distributions.

Use the primary pooled fitting scale for the main recovery analysis and a smaller diagnostic to show how recovery changes with data quantity. For example, run 100 datasets per true setting at 50 trials, then 30 datasets per true setting at 2,500 actual trial designs. The fixed design subset is recorded, and the same subset is used across true candidates and lapse levels. Choose representative true settings across model families, the fitted choices and their nearest competitors, and cognitive extremes. Every fitted candidate remains in the competition. True lapse levels 0.01, 0.1, 0.3, 0.6, and 1 expose both ordinary recovery and the unidentifiable uniform-response boundary.

```sh
.venv/bin/python action_recovery.py results/diagnostics/cache_shards/*.json \
  --candidate-manifest results/diagnostics/action_fitting_20261011/candidate_manifest.json \
  --truth-ids main_d0p9375_n0p5_c0p001 \
  --trial-counts 50 --repetitions 100 --truth-repeat 1 --fit-repeats 0 \
  --lapse-step 0.005 --output results/diagnostics/recovery_50_fold1.json
```

Choose actual candidate IDs from the frozen manifest. Swap `--truth-repeat` and `--fit-repeats` in a separate run to assess sensitivity to independent likelihood estimation. Use a new output name for every run. The output records source and artifact hashes, seed, actual trial indices, repetition routing, and the lapse grid.

Output includes model confusion counts and rates with Wilson intervals; exact cognitive candidate recovery; within-family normalized cognitive distance and parameter-specific bias, RMSE, and correlation; global and within-family lapse bias and RMSE; empirical coverage and width of within-family profile-likelihood support intervals; and expected score gaps to competing models and cognitive settings. Each synthetic dataset's fitted values are retained. The support interval profiles over all cognitive settings within the true family; its envelope can span disconnected support regions. Its nominal 95% cutoff is asymptotic, and the empirical coverage is the relevant check near boundaries or with discrete nuisance settings.

Exact numerical model ties choose model families uniformly, then tied candidate/lapse cells uniformly within that family. Different grid sizes therefore do not bias exact tie selection. A richer model family can still fit sampling noise more successfully; this is part of the model-recovery result.

Scoring uses a sparse matrix with one response indicator per trial and holds only one candidate's lapse-by-trial-by-path log probabilities at a time. This keeps scoring memory bounded when the undirected stopping grid contains many candidates. Benchmark a short Torch job before increasing the number of datasets. Limit BLAS threads to the allocated CPU count. No additional latent planning simulation is needed to increase synthetic repetitions.

Recovery establishes behavior for the finite trained cognitive grid and the finite lapse/stopping grids supplied to it, conditional on approximate MC-generating distributions. It does not establish continuous cognitive parameter identification or recovery under participant heterogeneity. Independent generating/fitting folds can reveal sensitivity to MC error, but both folds still approximate the model's generative law. The all-lapse boundary erases cognitive and model information and should be interpreted as a negative control.
