# Independent numerical confirmation of frozen action fits

`action_truth_confirmation.py` rescores the validation-selected family finalists using new, independent planning caches. It keeps every cognitive, Gamma-stopping, and movement-lapse parameter fixed and uses the original observed complete paths and participant split.

```sh
.venv/bin/python action_truth_confirmation.py \
  --fit results/diagnostics/action_fitting_20261011/fit.json \
  --candidates results/diagnostics/action_fitting_20261011/candidate_manifest.json \
  --cache results/diagnostics/action_fitting_20261011/confirmation_ordinary_*.json \
  --undirected-cache results/diagnostics/action_fitting_20261011/confirmation_gamma_*.json \
  --baseline-precision results/diagnostics/action_fitting_20261011/mc_precision.json \
  --output results/diagnostics/action_fitting_20261011/truth_confirmation.json
```

The ordinary confirmation caches must cover every checkpoint seed of each selected ordinary candidate and every original trial. The Gamma confirmation caches must cover the selected WM law with the same source metadata hash and stopping horizon; arbitrary fitted Gamma parameters are integrated from those trajectories. All new independent raw repetitions are retained in the numerical check. The helper performs no simulation, optimization, model reselection, or participant resampling.

Original cache manifest paths come from the frozen fit provenance. They must remain accessible. If relocated, supply the complete set with `--baseline-cache`; their manifest hashes must exactly match the original set. The helper verifies data, trial/path ordering, observed choices, problem hashes where present, frozen simulator source hashes, selected checkpoint identities, and new array hashes. The shared recovery RNG auditor compares all original fitting checkpoint/WM streams against the new selected generators on each same `(pid, trial_index)`. It also rejects same-problem seed reuse within the new selected streams. Coincident uint32 seeds across different problems remain outside that audit.

For ordinary finalists, cached within-history coefficient covariance provides analytic delta-log MC standard errors. The helper also records independent-repetition likelihoods as an empirical cross-check, without rereading ordinary arrays. For Gamma, uncertainty comes only from independent raw trajectory repetitions. Those SEs can be noisy with few repeats and are labeled explicitly.

If `--baseline-precision` is provided, ordinary original-cache SEs come from its analytic mean/M2 propagation; the file must refer to the identical frozen fit hash. Otherwise ordinary original-cache SEs use the fit report's repeat-only approximation. For Gamma, the helper rereads only the selected original WM law and rescores the frozen Gamma/lapse choice using all independent raw repetitions. It verifies each pooled split likelihood against the stored fit with absolute tolerance `1e-7` and relative tolerance `1e-10`. In the production four-repeat cache, this provides a repeat-only SE with three degrees of freedom; the pooled score still matches the fit's two groups of two repetitions. Few-repeat SE estimates remain noisy. The confirmation-minus-original SE adds the independent original and new variances. Candidate likelihood-gap SEs likewise add the independent selected-model variances. These calculations condition on the frozen parameters and do not change the original statistical comparison.

The output has `schema_version: 1` and `method: independent_MC_confirmation_of_frozen_selected_parameters`. Its stable fields are:

- `candidates`: each row contains `candidate_id`, `family`, `parameters`, `lapse_rate`, and `splits.train/validation/test`. Each split records `original_total_log_likelihood`, `confirmation_total_log_likelihood`, `confirmation_minus_original`, `confirmation_mc_se`, `baseline_mc_se`, `combined_delta_mc_se`, and `confirmation_repeat_total_log_likelihoods`, alongside the SE methods and analytic/repeat components.
- Gamma split rows additionally retain `baseline_independent_repeat_count`, `baseline_repeat_total_log_likelihoods`, `baseline_repeat_only_mc_se`, `baseline_rescored_total_log_likelihood`, and `baseline_rescored_minus_original` for the original raw-repeat calculation and pooled-score consistency check.
- `paired_family_gaps`: each selected-model pair records `model`, `reference`, and per-split `original_gap`, `confirmation_gap`, `change_in_gap`, `confirmation_gap_mc_se`, and `combined_change_mc_se`.
- `parameters_refitted` and `models_reselected` are false. `independence` retains the full same-problem RNG audit, and `provenance` records all input and analysis-source hashes.

Differences between original and confirmation scores measure numerical integration on the same responses. They are not a second participant sample or evidence that parameter/model selection changed. No score correction or decision rule is applied.
