# Action-fitting result artifacts

`tools/report_action_fits.py` turns completed, frozen fit and recovery reports into a scientific Markdown/JSON summary and PNG/PDF figures. Run it on Torch after fitting and recovery finish; it does not launch simulation, train policies, or change fitted choices.

```sh
.venv/bin/python tools/report_action_fits.py \
  --fit results/diagnostics/action_fitting_20261011/fit.json \
  --recovery results/diagnostics/action_fitting_20261011/recovery.json \
  --protocol-recovery results/diagnostics/action_fitting_20261011/protocol_recovery.json \
  --continuous-recovery results/diagnostics/action_fitting_20261011/continuous_gamma_recovery.json \
  --truth-confirmation results/diagnostics/action_fitting_20261011/truth_confirmation.json \
  --mc-precision results/diagnostics/action_fitting_20261011/mc_precision.json \
  --candidates results/diagnostics/action_fitting_20261011/candidate_manifest.json \
  --accounting results/diagnostics/action_fitting_20261011/sacct.psv \
  --output-dir results/diagnostics/action_fitting_20261011/report
```

The accounting input accepts a pipe-delimited `sacct -P` export with `JobIDRaw,JobName,State,AllocCPUS,AllocTRES,ElapsedRaw,TotalCPU,MaxRSS,ReqMem`. Capture the allocations and their steps. The report sums allocated CPU/GPU resources multiplied by elapsed allocation time, excludes duplicate step accounting, retains failures in usage, and reads peak RSS from steps. Its scope is limited to the supplied job IDs. A JSON accounting summary with `verified`, `cpu_hours`, `gpu_hours`, `scope`, `requested_memory_values`, and `peak_step_rss_gib` is also accepted.

Figure 1 shows held-out mean log scores with participant bootstrap uncertainty and raw participant paired differences. Figure 2 shows scenario-pooled model confusion with Wilson intervals, raw lapse-recovery draws and empirical recovery ranges, and finite-grid cognitive parameter recovery. The largest supplied trial design is used in Figure 2; other designs and individual recovery cells remain in the JSON. Pure-random lapse-one scenarios remain in the source recovery report but are excluded from that representative figure when lower-lapse scenarios exist. Optional Figure 3 separately shows continuous Gamma/lapse recovery conditional on a fixed WM law, including cap-aware expected fixation counts and maximum-target probability.

The summary identifies the pooled homogeneous estimand, inherited look-based checkpoint selection, frozen participant split, finite-grid scope, diagnostic support intervals, Gamma boundary/flatness caveats, independent numerical MC precision, and Slurm usage. Continuous cognitive identifiability, participant-specific parameter recovery, and a newly trained publication-size policy grid are not established by this analysis.

`--protocol-recovery` supplies primary full-design replay of the empirical fitting procedure, plotted separately as Figure 4: train-fitted lapse/stopping, validation-selected cognitive candidates within each family, then frozen test-family comparison including random movement. Its `procedure` metadata labels matched continuous Gamma fitting or a finite-grid approximation; different procedures must be reported in separate output directories. Figure 4 displays binomial uncertainty, every synthetic test-score gap and recovered lapse, and empirical recovery ranges. `--recovery` remains the secondary pooled-likelihood identifiability diagnostic on smaller designs. `--continuous-recovery` remains conditional on a fixed WM law. `--truth-confirmation` preserves independent finalist score-confirmation output and provenance separately from all synthetic recovery. The report does not infer equivalence between these procedures or MC fold precisions.

Pass multiple primary output paths after `--protocol-recovery` to aggregate independently seeded chunks. Aggregation requires the same frozen split, generating parameters, and fitting procedure, and rejects repeated synthetic draw identities. Each source report's seed, precision metadata, and provenance remain separately recorded in the summary.

`--mc-precision` includes the standalone cached-covariance report in full, verifies its frozen fit/split identity, and adds total-score and held-out gap tables. Ordinary policy MCSEs use analytic M2 propagation; Gamma MCSEs use trajectory repeats. Leading log-bias values remain diagnostics at frozen parameters. Participant bootstrap intervals describe sampling uncertainty separately from either numerical MC estimate.

Independent finalist confirmation tables show original and independently rescored train/validation/test totals, their changes and combined numerical MCSEs, plus frozen test-family gap comparisons. The report verifies the confirmation's frozen fit and split hashes and requires unchanged fitted parameters and selected models. The complete confirmation JSON, repeat scores, uncertainty methods, RNG audit, and provenance are retained.

Outputs are written exclusively by default; use a fresh directory for a new result revision. `--overwrite` explicitly permits regenerating artifacts from the same completed source reports. `--markdown` selects another summary destination. Source report and generator hashes are included in `summary.json`.
