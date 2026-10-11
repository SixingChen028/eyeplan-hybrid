# Numerical precision from cached planning moments

`action_mc_precision.py` reads completed ordinary Bernstein mean/M2 cache shards and a frozen fit report. It evaluates selected cognitive candidates at their already fitted movement-lapse rates. It streams one checkpoint/trial/repeat moment array at a time, performs no simulation or optimization, and leaves the frozen cache runner unchanged.

```sh
.venv/bin/python action_mc_precision.py \
  --fit results/diagnostics/action_fitting_20261011/fit.json \
  --candidates results/diagnostics/action_fitting_20261011/candidate_manifest.json \
  --cache results/diagnostics/action_fitting_20261011/cache_*.json \
  --output results/diagnostics/action_fitting_20261011/mc_precision.json
```

Pass ordinary cache manifests to `--cache`; Gamma trajectory manifests belong to the separate repeat-based precision calculation already saved in the fit report. By default, the helper analyzes the validation-selected main and unbounded candidates. `--candidate-ids` selects other fitted ordinary candidates. Every selected inventory checkpoint must cover all frozen trials exactly once, including when supplied in trial shards. The helper checks data/checkpoint/array hashes, trial identities, complete participant coverage, and agreement with the frozen training, validation, and selected test likelihoods. It rejects same-trial RNG seed reuse across selected checkpoint/repeat streams because the independent-stream propagation would then omit covariance.

For each ordinary repeat, let `c` be the vector of path Bernstein coefficients, `w` its own-path-degree lapse weights, `M2` the cached centered coefficient cross-product sum, and `n` the independent latent-history sample count. Then

```
p_hat = mean(c) @ w
Var(p_hat) = w.T @ M2 @ w / (n * (n - 1))
```

The helper combines repeat probabilities using sample-count weights. Their variance uses the squared normalized weights. It then averages checkpoint seeds with equal checkpoint weight, dividing the sum of checkpoint probability variances by the squared checkpoint count. This propagates uncertainty in the same arithmetic probability mixture that was logged during fitting. It conditions on the available trained policies and does not treat differences between their behavior as numerical error.

The delta method gives `Var(log(p_hat)) ≈ Var(p_hat) / p_hat**2`. Summing this over independently integrated problems yields the training, validation, and test total-log-likelihood MC variances. For two candidates with independent checkpoint/repeat streams, their likelihood-gap variance is the sum of their total-log-likelihood variances. The report includes these analytic ordinary comparisons, including the exact uniform-movement baseline with zero MC variance.

The leading log-transform bias is `E[log(p_hat)] - log(p) ≈ -Var(p_hat)/(2*p**2)`. Its plug-in estimate, summed bias, and corresponding bias-corrected likelihood/gap are explicitly diagnostic. Fitted parameters, validation choices, and held-out scores remain unchanged. Large relative probability MC errors weaken both the delta and bias approximations. The report therefore includes per-trial probability/MC-SE quantiles, the worst relative-error trials, counts above 0.1 and 0.25 relative error, and the fraction of total log variance contributed by the largest tail. `--omit-trials` omits the complete trial-level table while preserving summaries and the worst-tail records.

Gamma trajectory caches retain independent repeat means rather than within-history covariance. The helper copies their repeat-only split precision from the frozen fit report under `gamma_repeat_only`. Mixed Gamma/ordinary comparisons add the Gamma repeat-estimated variance to the ordinary analytic variance, with the method stated separately. Their Gamma contribution remains noisy with few repeats. The complete original independent-repeat diagnostics are preserved as an empirical cross-check of the analytic ordinary propagation.

All precision estimates condition on the frozen cognitive, stopping, and lapse parameters. They describe numerical integration uncertainty, alongside the separate participant bootstrap; they do not include participant uncertainty, policy-training variability, or parameter-optimization uncertainty. Problem-level integration is treated as independent; coincident uint32 seed hashes across different problems are not treated as duplicate observations.

A small local numerical proof checked covariance contraction against the variance of transformed individual samples, unequal-sample repeat averaging, equal-seed mixtures, streamed split/gap precision, hash rejection, and the sign/size of leading log bias against an exactly enumerated binomial sampling distribution. Its scratch script is `/tmp/eyeplan_action_mc_precision_proof.py`; it does not generate substantive results.
