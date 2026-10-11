"""Training-only continuous Gamma refinement using streamed undirected caches."""

import numpy as np

from modules.action_cache_fitting import undirected_observed_coefficients


def refine_all_gamma(paths, trial_metadata, coarse_fitted, train_indices, *, progress=None):
    """Refine every cached WM law, using coarse training maxima as starts.

    Validation and test responses are never passed to the stopping optimizer.
    Return candidate specifications for re-integrating complete distributions and
    numerical/boundary diagnostics for the continuous stopping fit.
    """
    from modules.undirected_likelihood_cache import refine_gamma_coefficients

    groups = {}
    for fit in coarse_fitted["fits"]:
        if fit["family"] == "undirected":
            law = (fit["parameters"]["wm_decay"], fit["parameters"]["wm_neighbor_activation"])
            groups.setdefault(law, []).append(fit)
    rows = np.asarray(train_indices, dtype=int)
    keys = [(str(trial_metadata[index]["pid"]), int(trial_metadata[index]["trial_index"])) for index in rows]
    specifications, diagnostics = {}, []
    for law in sorted(groups):
        ranked = sorted(groups[law], key=lambda fit: fit["total_log_likelihood"], reverse=True)
        starts = [{**fit["parameters"], "lapse_rate": fit["lapse_rate"]} for fit in ranked[:3]]
        coefficients, lengths = undirected_observed_coefficients(paths, law, keys)
        refined = refine_gamma_coefficients(coefficients, lengths, starts)
        specifications[law] = [{"gamma_shape": refined["gamma_shape"], "gamma_mean": refined["gamma_mean"]}]
        diagnostics.append({
            "wm_decay": law[0], "wm_neighbor_activation": law[1],
            "coarse_best_candidate": ranked[0]["candidate_id"],
            "coarse_best_train_log_likelihood": ranked[0]["total_log_likelihood"],
            "continuous_minus_coarse_train_log_likelihood": refined["train_log_likelihood"]
            - ranked[0]["total_log_likelihood"], **refined,
        })
        if progress is not None:
            progress(len(diagnostics), len(groups), diagnostics[-1])
    return specifications, diagnostics
