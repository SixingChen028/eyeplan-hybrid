"""Summarize completed action fits/recovery and render scientific figures on Torch."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, ScalarFormatter
import numpy as np


COLORS = {"main": "#27658c", "unbounded": "#ad6330", "undirected": "#4c846c", "random": "#777777"}
LABELS = {"main": "Optimal WM", "unbounded": "Unbounded WM", "undirected": "Undirected", "random": "Random"}


def read_json(path):
    return json.loads(Path(path).read_text())


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def wilson(successes, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return [float(center - radius), float(center + radius)]


def memory_gib(value):
    value = value.strip().rstrip("cn")
    if not value or value == "Unknown":
        return None
    units = {"K": 2**-20, "M": 2**-10, "G": 1, "T": 2**10}
    try:
        return float(value[:-1]) * units[value[-1]] if value[-1] in units else float(value) / 2**30
    except ValueError:
        return None


def accounting_summary(path):
    if path is None:
        return {"verified": False, "reason": "Slurm accounting was not supplied"}
    if Path(path).suffix == ".json":
        result = read_json(path)
        return {**result, "input": identity(path)}
    with Path(path).open() as file:
        records = list(csv.DictReader(file, delimiter="|"))
    # Allocation rows have no .batch/.extern suffix. Array-parent summaries are
    # excluded when their individual array allocations are present.
    allocations = [row for row in records if "." not in row["JobIDRaw"]]
    array_parents = {row["JobIDRaw"].split("_")[0] for row in allocations if "_" in row["JobIDRaw"]}
    allocations = [row for row in allocations if row["JobIDRaw"] not in array_parents]
    unique = {row["JobIDRaw"]: row for row in allocations}
    cpu_seconds = gpu_seconds = 0.0
    for row in unique.values():
        elapsed = float(row.get("ElapsedRaw") or 0)
        cpu_seconds += elapsed * float(row.get("AllocCPUS") or 0)
        gpu_counts = {}
        for item in row.get("AllocTRES", "").split(","):
            key, _, value = item.partition("=")
            if key == "gres/gpu" or key.startswith("gres/gpu:"):
                gpu_counts[key] = float(value)
        gpu_count = gpu_counts.get("gres/gpu", sum(gpu_counts.values()))
        gpu_seconds += elapsed * gpu_count
    rss = [memory_gib(row.get("MaxRSS", "")) for row in records]
    rss = [value for value in rss if value is not None]
    return {
        "verified": True, "input": identity(path), "allocation_count": len(unique),
        "cpu_hours": cpu_seconds / 3600, "gpu_hours": gpu_seconds / 3600,
        "hours_definition": "allocated resources multiplied by elapsed job time, including failed allocations",
        "state_counts": dict(Counter(row.get("State", "unknown") for row in unique.values())),
        "peak_step_rss_gib": max(rss) if rss else None,
        "requested_memory_values": sorted({row.get("ReqMem", "") for row in unique.values()}),
        "scope": "Only jobs present in supplied sacct export; historical checkpoint training is excluded unless listed",
    }


def paired_comparison(test, model, reference):
    if model == reference:
        return {"mean": 0.0, "interval": [0.0, 0.0]}
    for row in test["paired_comparisons"]:
        if {row["model"], row["reference"]} == {model, reference}:
            sign = 1 if row["model"] == model else -1
            interval = np.asarray(row["mean_log_score_95_interval"][0]) * sign
            return {"mean": sign * row["mean_log_score"][0], "interval": sorted(interval.tolist())}
    raise ValueError(f"missing held-out comparison: {model} versus {reference}")


def save_figure(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def fit_figure(fit, directory):
    test = fit["test"]
    models = test["models"]
    means = np.asarray(test["scores"]["mean_log_score"])
    intervals = np.asarray(test["scores"]["mean_log_score_95_interval"])
    reference = "main" if "main" in models else next(name for name in models if name != "random")
    compared = [model for model in models if model != reference]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1, 1.4]})
    for index, model in enumerate(models):
        axes[0].errorbar(means[index], index, xerr=[[means[index] - intervals[index, 0]],
                                                 [intervals[index, 1] - means[index]]],
                         fmt="o", color=COLORS.get(model, "#333333"), capsize=3, markersize=7)
    axes[0].set(yticks=np.arange(len(models)), yticklabels=[LABELS.get(model, model) for model in models],
                xlabel="Held-out mean log score (nats / trial)", title="A   Predictive performance")
    rng = np.random.default_rng(20261011)
    for index, model in enumerate(compared):
        values = [(row["total_log_scores"][model] - row["total_log_scores"][reference]) / row["trial_count"]
                  for row in test["participant_scores"]]
        axes[1].scatter(values, index + rng.uniform(-.13, .13, len(values)), color=COLORS.get(model, "#333333"),
                        s=13, alpha=.38, linewidths=0)
        comparison = paired_comparison(test, model, reference)
        mean, interval = comparison["mean"], comparison["interval"]
        axes[1].errorbar(mean, index, xerr=[[mean - interval[0]], [interval[1] - mean]],
                         fmt="D", color="black", capsize=4, markersize=6)
    axes[1].axvline(0, color="#777777", lw=.8, ls="--")
    axes[1].set(yticks=np.arange(len(compared)), yticklabels=[LABELS.get(model, model) for model in compared],
                xlabel=f"Log-score difference versus {LABELS.get(reference, reference)} (nats / trial)",
                title="B   Paired participant differences")
    for axis in axes:
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=.16)
    figure.suptitle(f"Held-out participants: {test['scores']['participant_count']}; "
                    f"trials: {test['scores']['trial_count']}", fontsize=12)
    figure.text(.02, -.02, "Bars: participant-cluster bootstrap 95% intervals. Small dots: participant means. "
                "Black diamonds: pooled trial-weighted differences.", fontsize=9)
    figure.tight_layout()
    save_figure(figure, directory, "figure1_heldout")


def recovery_figure(recoveries, directory):
    datasets = [row for report in recoveries for row in report["datasets"]]
    if not datasets:
        return None
    designs = sorted({row["design"] for row in datasets},
                     key=lambda design: max(row["trials"] for row in datasets if row["design"] == design))
    design = designs[-1]
    selected = [row for row in datasets if row["design"] == design and row["true_lapse"] < 1]
    if not selected:
        selected = [row for row in datasets if row["design"] == design]
    families = sorted({row["true_family"] for row in selected})
    fitted_families = sorted({row["fitted_family"] for row in selected} | set(families))
    matrix = np.zeros((len(families), len(fitted_families)), dtype=int)
    for row in selected:
        matrix[families.index(row["true_family"]), fitted_families.index(row["fitted_family"])] += 1
    totals = matrix.sum(axis=1)
    fractions = matrix / totals[:, None]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    axes[0].imshow(fractions, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(len(families)):
        for j in range(len(fitted_families)):
            lower, upper = wilson(matrix[i, j], totals[i])
            axes[0].text(j, i, f"{fractions[i, j]:.2f}\n[{lower:.2f}, {upper:.2f}]", ha="center", va="center",
                         fontsize=8, color="white" if fractions[i, j] > .58 else "black")
    axes[0].set(xticks=np.arange(len(fitted_families)),
                xticklabels=[LABELS.get(family, family) for family in fitted_families],
                yticks=np.arange(len(families)), yticklabels=[LABELS.get(family, family) for family in families],
                xlabel="Recovered family", ylabel="Generating family", title="A   Model recovery")
    axes[0].tick_params(axis="x", labelrotation=30)
    rng = np.random.default_rng(20261011)
    cognitive_key = next((key for key in ("wm_decay", "wm_neighbor_activation", "cost", "gamma_mean")
                          if len({row["true_parameters"].get(key) for row in selected
                                  if key in row["true_parameters"]}) > 1), None)
    cognitive_values = [row["true_parameters"][cognitive_key] for row in selected
                        if cognitive_key and cognitive_key in row["true_parameters"]]
    cognitive_span = max(cognitive_values) - min(cognitive_values) if cognitive_values else 1
    for index, family in enumerate(families):
        rows = [row for row in selected if row["true_family"] == family]
        sample = [rows[i] for i in rng.choice(len(rows), min(300, len(rows)), replace=False)]
        color = COLORS.get(family, "#333333")
        axes[1].scatter([row["true_lapse"] + rng.uniform(-.006, .006) for row in sample],
                        [row["within_family_lapse"] for row in sample], s=7, alpha=.15, color=color, linewidths=0)
        for lapse in sorted({row["true_lapse"] for row in rows}):
            values = [row["within_family_lapse"] for row in rows if row["true_lapse"] == lapse]
            lower, median, upper = np.quantile(values, [.025, .5, .975])
            offset = (index - (len(families) - 1) / 2) * .012
            axes[1].errorbar(lapse + offset, median, yerr=[[median - lower], [upper - median]], color=color,
                             fmt="o", capsize=2, markersize=4)
        if cognitive_key:
            sample = [row for row in sample if cognitive_key in row["true_parameters"]
                      and cognitive_key in row["within_family_parameters"]]
            axes[2].scatter([row["true_parameters"][cognitive_key] + rng.uniform(-.008, .008) * cognitive_span
                             for row in sample],
                            [row["within_family_parameters"][cognitive_key] + rng.uniform(-.008, .008) * cognitive_span
                             for row in sample], s=12, alpha=.19,
                            color=color, label=LABELS.get(family, family), linewidths=0)
            values = sorted({row["true_parameters"][cognitive_key] for row in rows
                             if cognitive_key in row["true_parameters"]})
            for value in values:
                recovered = [row["within_family_parameters"][cognitive_key] for row in rows
                             if row["true_parameters"].get(cognitive_key) == value]
                lower, median, upper = np.quantile(recovered, [.025, .5, .975])
                offset = (index - (len(families) - 1) / 2) * .02 * cognitive_span
                axes[2].errorbar(value + offset, median, yerr=[[median - lower], [upper - median]], color=color,
                                 fmt="o", capsize=2, markersize=4)
    axes[1].plot([0, 1], [0, 1], color="#555555", lw=.8, ls="--")
    axes[1].set(xlabel="Generating movement lapse", ylabel="Recovered lapse within true family",
                xlim=(0, 1.02), ylim=(0, 1.02), title="B   Lapse recovery")
    if cognitive_key:
        values = [row["true_parameters"][cognitive_key] for row in selected if cognitive_key in row["true_parameters"]]
        limits = [min(values), max(values)]
        axes[2].plot(limits, limits, color="#555555", lw=.8, ls="--")
        axes[2].set(xlabel=f"Generating {cognitive_key}", ylabel=f"Recovered {cognitive_key}",
                    title="C   Finite-grid cognitive recovery")
        axes[2].legend(fontsize=8, loc="best")
    else:
        axes[2].text(.5, .5, "No varying numeric\ncognitive parameter", ha="center", va="center")
        axes[2].set_axis_off()
    for axis in axes[1:]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15)
    lapses = sorted({row["true_lapse"] for row in selected})
    figure.suptitle(f"Secondary pooled recovery: {design}; {len(selected):,} synthetic datasets; "
                    f"generating lapses {lapses}", fontsize=11)
    figure.text(.02, -.03, "A: scenario-pooled proportions and Wilson 95% intervals. B: medians and empirical 95% "
                "recovery ranges. B/C: at most 300 jittered raw draws per family. "
                "Independent fitting/generating MC.", fontsize=8)
    figure.tight_layout()
    save_figure(figure, directory, "figure2_recovery")
    return {"design": design, "datasets": len(selected), "generating_lapses": lapses, "true_families": families,
            "fitted_families": fitted_families, "counts": matrix.tolist(), "cognitive_scatter_parameter": cognitive_key}


def scientific_summary(fit, recoveries, accounting):
    fits = {row["candidate_id"]: row for row in fit["fits"]["fits"]}
    selected = {family: fits[candidate_id] for family, candidate_id in fit["fits"]["selected_by_validation"].items()}
    test = fit["test"]
    rankings = sorted(zip(test["models"], test["scores"]["mean_log_score"]), key=lambda pair: pair[1], reverse=True)
    return {
        "method": fit["method"], "estimand": "pooled homogeneous model; equal trial weight",
        "split": {key: fit["split"][key] for key in ("participant_counts", "trial_counts", "split_sha256")},
        "selected": selected, "heldout": test, "ranking": [{"family": family, "mean_log_score": score}
                                                             for family, score in rankings],
        "cognitive_grid": fit["grid_diagnostics"], "numerical_precision": fit["numerical_precision"],
        "predictive_checks": fit.get("predictive_checks"),
        "gamma_refinement": fit.get("gamma_refinement", fit["fits"].get("gamma_refinement")),
        "secondary_pooled_recovery": [{key: report.get(key) for key in
                     ("method", "scope", "repetitions", "true_lapses", "trial_sets",
                     "cells", "parameter_summary", "identifiability", "provenance")} for report in recoveries],
        "compute": accounting,
        "limitations": [
            "Cognitive conclusions and recovery are conditional on the finite trained checkpoint grid.",
            "Previously trained checkpoint settings were selected using earlier look-based analyses; this split is an "
            "internal action-prediction holdout conditional on that inherited candidate inventory.",
            "The fitted and recovered model is pooled and homogeneous, "
            "not a participant-specific or hierarchical model.",
            "Likelihood support cutoffs are diagnostic; boundaries, finite nuisance grids "
            "and MC approximation affect coverage.",
            "Gamma nuisance fitting may refine the finite coarse grid continuously. Recovery on the coarse grid "
            "does not establish recovery of the continuous Gamma optimizer; inspect boundary/flatness diagnostics.",
            "Participant bootstrap intervals condition on frozen selected models and quantify participant sampling; "
            "numerical MC repeat spreads and synthetic recovery replication are separate uncertainties.",
        ],
    }


def sensitivity_comparison(primary, sensitivity):
    """Compare frozen main fits using paired participant score sufficient statistics."""
    if sensitivity["scope"] != "sensitivity" or set(sensitivity["fits"]["selected_by_validation"]) != {"main"}:
        raise ValueError("sensitivity input must contain the separately fitted main sensitivity candidate")
    if len(sensitivity["fits"]["fits"]) != 1:
        raise ValueError("matched-main sensitivity input must contain one prespecified candidate")
    if primary["split"] != sensitivity["split"]:
        raise ValueError("primary and sensitivity fits require the same frozen participant split")
    for key in ("data_sha256", "candidate_manifest_sha256"):
        if primary["provenance"][key] != sensitivity["provenance"][key]:
            raise ValueError("primary and sensitivity fits require the same data and candidate inventory")

    def selected_main(report):
        candidate_id = report["fits"]["selected_by_validation"]["main"]
        return next(row for row in report["fits"]["fits"] if row["candidate_id"] == candidate_id)

    main, matched = selected_main(primary), selected_main(sensitivity)
    participants = []
    originals = {str(row["pid"]): row for row in primary["test"]["participant_scores"]}
    new = {str(row["pid"]): row for row in sensitivity["test"]["participant_scores"]}
    if (not originals or originals.keys() != new.keys()
            or len(originals) != len(primary["test"]["participant_scores"])
            or len(new) != len(sensitivity["test"]["participant_scores"])):
        raise ValueError("primary and sensitivity test participant identities differ or contain duplicates")
    for pid in sorted(originals):
        left, right = originals[pid], new[pid]
        if left["trial_count"] != right["trial_count"] or left["trial_count"] < 1:
            raise ValueError("primary and sensitivity test participant trial counts differ")
        participants.append({"pid": pid, "trial_count": left["trial_count"],
                             "primary_total_log_score": left["total_log_scores"]["main"],
                             "sensitivity_total_log_score": right["total_log_scores"]["main"],
                             "difference_total_log_score": right["total_log_scores"]["main"]
                             - left["total_log_scores"]["main"]})
    counts = np.asarray([row["trial_count"] for row in participants])
    differences = np.asarray([row["difference_total_log_score"] for row in participants])
    for report, field in ((primary, "primary_total_log_score"), (sensitivity, "sensitivity_total_log_score")):
        if (counts.sum() != report["split"]["trial_counts"]["test"]
                or not np.isclose(sum(row[field] for row in participants),
                                  report["test"]["total_log_likelihoods"]["main"], atol=1e-6, rtol=1e-10)):
            raise ValueError("stored participant scores do not reproduce the frozen test score/count")
    bootstrap = primary["test"]["scores"]
    repetitions, seed = bootstrap["bootstrap_repetitions"], bootstrap["bootstrap_seed"]
    draws = np.random.default_rng(seed).integers(0, len(counts), size=(repetitions, len(counts)))
    means = differences[draws].sum(axis=1) / counts[draws].sum(axis=1)
    totals = {}
    for split, field in (("train", "total_log_likelihood"), ("validation", "validation_log_likelihood")):
        totals[split] = {"primary": main[field], "sensitivity": matched[field],
                         "sensitivity_minus_primary": matched[field] - main[field]}
    left, right = (report["test"]["total_log_likelihoods"]["main"] for report in (primary, sensitivity))
    totals["test"] = {"primary": left, "sensitivity": right, "sensitivity_minus_primary": right - left}
    return {
        "method": "paired_participant_bootstrap_of_frozen_main_score_differences",
        "primary_candidate": main, "sensitivity_candidate": matched, "total_log_likelihoods": totals,
        "test_mean_log_score_difference": float(differences.sum() / counts.sum()),
        "test_mean_log_score_difference_95_interval": np.quantile(means, [.025, .975]).tolist(),
        "bootstrap_probability_above_zero": float((means > 0).mean()),
        "equal_participant_mean_log_score_difference": float((differences / counts).mean()),
        "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
        "participant_count": len(counts), "trial_count": int(counts.sum()), "participant_scores": participants,
        "scope": "Prespecified matched-main sensitivity, reported separately and interpreted as exploratory "
                 "conditional on the inherited checkpoints and frozen split. No primary model is reselected.",
        "uncertainty": "Participant sampling conditional on both frozen fits; excludes MC integration, "
                       "checkpoint-training and parameter-selection uncertainty.",
    }


def protocol_recovery_figure(reports, directory):
    """Plot full train/validation/test procedure recovery separately from pooled ML."""
    rows = [row for report in reports for row in report["datasets"]]
    if not rows:
        return None
    procedures = sorted({report["procedure"] for report in reports})
    if len(procedures) != 1:
        raise ValueError("report each exact/approximate fitting protocol in its own artifact directory")
    split_hashes = {report["split"]["split_sha256"] for report in reports}
    if len(split_hashes) != 1:
        raise ValueError("protocol aggregation requires the same frozen participant split")
    generating_settings = []
    draws = set()
    for report in reports:
        settings = {}
        for row in report["datasets"]:
            key = (report["seed"], row["true_candidate"], row["true_lapse"], row["repetition"])
            if key in draws:
                raise ValueError("protocol aggregation contains duplicate synthetic draws")
            draws.add(key)
            settings[row["true_candidate"]] = (row["true_family"], row["true_lapse"], row["true_parameters"])
        generating_settings.append(settings)
    if any(settings != generating_settings[0] for settings in generating_settings[1:]):
        raise ValueError("protocol aggregation requires the same generating families and fitted parameters")
    gamma_methods = {report["gamma_recovery"] for report in reports}
    if len(gamma_methods) != 1:
        raise ValueError("protocol aggregation requires the same Gamma recovery procedure")
    gamma_method = gamma_methods.pop()
    procedure_label = {
        "continuous_at_every_working_memory_law": "Matched continuous Gamma and lapse fitting",
        "finite_gamma_initialization_grid_approximation": "Finite Gamma approximation",
    }.get(gamma_method, gamma_method.replace("_", " "))
    families = sorted({row["true_family"] for row in rows})
    fitted_families = sorted({row["fitted_family"] for row in rows} | set(families))
    matrix = np.zeros((len(families), len(fitted_families)), dtype=int)
    for row in rows:
        matrix[families.index(row["true_family"]), fitted_families.index(row["fitted_family"])] += 1
    totals = matrix.sum(axis=1)
    fractions = matrix / totals[:, None]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    axes[0].imshow(fractions, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(len(families)):
        for j in range(len(fitted_families)):
            lower, upper = wilson(matrix[i, j], totals[i])
            axes[0].text(j, i, f"{fractions[i, j]:.2f}\n[{lower:.2f}, {upper:.2f}]", ha="center", va="center",
                         fontsize=8, color="white" if fractions[i, j] > .58 else "black")
    axes[0].set(xticks=np.arange(len(fitted_families)),
                xticklabels=[LABELS.get(family, family) for family in fitted_families],
                yticks=np.arange(len(families)), yticklabels=[LABELS.get(family, family) for family in families],
                xlabel="Family with best frozen test score", ylabel="Generating family", title="A   Protocol recovery")
    axes[0].tick_params(axis="x", labelrotation=35)
    rng = np.random.default_rng(20261011)
    test_counts = {report["split"]["trial_counts"]["test"] for report in reports}
    if len(test_counts) != 1:
        raise ValueError("protocol figure requires a shared frozen test design")
    test_count = test_counts.pop()
    for index, family in enumerate(families):
        selected = [row for row in rows if row["true_family"] == family]
        color = COLORS.get(family, "#333333")
        gaps = np.asarray([row["test_gap_to_best_competitor"] / test_count for row in selected])
        lower, median, upper = np.quantile(gaps, [.025, .5, .975])
        axes[1].scatter(gaps, index + rng.uniform(-.11, .11, len(gaps)), s=18, color=color, alpha=.4, linewidths=0)
        axes[1].errorbar(median, index, xerr=[[median - lower], [upper - median]], fmt="D", color=color,
                         capsize=3, markersize=5)
        recovered = np.asarray([row["within_family_lapse"] for row in selected])
        lower, median, upper = np.quantile(recovered, [.025, .5, .975])
        axes[2].scatter(index + rng.uniform(-.11, .11, len(recovered)), recovered, s=18, color=color,
                        alpha=.4, linewidths=0)
        axes[2].errorbar(index, median, yerr=[[median - lower], [upper - median]], fmt="D", color=color,
                         capsize=3, markersize=5)
        truths = sorted({row["true_lapse"] for row in selected})
        for truth in truths:
            axes[2].plot([index - .26, index + .26], [truth, truth], color="black", lw=.9, ls="--")
    axes[1].axvline(0, color="#666666", lw=.8, ls="--")
    axes[1].set(yticks=np.arange(len(families)), yticklabels=[LABELS.get(family, family) for family in families],
                xlabel="Generating-family test advantage (nats / trial)", title="B   Against best competitor")
    axes[1].invert_yaxis()
    axes[2].set(xticks=np.arange(len(families)), xticklabels=[LABELS.get(family, family) for family in families],
                ylabel="Validation-selected within-family lapse", title="C   Parameter recovery")
    axes[2].tick_params(axis="x", labelrotation=30)
    for axis in axes[1:]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15)
    figure.suptitle(f"Primary full-design recovery: {procedure_label}; {len(rows)} synthetic datasets", fontsize=12)
    figure.text(.02, -.02, "A: Wilson 95% intervals. B/C: all raw fits, medians and empirical 95% recovery ranges. "
                "C dashed lines: generating lapse. Test family comparisons include exact random movement.", fontsize=8)
    figure.tight_layout()
    save_figure(figure, directory, "figure4_protocol_recovery")
    return {"procedure": procedures[0], "procedure_label": procedure_label,
            "split_sha256": split_hashes.pop(), "datasets": len(rows), "test_trials": test_count,
            "true_families": families, "fitted_families": fitted_families, "counts": matrix.tolist()}


def continuous_recovery_figure(reports, directory):
    rows = [row for report in reports for row in report["datasets"]]
    if not rows:
        return None
    design = max({row["design"] for row in rows},
                 key=lambda value: max(row["trials"] for row in rows if row["design"] == value))
    rows = [row for row in rows if row["design"] == design]
    metrics = ("gamma_shape", "gamma_mean", "gamma_scale", "lapse_rate", "mean_total_fixations",
               "maximum_target_fraction")
    labels = ("Gamma shape", "Gamma mean before capping", "Gamma scale", "Movement lapse",
              "Expected total fixations after capping", "Maximum-target probability")
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    rng = np.random.default_rng(20261011)
    sample = [rows[index] for index in rng.choice(len(rows), min(500, len(rows)), replace=False)]
    for axis, metric, label in zip(axes.ravel(), metrics, labels):
        truth = np.asarray([row["truth"][metric] for row in sample])
        fitted = np.asarray([row["fit"][metric] for row in sample])
        axis.scatter(truth, fitted, s=13, color=COLORS["undirected"], alpha=.22, linewidths=0)
        for value in sorted({row["truth"][metric] for row in rows}):
            recovered = [row["fit"][metric] for row in rows if row["truth"][metric] == value]
            lower, median, upper = np.quantile(recovered, [.025, .5, .975])
            axis.errorbar(value, median, yerr=[[median - lower], [upper - median]], fmt="o",
                          color="#223b30", capsize=3, markersize=4)
        minimum, maximum = min(truth.min(), fitted.min()), max(truth.max(), fitted.max())
        axis.plot([minimum, maximum], [minimum, maximum], color="#555555", ls="--", lw=.8)
        if metric.startswith("gamma_"):
            axis.set_xscale("log")
            axis.set_yscale("log")
            values = np.unique(truth)
            axis.xaxis.set_major_locator(FixedLocator(values) if len(values) < 7 else LogLocator(numticks=5))
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=6))
            axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set(xlabel=f"Generating {label.lower()}", ylabel="Recovered value", title=label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15)
    figure.suptitle(f"Continuous Gamma recovery conditional on fixed WM: {design}; "
                    f"{len(rows):,} synthetic datasets", fontsize=12)
    figure.text(.02, -.01, "Small dots: at most 500 raw synthetic fits. Large dots/bars: median and empirical 95% "
                "recovery range. Gamma axes logarithmic. Dashed lines: identity.", fontsize=9)
    figure.tight_layout()
    save_figure(figure, directory, "figure3_continuous_gamma")
    return {"design": design, "datasets": len(rows), "parameters": list(metrics)}


def cache_scope(fit):
    methods, checkpoints, arrays, samples, repeats = Counter(), set(), {}, [], []
    for entry in fit.get("provenance", {}).get("cache_manifests", []):
        path = Path(entry["path"])
        if not path.is_file():
            continue
        manifest = read_json(path)
        methods[manifest["method"]] += 1
        samples.append(manifest.get("samples", manifest.get("samples_per_repeat")))
        repeats.append(manifest.get("repeats"))
        for run in manifest["runs"]:
            checkpoints.add(run["run_dir"])
        array_path = path.parent / manifest["arrays_file"]
        if array_path.is_file():
            arrays[str(array_path.resolve())] = array_path.stat().st_size
    return {"manifest_counts_by_method": dict(methods), "unique_run_directories": len(checkpoints),
            "latent_samples_per_repeat_values": sorted({value for value in samples if value is not None}),
            "independent_repeat_count_values": sorted({value for value in repeats if value is not None}),
            "compressed_array_files": len(arrays), "compressed_array_gib": sum(arrays.values()) / 2**30,
            "scope": "Readable cache paths when reporting; run directories include learned and undirected laws"}


def mc_precision_markdown(report):
    if report is None:
        return []
    lines = ["", "Ordinary-policy MCSEs propagate cached within-history coefficient covariance (M2) through "
             "the frozen probability mixtures. Gamma MCSEs use independent trajectory-repeat differences; "
             "their precision is limited by the number of repeats. All scores below are total nats, not nats/trial.",
             "", "| Family | Split | Frozen log likelihood | Numerical MCSE | Leading log-bias diagnostic |",
             "| --- | --- | ---: | ---: | ---: |"]
    for candidate in report["candidates"].values():
        for split, row in candidate["splits"].items():
            lines.append(f"| {LABELS.get(candidate['family'], candidate['family'])} | {split} "
                         f"| {row['total_log_likelihood']:.3f} | {row['delta_total_log_likelihood_mc_se']:.3f} "
                         f"| {row['estimated_total_log_bias']:.3f} |")
    gamma = report.get("gamma_repeat_only")
    if gamma and gamma.get("splits"):
        for split, row in gamma["splits"].items():
            mcse = row.get("approx_pooled_log_likelihood_mc_se")
            value = f"{mcse:.3f}" if mcse is not None else "unavailable"
            lines.append(f"| Undirected (repeat only) | {split} | {row['pooled_total_log_likelihood']:.3f} "
                         f"| {value} | unavailable |")
    lines += ["", "Leading log-bias values are delta-method diagnostics at fixed parameters. "
              "They do not replace the fitted scores and exclude selection and optimizer effects.",
              "", "| Frozen test comparison | Log-likelihood gap | Numerical gap MCSE | Method |",
              "| --- | ---: | ---: | --- |"]
    for comparison in report["paired_ordinary_comparisons"]:
        row = comparison["splits"]["test"]
        lines.append(f"| `{comparison['model']}` − `{comparison['reference']}` "
                     f"| {row['total_log_likelihood_gap']:.3f} | {row['delta_gap_mc_se']:.3f} | Analytic M2 |")
    for comparison in report.get("mixed_gamma_comparisons", []):
        row = comparison["splits"]["test"]
        mcse = row.get("mixed_gap_mc_se")
        value = f"{mcse:.3f}" if mcse is not None else "unavailable"
        lines.append(f"| `{comparison['model']}` − `{comparison['reference']}` "
                     f"| {row['total_log_likelihood_gap']:.3f} | {value} | Ordinary M2 + Gamma repeats |")
    return lines


def confirmation_markdown(reports):
    if not reports:
        return []
    lines = ["", "## Independent finalist confirmation", "",
             "New independent histories rescore the same responses at the frozen selected parameters. "
             "No parameters are refitted or models reselected. Total scores and numerical errors are in nats; "
             "the combined delta MCSE includes original and new integration uncertainty. "
             "Ordinary MCSEs use analytic M2 propagation; Gamma MCSEs use independent trajectory repeats."]
    for index, report in enumerate(reports, start=1):
        lines += ["", f"Confirmation input {index}: same-problem RNG audit "
                  f"`{report['independence']['independent']}`.", "",
                  "| Family | Split | Original score | Independent score | Change | Combined delta MCSE |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for candidate in report["candidates"]:
            for split, row in candidate["splits"].items():
                mcse = row.get("combined_delta_mc_se")
                value = f"{mcse:.3f}" if mcse is not None else "unavailable"
                lines.append(f"| {LABELS.get(candidate['family'], candidate['family'])} | {split} "
                             f"| {row['original_total_log_likelihood']:.3f} "
                             f"| {row['confirmation_total_log_likelihood']:.3f} "
                             f"| {row['confirmation_minus_original']:.3f} | {value} |")
        lines += ["", "| Frozen test comparison | Original gap | Independent gap | Change | Combined change MCSE |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for comparison in report["paired_family_gaps"]:
            row = comparison["splits"]["test"]
            mcse = row.get("combined_change_mc_se")
            value = f"{mcse:.3f}" if mcse is not None else "unavailable"
            model, reference = comparison["model"], comparison["reference"]
            lines.append(f"| {LABELS.get(model, model)} − {LABELS.get(reference, reference)} "
                         f"| {row['original_gap']:.3f} | {row['confirmation_gap']:.3f} "
                         f"| {row['change_in_gap']:.3f} | {value} |")
    lines += ["", "These comparisons assess integration stability separately from participant sampling and "
              "synthetic recovery. Gamma errors can be noisy with few repeats. Full repeat scores, uncertainty "
              "methods, cache identities, RNG checks, and source hashes remain in `summary.json`."]
    return lines


def sensitivity_markdown(report):
    if report is None:
        return []
    comparison = report["comparison"]
    candidate = comparison["sensitivity_candidate"]
    interval = comparison["test_mean_log_score_difference_95_interval"]
    lines = ["", "## Matched-main sensitivity", "", comparison["scope"], "",
             f"Sensitivity candidate: `{candidate['candidate_id']}`, parameters "
             f"`{json.dumps(candidate['parameters'], sort_keys=True)}`, train-fitted lapse "
             f"{candidate['lapse_rate']:.4f}.", "",
             "| Split | Selected primary main total score | Sensitivity total score | Sensitivity − primary |",
             "| --- | ---: | ---: | ---: |"]
    for split, row in comparison["total_log_likelihoods"].items():
        lines.append(f"| {split} | {row['primary']:.3f} | {row['sensitivity']:.3f} "
                     f"| {row['sensitivity_minus_primary']:.3f} |")
    lines += ["", f"Paired test mean difference: {comparison['test_mean_log_score_difference']:.5f} nats/trial "
              f"(participant bootstrap 95% interval [{interval[0]:.5f}, {interval[1]:.5f}]; "
              f"{comparison['participant_count']} participants, {comparison['trial_count']} trials, "
              f"{comparison['bootstrap_repetitions']} resamples). "
              f"Equal-participant mean difference: "
              f"{comparison['equal_participant_mean_log_score_difference']:.5f} nats/trial.",
              comparison["uncertainty"], "Raw paired participant scores, separate sensitivity-fit diagnostics, "
              "and both source identities remain in `summary.json`."]
    return lines


def markdown_report(summary, recovery_panel, figure_prefix=""):
    split, test = summary["split"], summary["heldout"]
    lines = ["# Action-only model-fitting results", "",
             "Completed action fits and independent-cache recovery are summarized below.",
             "", "## Held-out prediction", "",
             "| Family | Mean log score, nats/trial | Participant bootstrap 95% interval |",
             "| --- | ---: | --- |"]
    for family, mean, interval in zip(test["models"], test["scores"]["mean_log_score"],
                                      test["scores"]["mean_log_score_95_interval"]):
        lines.append(f"| {LABELS.get(family, family)} | {mean:.4f} | [{interval[0]:.4f}, {interval[1]:.4f}] |")
    lines += ["", f"Participants by split: `{split['participant_counts']}`. "
              f"Trials by split: `{split['trial_counts']}`.",
              "Lapse was fitted on training trials; cognitive candidates were selected using validation likelihood. "
              "The test set scored those frozen choices without refitting.", "",
              f"![Held-out model comparisons]({figure_prefix}figure1_heldout.png)",
              "", "## Selected parameters", "",
              "| Family | Candidate | Parameters | Movement lapse | Diagnostic lapse support |",
              "| --- | --- | --- | ---: | --- |"]
    for family, row in summary["selected"].items():
        interval = row["profile"]["approx_95_interval"]
        parameters = json.dumps(row["parameters"], sort_keys=True)
        lines.append(f"| {LABELS.get(family, family)} | `{row['candidate_id']}` | `{parameters}` "
                     f"| {row['lapse_rate']:.4f} | [{interval[0]:.4f}, {interval[1]:.4f}] |")
    lines += ["", "## Numerical precision", "", f"Independent numerical repetitions: "
              f"{summary['numerical_precision']['independent_repeat_count']}.",
              "Pooled log-score precision and paired model-gap MC standard errors are recorded in `summary.json`. "
              "These integration diagnostics are distinct from participant bootstrap uncertainty."]
    lines += mc_precision_markdown(summary.get("analytic_mc_precision"))
    lines += ["", "## Primary fitting-protocol recovery", ""]
    protocol = summary.get("protocol_recovery_figure")
    if protocol:
        lines += [f"Primary recovery replays the frozen train/validation/test design using procedure "
                  f"`{protocol['procedure']}` across {protocol['datasets']} synthetic datasets. "
                  "Lapse and stopping parameters are trained, cognitive choices are validated within each family, "
                  "and family ranking uses their frozen test scores including the exact random baseline. "
                  "The output's procedure label distinguishes matched continuous Gamma fitting from a finite-Gamma "
                  "approximation. Independent truth-cache identities and fitted MC precision are preserved "
                  "with the source protocol summaries.", "",
                  f"![Primary protocol recovery]({figure_prefix}figure4_protocol_recovery.png)"]
    else:
        lines.append("Primary fitting-protocol recovery was not supplied; it is unverified.")
    lines += ["", "## Secondary pooled identifiability", ""]
    if recovery_panel:
        lines += [f"The representative figure uses `{recovery_panel['design']}` and "
                  f"{recovery_panel['datasets']:,} synthetic datasets. All designs, generating settings, "
                  "cell-specific recovery rates, coverage, bias, RMSE and expected score gaps remain in `summary.json` "
                  "and the source recovery reports.", "",
                  f"![Model and parameter recovery]({figure_prefix}figure2_recovery.png)"]
    else:
        lines.append("No secondary pooled recovery report was supplied.")
    if summary.get("continuous_gamma_recovery"):
        lines += ["", "Continuous Gamma/lapse recovery uses the empirical multistart optimizer conditional on "
                  "a fixed WM law, with independent generating/fitting trajectory folds. This assesses continuous "
                  "stopping recovery separately from finite-grid joint cognitive/model recovery. Bias, RMSE, "
                  "optimizer success, boundary rates and weak-curvature rates remain in `summary.json`.", "",
                  f"![Continuous stopping recovery]({figure_prefix}figure3_continuous_gamma.png)"]
    lines += ["", "## Scope and interpretation", ""] + [f"- {value}" for value in summary["limitations"]]
    lines += confirmation_markdown(summary.get("independent_truth_confirmation", []))
    lines += sensitivity_markdown(summary.get("matched_main_sensitivity"))
    compute = summary["compute"]
    lines += ["", "## Compute and caching", ""]
    if compute.get("verified"):
        lines.append(f"Supplied Slurm accounting records {compute['cpu_hours']:.3f} allocated CPU hours and "
                     f"{compute['gpu_hours']:.3f} allocated GPU hours. " + compute.get("scope", ""))
        lines.append(f"Requested memory values: `{compute.get('requested_memory_values', [])}`; "
                     f"peak recorded step RSS: `{compute.get('peak_step_rss_gib')}` GiB.")
    else:
        lines.append("Actual Slurm resource usage is unverified because accounting was not supplied.")
    lines += ["Likelihood caches reuse planning histories across all movement lapse rates; undirected trajectory "
              "caches also reuse histories across Gamma stopping settings. Checkpoint seed probabilities are "
              "averaged before logs. Hash-verified cache arrays, manifests and completion receipts preserve "
              "the frozen code, data, checkpoint and RNG identities. "
              "A complete newly trained policy grid is not included.",
              f"Cache scope: `{json.dumps(summary['cache_scope'], sort_keys=True)}`.",
              "", "## Artifacts", "", "- `summary.json`: selected fits, held-out results, "
              "recovery summaries, numerical precision and provenance.",
              "- `figure1_heldout.png` / `.pdf`: held-out comparison and raw participant paired differences.",
              "- `figure2_recovery.png` / `.pdf`: model confusion and finite-grid parameter recovery, when supplied.",
              ""]
    if summary.get("continuous_gamma_recovery"):
        lines += ["- `figure3_continuous_gamma.png` / `.pdf`: "
                  "continuous stopping recovery conditional on fixed WM.", ""]
    if protocol:
        lines += ["- `figure4_protocol_recovery.png` / `.pdf`: primary train/validation/test procedure recovery.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--recovery", nargs="*", type=Path, default=[])
    parser.add_argument("--protocol-recovery", nargs="*", type=Path, default=[])
    parser.add_argument("--continuous-recovery", nargs="*", type=Path, default=[])
    parser.add_argument("--truth-confirmation", nargs="*", type=Path, default=[])
    parser.add_argument("--mc-precision", type=Path)
    parser.add_argument("--sensitivity-fit", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--accounting", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.markdown or args.output_dir / "report.md"
    products = [args.output_dir / name for name in ("summary.json", "figure1_heldout.png", "figure1_heldout.pdf")]
    if args.recovery:
        products += [args.output_dir / name for name in ("figure2_recovery.png", "figure2_recovery.pdf")]
    if args.continuous_recovery:
        products += [args.output_dir / name for name in
                     ("figure3_continuous_gamma.png", "figure3_continuous_gamma.pdf")]
    if args.protocol_recovery:
        products += [args.output_dir / name for name in
                     ("figure4_protocol_recovery.png", "figure4_protocol_recovery.pdf")]
    if not args.overwrite and any(path.exists() for path in [*products, markdown_path]):
        parser.error("an output exists; choose a new output directory or pass --overwrite")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fit = read_json(args.fit)
    recoveries = [read_json(path) for path in args.recovery]
    continuous = [read_json(path) for path in args.continuous_recovery]
    protocol = [read_json(path) for path in args.protocol_recovery]
    mc_precision = read_json(args.mc_precision) if args.mc_precision else None
    confirmations = [read_json(path) for path in args.truth_confirmation]
    sensitivity = read_json(args.sensitivity_fit) if args.sensitivity_fit else None
    sensitivity_result = sensitivity_comparison(fit, sensitivity) if sensitivity is not None else None
    fit_sha256 = identity(args.fit)["sha256"]
    if mc_precision is not None and (mc_precision["split_sha256"] != fit["split"]["split_sha256"]
                                    or mc_precision["provenance"]["fit"]["sha256"] != fit_sha256):
        raise ValueError("MC precision report differs from the frozen fit or participant split")
    for report in confirmations:
        if (report["split_sha256"] != fit["split"]["split_sha256"]
                or report["provenance"]["fit"]["sha256"] != fit_sha256
                or report["parameters_refitted"] or report["models_reselected"]):
            raise ValueError("confirmation report must rescore this frozen fit and split without reselection")
    summary = scientific_summary(fit, recoveries, accounting_summary(args.accounting))
    summary["cache_scope"] = cache_scope(fit)
    summary["provenance"] = {"fit": identity(args.fit), "recovery": [identity(path) for path in args.recovery],
                             "continuous_recovery": [identity(path) for path in args.continuous_recovery],
                             "protocol_recovery": [identity(path) for path in args.protocol_recovery],
                             "truth_confirmation": [identity(path) for path in args.truth_confirmation],
                             "mc_precision": identity(args.mc_precision) if args.mc_precision else None,
                             "sensitivity_fit": identity(args.sensitivity_fit) if args.sensitivity_fit else None,
                             "candidates": identity(args.candidates) if args.candidates else None,
                             "report_script": identity(__file__)}
    fit_figure(fit, args.output_dir)
    recovery_panel = recovery_figure(recoveries, args.output_dir)
    summary["recovery_figure"] = recovery_panel
    summary["continuous_gamma_recovery"] = [{key: report.get(key) for key in
                                             ("method", "scope", "working_memory_law", "repetitions", "cells",
                                              "summary_by_design", "provenance")} for report in continuous]
    summary["continuous_gamma_figure"] = continuous_recovery_figure(continuous, args.output_dir)
    summary["protocol_recovery"] = [{key: value for key, value in report.items() if key != "datasets"}
                                    for report in protocol]
    summary["protocol_recovery_figure"] = protocol_recovery_figure(protocol, args.output_dir)
    summary["independent_truth_confirmation"] = confirmations
    summary["analytic_mc_precision"] = mc_precision
    summary["matched_main_sensitivity"] = ({"fit": sensitivity,
                                            "comparison": sensitivity_result}
                                           if sensitivity is not None else None)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    figure_prefix = os.path.relpath(args.output_dir.resolve(), markdown_path.parent.resolve()) + "/"
    markdown_path.write_text(markdown_report(summary, recovery_panel, figure_prefix))
    print(f"wrote={args.output_dir / 'summary.json'} markdown={markdown_path}")


if __name__ == "__main__":
    main()
