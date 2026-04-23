import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedAnalysisTarget:
    experiment: str
    run_dirs: list[str]
    kind: str  # "run", "experiment", or "wildcard"


def _to_abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _is_run_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "metadata.json"))


def _list_direct_run_dirs(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    run_dirs: list[str] = []
    for entry in os.scandir(root):
        if entry.is_dir() and _is_run_dir(entry.path):
            run_dirs.append(_to_abs(entry.path))
    run_dirs.sort()
    return run_dirs


def _list_direct_dirs(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    dirs: list[str] = []
    for entry in os.scandir(root):
        if entry.is_dir():
            dirs.append(_to_abs(entry.path))
    dirs.sort()
    return dirs


def _infer_experiment_from_rel_parts(rel_parts: list[str], fallback: str) -> str:
    if len(rel_parts) >= 2 and rel_parts[0] == "runs":
        return rel_parts[1]
    if len(rel_parts) >= 2 and rel_parts[0] == "analysis":
        return rel_parts[1]
    if len(rel_parts) >= 1 and rel_parts[0] not in {"", ".", "runs", "analysis"}:
        return rel_parts[0]
    return fallback


def _safe_rel_parts(path: str, root: str) -> list[str]:
    try:
        if os.path.commonpath([_to_abs(path), _to_abs(root)]) != _to_abs(root):
            return []
        rel = os.path.relpath(_to_abs(path), _to_abs(root))
    except ValueError:
        return []
    return rel.split(os.sep)


def _dedupe_sorted(paths: list[str]) -> list[str]:
    return sorted(set(_to_abs(path) for path in paths))


def _experiment_roots(results_root: str, experiment: str) -> list[str]:
    roots = [
        os.path.join(results_root, "runs", experiment),
        os.path.join(results_root, experiment),  # legacy layout
    ]
    deduped: list[str] = []
    seen = set()
    for root in roots:
        abs_root = _to_abs(root)
        if abs_root not in seen:
            deduped.append(abs_root)
            seen.add(abs_root)
    return deduped


def get_experiment_runs_dir(results_root: str, experiment: str) -> str:
    return _to_abs(os.path.join(results_root, "runs", experiment))


def list_experiment_run_dirs(results_root: str, experiment: str) -> list[str]:
    run_dirs: list[str] = []
    for root in _experiment_roots(results_root, experiment):
        run_dirs.extend(_list_direct_run_dirs(root))
    return _dedupe_sorted(run_dirs)


def list_experiment_candidate_dirs(results_root: str, experiment: str) -> list[str]:
    dirs: list[str] = []
    for root in _experiment_roots(results_root, experiment):
        dirs.extend(_list_direct_dirs(root))
    return _dedupe_sorted(dirs)


def _find_run_dir(results_root: str, experiment: str, run_id: str) -> str | None:
    for root in _experiment_roots(results_root, experiment):
        candidate = os.path.join(root, run_id)
        if _is_run_dir(candidate):
            return _to_abs(candidate)
    return None


def _infer_experiment_from_path(path: str, results_root: str) -> str:
    rel_parts = _safe_rel_parts(path, results_root)
    fallback = os.path.basename(os.path.dirname(_to_abs(path)))
    return _infer_experiment_from_rel_parts(rel_parts, fallback=fallback)


def resolve_analysis_target(target: str, results_root: str) -> ResolvedAnalysisTarget:
    results_root = _to_abs(results_root)
    if not target or not target.strip():
        raise ValueError("target must be non-empty")

    target = target.strip()
    wildcard = target.endswith("/*")
    base_target = target[:-2] if wildcard else target
    if not base_target:
        raise ValueError("Invalid target: wildcard target must include an experiment or path before '/*'")

    if os.path.exists(os.path.expanduser(base_target)):
        abs_base = _to_abs(base_target)
        if _is_run_dir(abs_base):
            experiment = _infer_experiment_from_path(abs_base, results_root)
            kind = "wildcard" if wildcard else "run"
            return ResolvedAnalysisTarget(experiment=experiment, run_dirs=[abs_base], kind=kind)

        if not os.path.isdir(abs_base):
            raise FileNotFoundError(f"Target exists but is not a directory: {abs_base}")

        run_dirs = _list_direct_run_dirs(abs_base)
        if run_dirs:
            experiment = _infer_experiment_from_path(abs_base, results_root)
            kind = "wildcard" if wildcard else "experiment"
            return ResolvedAnalysisTarget(experiment=experiment, run_dirs=run_dirs, kind=kind)

        rel_parts = _safe_rel_parts(abs_base, results_root)
        if len(rel_parts) >= 4 and rel_parts[0] == "analysis" and rel_parts[2] == "runs":
            experiment = rel_parts[1]
            run_id = rel_parts[3]
            run_dir = _find_run_dir(results_root, experiment, run_id)
            if run_dir is not None:
                kind = "wildcard" if wildcard else "run"
                return ResolvedAnalysisTarget(experiment=experiment, run_dirs=[run_dir], kind=kind)

        experiment = _infer_experiment_from_path(abs_base, results_root)
        run_dirs = list_experiment_run_dirs(results_root, experiment)
        if run_dirs:
            kind = "wildcard" if wildcard else "experiment"
            return ResolvedAnalysisTarget(experiment=experiment, run_dirs=run_dirs, kind=kind)
        raise FileNotFoundError(f"No run directories found for target path: {abs_base}")

    shorthand = base_target.strip("/")
    if "/" in shorthand:
        experiment, run_id = shorthand.split("/", 1)
        if run_id == "*":
            wildcard = True
        else:
            run_dir = _find_run_dir(results_root, experiment, run_id)
            if run_dir is None:
                raise FileNotFoundError(
                    f"Run not found for target '{target}'. Expected under '{results_root}/runs/{experiment}/{run_id}' "
                    f"or '{results_root}/{experiment}/{run_id}'."
                )
            kind = "wildcard" if wildcard else "run"
            return ResolvedAnalysisTarget(experiment=experiment, run_dirs=[run_dir], kind=kind)

    experiment = shorthand.split("/", 1)[0]
    run_dirs = list_experiment_run_dirs(results_root, experiment)
    if not run_dirs:
        raise FileNotFoundError(
            f"Experiment not found for target '{target}'. Expected run directories under "
            f"'{results_root}/runs/{experiment}' or '{results_root}/{experiment}'."
        )
    kind = "wildcard" if wildcard else "experiment"
    return ResolvedAnalysisTarget(experiment=experiment, run_dirs=run_dirs, kind=kind)


def select_most_recent_run(run_dirs: list[str]) -> str:
    if not run_dirs:
        raise ValueError("run_dirs must contain at least one run directory")
    return max(run_dirs, key=lambda path: (os.path.getmtime(path), os.path.basename(path)))


def get_run_analysis_dir(results_root: str, experiment: str, run_id: str) -> str:
    return os.path.join(_to_abs(results_root), "analysis", experiment, "runs", run_id)


def get_summary_analysis_dir(results_root: str, experiment: str) -> str:
    return os.path.join(_to_abs(results_root), "analysis", experiment, "summary")
