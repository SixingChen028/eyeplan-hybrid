import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from modules.analysis_targets import list_experiment_run_dirs


@dataclass(frozen=True)
class Query:
    key: str
    op: str  # "eq", "lt", "gt", "le", "ge"
    raw_value: str
    value: object


def _parse_scalar(value: str) -> object:
    text = value.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." not in text and "e" not in lower:
            return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_query_value(raw_value: str) -> tuple[str, object]:
    for prefix, op in ((">=", "ge"), ("<=", "le"), (">", "gt"), ("<", "lt")):
        if raw_value.startswith(prefix):
            return op, _parse_scalar(raw_value[len(prefix):])
    return "eq", _parse_scalar(raw_value)


def _parse_query_tokens(tokens: list[str]) -> list[Query]:
    queries: list[Query] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected token: {token}")

        if "=" in token:
            key_token, raw_value = token.split("=", 1)
            i += 1
        else:
            key_token = token
            if i + 1 >= len(tokens):
                raise ValueError(f"Missing value for argument: {token}")
            raw_value = tokens[i + 1]
            i += 2

        key = key_token[2:]
        if key == "":
            raise ValueError(f"Invalid argument: {key_token}")

        op, value = _parse_query_value(raw_value)
        queries.append(Query(key=key, op=op, raw_value=raw_value, value=value))
    return queries


def _parse_within(value: str) -> timedelta:
    text = value.strip().lower()
    units = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }
    if len(text) < 2 or text[-1] not in units:
        raise ValueError("Invalid --within format. Use forms like 30m, 2h, 2d, 1w.")

    amount_text = text[:-1]
    try:
        amount = float(amount_text)
    except ValueError as exc:
        raise ValueError(f"Invalid --within amount: {amount_text}") from exc
    if amount < 0:
        raise ValueError("--within must be non-negative.")
    return timedelta(seconds=amount * units[text[-1]])


def _read_json(path: str) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def _to_float(value: object) -> float | None:
    try:
        result = float(value)
        if result != result:
            return None
        return result
    except (TypeError, ValueError):
        return None


def _matches_query(actual: object, query: Query) -> bool:
    if query.op == "eq":
        q_num = _to_float(query.value)
        a_num = _to_float(actual)
        if q_num is not None and a_num is not None:
            return a_num == q_num
        return str(actual) == str(query.value)

    q_num = _to_float(query.value)
    a_num = _to_float(actual)
    if q_num is None or a_num is None:
        return False

    if query.op == "lt":
        return a_num < q_num
    if query.op == "gt":
        return a_num > q_num
    if query.op == "le":
        return a_num <= q_num
    if query.op == "ge":
        return a_num >= q_num
    return False


def _get_started_at(metadata: dict, run_dir: str) -> datetime:
    started_at = metadata.get("started_at_utc")
    if isinstance(started_at, str):
        try:
            parsed = datetime.fromisoformat(started_at)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(run_dir), tz=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find run directories matching metadata args filters."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=os.path.join(os.getcwd(), "results"),
        help="Results root containing runs/<experiment>/...",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment name under results_root/runs/<experiment>.",
    )
    parser.add_argument(
        "--within",
        type=str,
        default=None,
        help="Only include runs started within this duration, e.g. 2d, 12h, 30m.",
    )

    args, query_tokens = parser.parse_known_args()
    queries = _parse_query_tokens(query_tokens)
    within_delta = _parse_within(args.within) if args.within else None
    cutoff = (
        datetime.now(timezone.utc) - within_delta if within_delta is not None else None
    )

    run_dirs = list_experiment_run_dirs(args.results_root, args.experiment)
    if not run_dirs:
        raise FileNotFoundError(
            f"No runs found for experiment '{args.experiment}' in '{args.results_root}'."
        )

    matches: list[tuple[datetime, str, dict]] = []
    for run_dir in run_dirs:
        metadata_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            continue

        metadata = _read_json(metadata_path)
        started_at = _get_started_at(metadata, run_dir)
        if cutoff is not None and started_at < cutoff:
            continue

        arg_values = metadata.get("args", {})
        if not isinstance(arg_values, dict):
            continue

        if any(
            query.key not in arg_values
            or not _matches_query(arg_values.get(query.key), query)
            for query in queries
        ):
            continue

        matches.append((started_at, run_dir, arg_values))

    matches.sort(key=lambda item: (item[0], item[1]))
    for _, run_dir, arg_values in matches:
        parts = [run_dir]
        for query in queries:
            parts.append(f"{query.key}={arg_values.get(query.key)}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
