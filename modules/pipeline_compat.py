PIPELINE_COMPAT_VERSION = 1
PARAMS_FORMAT_VERSION = 1

PIPELINE_COMPAT_KEY = "pipeline_compat_version"
LEGACY_ENVIRONMENT_COMPAT_KEY = "environment_compat_version"


def read_pipeline_compat_version(record: dict, *, source: str):
    version = get_pipeline_compat_version(record)
    if version is None:
        raise ValueError(f"Missing pipeline compatibility version in {source}.")
    return version


def get_pipeline_compat_version(record: dict):
    if PIPELINE_COMPAT_KEY in record:
        return record[PIPELINE_COMPAT_KEY]
    if LEGACY_ENVIRONMENT_COMPAT_KEY in record:
        return record[LEGACY_ENVIRONMENT_COMPAT_KEY]
    return None


def assert_pipeline_compat_version(recorded_version, *, source: str) -> None:
    if recorded_version is None:
        raise ValueError(f"Missing pipeline compatibility version in {source}.")

    try:
        version = int(recorded_version)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid pipeline compatibility version in {source}: {recorded_version!r}") from error

    if version != PIPELINE_COMPAT_VERSION:
        raise ValueError(
            "Pipeline compatibility version mismatch: "
            f"{source} has {version}, current pipeline has {PIPELINE_COMPAT_VERSION}."
        )
