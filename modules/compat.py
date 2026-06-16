COMPAT_VERSION = 5
PARAMS_FORMAT_VERSION = 1

COMPAT_KEY = "compat_version"


def read_compat_version(record: dict, *, source: str):
    version = get_compat_version(record)
    if version is None:
        raise ValueError(f"Missing compatibility version in {source}.")
    return version


def get_compat_version(record: dict):
    if COMPAT_KEY in record:
        return record[COMPAT_KEY]
    return None


def assert_compat_version(recorded_version, *, source: str) -> None:
    if recorded_version is None:
        raise ValueError(f"Missing compatibility version in {source}.")

    try:
        version = int(recorded_version)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid compatibility version in {source}: {recorded_version!r}") from error

    if version != COMPAT_VERSION:
        raise ValueError(
            "Compatibility version mismatch: "
            f"{source} has {version}, current version is {COMPAT_VERSION}."
        )
