ENVIRONMENT_COMPAT_VERSION = 1
PARAMS_FORMAT_VERSION = 1


def assert_environment_compat_version(recorded_version, *, source: str) -> None:
    if recorded_version is None:
        raise ValueError(f"Missing environment compatibility version in {source}.")

    try:
        version = int(recorded_version)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid environment compatibility version in {source}: {recorded_version!r}") from error

    if version != ENVIRONMENT_COMPAT_VERSION:
        raise ValueError(
            "Environment compatibility version mismatch: "
            f"{source} has {version}, current environment has {ENVIRONMENT_COMPAT_VERSION}."
        )

