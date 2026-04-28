import argparse

from modules.argument import ArgParser, parse_bool, parse_recency_decay


def test_parse_bool_accepts_true_false_variants():
    assert parse_bool("true") is True
    assert parse_bool("False") is False
    assert parse_bool("1") is True
    assert parse_bool("0") is False
    assert parse_bool("yes") is True
    assert parse_bool("no") is False


def test_parse_bool_rejects_invalid_value():
    try:
        parse_bool("maybe")
    except argparse.ArgumentTypeError:
        return

    assert False, "parse_bool should reject invalid boolean strings"


def test_parse_recency_decay_accepts_modes_and_valid_numbers():
    assert parse_recency_decay("off") == "off"
    assert parse_recency_decay("auto") == "auto"
    assert parse_recency_decay("0") == 0.0
    assert parse_recency_decay("0.5") == 0.5


def test_parse_recency_decay_rejects_invalid_values():
    for value in ["1", "-0.1", "bad"]:
        try:
            parse_recency_decay(value)
        except argparse.ArgumentTypeError:
            continue

        assert False, f"parse_recency_decay should reject {value!r}"


def test_arg_parser_parses_seed_and_bool_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--seed",
            "123",
            "--shuffle_nodes",
            "false",
            "--mask_fixation",
            "0",
            "--canonicalize",
            "--ppo_normalize_advantages",
            "true",
            "--lamda_backup",
            "0.6",
            "--recency_decay",
            "auto",
        ],
    )

    args = ArgParser().args
    assert args.seed == 123
    assert args.shuffle_nodes is False
    assert args.mask_fixation is False
    assert args.canonicalize is True
    assert args.ppo_normalize_advantages is True
    assert args.lamda_backup == 0.6
    assert args.recency_decay == "auto"
