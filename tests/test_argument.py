import argparse

from modules.argument import ArgParser, parse_bool


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
            "--ppo_normalize_advantages",
            "true",
        ],
    )

    args = ArgParser().args
    assert args.seed == 123
    assert args.shuffle_nodes is False
    assert args.mask_fixation is False
    assert args.ppo_normalize_advantages is True
