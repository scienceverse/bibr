from pathlib import Path


def test_parse_env_strips_double_quotes(tmp_path: Path):
    from bibr.env_utils import parse_env

    p = tmp_path / ".env"
    p.write_text('REDIS_PASSWORD="my password"\n')
    assert parse_env(p) == {"REDIS_PASSWORD": "my password"}


def test_parse_env_strips_single_quotes(tmp_path: Path):
    from bibr.env_utils import parse_env

    p = tmp_path / ".env"
    p.write_text("KEY='val'\n")
    assert parse_env(p) == {"KEY": "val"}


def test_parse_env_keeps_internal_quotes(tmp_path: Path):
    from bibr.env_utils import parse_env

    p = tmp_path / ".env"
    p.write_text('KEY=he said "hi"\n')
    # Internal quotes (not surrounding) must be preserved.
    assert parse_env(p) == {"KEY": 'he said "hi"'}


def test_merge_env_round_trips_unquoted(tmp_path: Path):
    from bibr.env_utils import merge_env, parse_env

    p = tmp_path / ".env"
    p.write_text('REDIS_PASSWORD="my password"\n')
    merge_env(p, {"NEW_KEY": "hello"})
    out = parse_env(p)
    assert out["REDIS_PASSWORD"] == "my password"
    assert out["NEW_KEY"] == "hello"
