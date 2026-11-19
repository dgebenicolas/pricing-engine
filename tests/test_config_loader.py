import pytest

from pricing_engine.config_loader import CONFIG_DIR, config_path, load_yaml


def test_config_dir_exists():
    assert CONFIG_DIR.is_dir()


def test_load_yaml_roundtrip(tmp_path):
    f = tmp_path / "basis.yaml"
    f.write_text("interest: 0.055\nlapse: null\n", encoding="utf-8")
    assert load_yaml(f) == {"interest": 0.055, "lapse": None}


def test_load_yaml_empty_file(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("", encoding="utf-8")
    assert load_yaml(f) == {}


def test_load_yaml_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


def test_config_path_missing():
    with pytest.raises(FileNotFoundError):
        config_path("does_not_exist.yaml")
