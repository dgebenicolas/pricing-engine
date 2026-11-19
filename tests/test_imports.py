import importlib

import pytest

SUBPACKAGES = [
    "pricing_engine",
    "pricing_engine.config_loader",
    "pricing_engine.model_point",
    "pricing_engine.assumptions",
    "pricing_engine.products",
    "pricing_engine.engine",
    "pricing_engine.analysis",
    "pricing_engine.app",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_import(name):
    importlib.import_module(name)
