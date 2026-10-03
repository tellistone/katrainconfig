from types import SimpleNamespace

import pytest

from katrain.core.ai import STRATEGY_REGISTRY
from katrain.core.katago_settings import (
    KATAGO_PARAMS,
    KATAGO_PARAMS_BY_NAME,
    clean_katago_overrides,
    split_katago_overrides,
)


class TestKataGoSettings:
    def test_catalog(self):
        assert len(KATAGO_PARAMS_BY_NAME) == len(KATAGO_PARAMS)
        for param in KATAGO_PARAMS:
            assert param.kind in ("int", "float", "bool", "choice", "str")
            if param.kind in ("int", "float"):
                assert param.min is not None and param.max is not None and param.min <= param.max
            if param.kind == "choice":
                assert param.choices

    def test_parse(self):
        params = KATAGO_PARAMS_BY_NAME
        assert params["maxVisits"].parse("40") == 40
        assert params["rootPolicyTemperature"].parse("1.5") == 1.5
        assert params["useLcbForSelection"].parse("false") is False
        assert params["playoutDoublingAdvantagePla"].parse("W") == "W"
        for name, bad in [
            ("maxVisits", "0"),
            ("maxVisits", "2.5"),
            ("maxVisits", "inf"),
            ("cpuctExploration", "11"),
            ("cpuctExploration", "nan"),
            ("noisePruningCap", "inf"),
            ("antiMirror", "x"),
        ]:
            with pytest.raises(ValueError):
                params[name].parse(bad)

    def test_clean_and_split(self):
        cleaned = clean_katago_overrides(
            {"maxVisits": 30, "wideRootNoise": "0.1", "nope": 1, "lcbStdevs": 100, "fpuParentWeight": 0.5}
        )
        assert cleaned == {"maxVisits": 30, "wideRootNoise": 0.1, "fpuParentWeight": 0.5}
        visits, settings = split_katago_overrides(cleaned)
        assert visits == 30
        assert settings == {"wideRootNoise": 0.1}  # fpuParentWeight only applies without fpuParentWeightByVisitedPolicy
        _, settings = split_katago_overrides({"fpuParentWeightByVisitedPolicy": False, "fpuParentWeight": 0.5})
        assert settings == {"fpuParentWeightByVisitedPolicy": False, "fpuParentWeight": 0.5}

    def test_clean_ignores_malformed_section(self):
        assert clean_katago_overrides(["maxVisits", 30]) == {}
        assert clean_katago_overrides(None) == {}

    def test_every_strategy_accepts_katago_settings(self):
        game = SimpleNamespace(engines={}, katrain=SimpleNamespace(log=lambda *args: None), current_node=None)
        for strategy_class in set(STRATEGY_REGISTRY.values()):
            strategy = strategy_class(game, {}, {"maxVisits": 30, "wideRootNoise": 0.1})
            assert strategy.katago_visits == 30
            assert strategy.katago_overrides == {"wideRootNoise": 0.1}
