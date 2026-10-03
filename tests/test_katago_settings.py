import pytest

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
        for name, bad in [("maxVisits", "0"), ("maxVisits", "2.5"), ("cpuctExploration", "11"), ("antiMirror", "x")]:
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
