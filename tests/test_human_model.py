import threading
from types import SimpleNamespace

import pytest

from katrain.core.ai import HumanModelStrategy, human_model_engine, human_sl_rank_profile
from katrain.core.base_katrain import KaTrainBase
from katrain.core.constants import AI_CALIBRATED_HUMAN, AI_SIMPLE_OWNERSHIP


class TestHumanModel:
    def test_rank_profile(self):
        assert human_sl_rank_profile(13) == "rank_13k"
        assert human_sl_rank_profile(1) == "rank_1k"
        assert human_sl_rank_profile(0) == "rank_1d"
        assert human_sl_rank_profile(-8) == "rank_9d"

    def test_search_settings(self):
        engine = SimpleNamespace(query_generation=0)
        game = SimpleNamespace(
            engines={"B": engine, "W": engine},
            katrain=SimpleNamespace(log=lambda *args: None),
            current_node=SimpleNamespace(player="B"),
        )
        settings = {"human_kyu_rank": 5, "static_score_utility": 0.5, "dynamic_score_utility": 0.6}
        strategy = HumanModelStrategy(game, settings)
        assert strategy.own_search_settings() == {"staticScoreUtilityFactor": 0.5, "dynamicScoreUtilityFactor": 0.6}
        strategy.human_engine = SimpleNamespace(query_generation=0)
        assert strategy.own_search_settings() == {
            "staticScoreUtilityFactor": 0.5,
            "dynamicScoreUtilityFactor": 0.6,
            "humanSLProfile": "rank_5k",
            "ignorePreRootHistory": False,
        }

    def test_no_human_model(self):
        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        katrain._config["engine"]["humanlike_model"] = ""
        assert human_model_engine(katrain) is None
        katrain._config["engine"]["humanlike_model"] = "/nonexistent/human.bin.gz"
        assert human_model_engine(katrain) is None

    def test_user_config_gets_new_opponents(self):
        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        katrain._config["ai"] = {AI_CALIBRATED_HUMAN: {"human_kyu_rank": 3}}
        katrain._add_missing_ai_settings()
        katrain._config["ai"][AI_SIMPLE_OWNERSHIP] = "corrupt"
        katrain._add_missing_ai_settings()
        assert katrain._config["ai"][AI_CALIBRATED_HUMAN]["human_kyu_rank"] == 3
        assert katrain._config["ai"][AI_CALIBRATED_HUMAN]["static_score_utility"] == 0.5
        assert AI_SIMPLE_OWNERSHIP in katrain._config["ai"]

    def _strategy(self):
        main = SimpleNamespace(query_generation=0)
        game = SimpleNamespace(
            engines={"B": main, "W": main},
            katrain=SimpleNamespace(log=lambda *args: None),
            current_node=SimpleNamespace(player="B"),
        )
        settings = {"human_kyu_rank": 13, "static_score_utility": 0.5, "dynamic_score_utility": 0.6}
        return HumanModelStrategy(game, settings), main

    def test_dead_human_engine_is_an_error(self):
        strategy, _ = self._strategy()
        strategy.human_engine = SimpleNamespace(
            query_generation=0,
            katago_process=None,
            request_analysis=lambda *args, **kwargs: None,
            check_alive=lambda **kwargs: False,
        )
        assert strategy.request_analysis({}) is None  # returns instead of waiting forever

    def test_shutdown_or_new_game_stops_the_wait(self):
        from katrain.core.ai import AnalysisDiscardedException, shutdown_human_model_engine

        strategy, main = self._strategy()
        human = SimpleNamespace(
            query_generation=0,
            katago_process=object(),
            thread_lock=threading.RLock(),
            check_alive=lambda **kwargs: True,
            shutdown=lambda finish: None,
        )
        katrain = SimpleNamespace(_human_model_engine=({}, human))
        human.request_analysis = lambda *args, **kwargs: shutdown_human_model_engine(katrain)
        strategy.human_engine = human
        with pytest.raises(AnalysisDiscardedException):
            strategy.request_analysis({})
        assert katrain._human_model_engine == (None, None)

        strategy, main = self._strategy()
        human.request_analysis = lambda *args, **kwargs: setattr(main, "query_generation", 1)
        strategy.human_engine = human
        with pytest.raises(AnalysisDiscardedException):
            strategy.request_analysis({})

    def test_human_search_settings_need_human_model(self):
        sent = []
        strategy, _ = self._strategy()

        def request(*args, extra_settings=None, callback=None, **kwargs):
            sent.append(extra_settings)
            callback({"moveInfos": []}, False)

        for command in [["katago", "analysis", "-model", "m"], ["katago", "-model", "h", "-human-model", "h"]]:
            strategy.human_engine = SimpleNamespace(
                query_generation=0, command=command, request_analysis=request, check_alive=lambda **kwargs: True
            )
            strategy.katago_overrides = {"humanSLCpuctExploration": 2.0, "humanSLProfile": "rank_1d"}
            strategy.request_analysis({})
        assert "humanSLCpuctExploration" not in sent[0] and sent[0]["humanSLProfile"] == "rank_1d"
        assert sent[1]["humanSLCpuctExploration"] == 2.0

    def test_angry_human_style_removed(self):
        from katrain.core.ai import STRATEGY_REGISTRY
        from katrain.core.constants import AI_STRATEGIES, AI_STRATEGIES_RECOMMENDED_ORDER, AI_STRENGTH

        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        for strategies in [STRATEGY_REGISTRY, AI_STRATEGIES, AI_STRATEGIES_RECOMMENDED_ORDER, AI_STRENGTH]:
            assert "ai:angryhuman" not in strategies
        assert "ai:angryhuman" not in katrain.config("ai")
        assert HumanModelStrategy not in STRATEGY_REGISTRY.values()
