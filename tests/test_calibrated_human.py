import os
from types import SimpleNamespace

import pytest

from katrain.core.ai import ai_rank_estimation
from katrain.core.base_katrain import KaTrainBase
from katrain.core.constants import AI_ANGRY_HUMAN, AI_CALIBRATED_HUMAN, AI_RANK, AI_SIMPLE_OWNERSHIP
from katrain.core.engine import KataGoEngine
from katrain.core.game import Game


class TestCalibratedHuman:
    def test_defaults(self):
        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        settings = katrain.config(f"ai/{AI_CALIBRATED_HUMAN}")
        for strategy in [AI_RANK, AI_SIMPLE_OWNERSHIP, AI_ANGRY_HUMAN]:
            assert set(katrain.config(f"ai/{strategy}")) <= set(settings)
        assert settings["human_kyu_rank"] == 13
        assert settings["kyu_rank"] == 13
        assert settings["static_score_utility"] > 0.1 and settings["dynamic_score_utility"] > 0.3
        assert ai_rank_estimation(AI_CALIBRATED_HUMAN, settings) == -12

    def test_allow_moves_query(self):
        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        sent = []
        engine = SimpleNamespace(
            config=katrain.config("engine"),
            override_settings={},
            base_priority=0,
            get_rules=lambda ruleset: ruleset,
            send_query=lambda query, *args: sent.append(query),
            PONDER_KEY="_kt_continuous",
            katrain=katrain,
        )
        game = Game(katrain, SimpleNamespace(on_new_game=lambda: None, request_analysis=lambda *a, **k: None))
        KataGoEngine.request_analysis(engine, game.root, callback=None, allow_moves=["D4", "Q16"])
        assert sent[0]["allowMoves"] == [{"moves": ["D4", "Q16"], "player": "B", "untilDepth": 1}]
        KataGoEngine.request_analysis(engine, game.root, callback=None)
        assert "allowMoves" not in sent[1]

    @pytest.mark.skipif(not os.environ.get("KATRAIN_HUMAN_MODEL"), reason="needs KATRAIN_HUMAN_MODEL")
    def test_play_with_human_model(self, monkeypatch):
        from katrain.core.ai import CalibratedHumanStrategy, generate_ai_move, shutdown_human_model_engine

        allowed = []
        request_analysis = CalibratedHumanStrategy.request_analysis

        def record_allowed(self, *args, allow_moves=None, **kwargs):
            allowed.append(allow_moves)
            return request_analysis(self, *args, allow_moves=allow_moves, **kwargs)

        monkeypatch.setattr(CalibratedHumanStrategy, "request_analysis", record_allowed)

        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        katrain._config["engine"]["humanlike_model"] = os.environ["KATRAIN_HUMAN_MODEL"]
        engine = KataGoEngine(katrain, katrain.config("engine"))
        game = Game(katrain, engine)
        try:
            for _ in range(6):
                allowed.clear()
                move, node = generate_ai_move(game, AI_CALIBRATED_HUMAN, katrain.config(f"ai/{AI_CALIBRATED_HUMAN}"))
                katrain.log(f"Calibrated Human Style -> {move}: {node.ai_thoughts}", 0)
                assert move.coords is not None
                assert node.ai_thoughts.startswith("Human-like model at rank_13k.")
                assert allowed[0] is None  # the full search for the policy
                if "Searched only these" in node.ai_thoughts:
                    assert len(allowed) == 2 and move.gtp() in allowed[1]
        finally:
            shutdown_human_model_engine(katrain)
            engine.shutdown(finish=False)
