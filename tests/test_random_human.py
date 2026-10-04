import os
import random
from types import SimpleNamespace

import pytest

from katrain.core.ai import RandomRankedHumanStrategy, ai_rank_estimation, random_human_personality
from katrain.core.base_katrain import KaTrainBase
from katrain.core.constants import (
    AI_CALIBRATED_HUMAN,
    AI_RANDOM_HUMAN,
    AI_RANDOM_HUMAN_NEVER_ZERO,
    AI_RANDOM_HUMAN_RANGES,
    AI_RANDOM_HUMAN_ZERO_CHANCE,
    AI_STRATEGIES,
    AI_STRATEGIES_RECOMMENDED_ORDER,
)
from katrain.core.engine import KataGoEngine
from katrain.core.game import Game
from katrain.core.katago_settings import KATAGO_SETTINGS_CONFIG_SECTION, clean_katago_overrides

HIDDEN = {
    "attach_penalty",
    "dynamic_score_utility",
    "human_kyu_rank",
    "opponent_fac",
    "settled_weight",
    "tenuki_penalty",
}


def fake_game(next_player="B"):
    engine = SimpleNamespace(query_generation=0)
    return SimpleNamespace(
        engines={"B": engine, "W": engine},
        katrain=SimpleNamespace(log=lambda *args: None),
        current_node=SimpleNamespace(next_player=next_player),
        ai_personalities={},
    )


class TestRandomHuman:
    def test_defaults(self):
        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        settings = katrain.config(f"ai/{AI_RANDOM_HUMAN}")
        calibrated = katrain.config(f"ai/{AI_CALIBRATED_HUMAN}")
        assert not HIDDEN & set(settings)  # not shown in the AI settings
        assert set(settings) | HIDDEN == set(calibrated)
        assert settings == {
            "kyu_rank": 13,
            "max_points_lost": 300,
            "min_visits": 1,
            "pick_scale": 1.0,
            "pick_weighting": 0.3,
            "static_score_utility": 0.5,
        }
        katago = katrain.config(f"{KATAGO_SETTINGS_CONFIG_SECTION}/{AI_RANDOM_HUMAN}")
        assert katago == {"winLossUtilityFactor": 0.7, "playoutDoublingAdvantage": 0.6}
        assert clean_katago_overrides(katago) == katago  # valid KataGo settings
        assert AI_RANDOM_HUMAN in AI_STRATEGIES and AI_RANDOM_HUMAN in AI_STRATEGIES_RECOMMENDED_ORDER
        assert ai_rank_estimation(AI_RANDOM_HUMAN, settings) == -12  # 13k

    def test_katago_defaults_for_existing_configs(self):
        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        katrain._config.pop(KATAGO_SETTINGS_CONFIG_SECTION)  # a config from before these defaults
        katrain._add_missing_ai_settings()
        assert katrain.config(f"{KATAGO_SETTINGS_CONFIG_SECTION}/{AI_RANDOM_HUMAN}")["winLossUtilityFactor"] == 0.7
        katrain._config[KATAGO_SETTINGS_CONFIG_SECTION][AI_RANDOM_HUMAN] = {}  # the user cleared them
        katrain._add_missing_ai_settings()
        assert katrain.config(f"{KATAGO_SETTINGS_CONFIG_SECTION}/{AI_RANDOM_HUMAN}") == {}
        katrain._config[KATAGO_SETTINGS_CONFIG_SECTION] = "corrupt"
        katrain._add_missing_ai_settings()
        assert katrain.config(f"{KATAGO_SETTINGS_CONFIG_SECTION}/{AI_RANDOM_HUMAN}")["winLossUtilityFactor"] == 0.7

    def test_personality_ranges(self, monkeypatch):
        assert set(AI_RANDOM_HUMAN_RANGES) | {"human_kyu_rank"} == HIDDEN
        assert AI_RANDOM_HUMAN_RANGES == {
            "attach_penalty": (-2.0, 2.0),
            "dynamic_score_utility": (0.5, 1.0),
            "opponent_fac": (-1.0, 1.0),
            "settled_weight": (-0.3, 1.0),
            "tenuki_penalty": (-3.0, 3.0),
        }
        assert AI_RANDOM_HUMAN_ZERO_CHANCE == 0.5 and AI_RANDOM_HUMAN_NEVER_ZERO == {"dynamic_score_utility"}
        rolls = [random_human_personality(fake_game(), "B") for _ in range(2000)]
        for k, (lo, hi) in AI_RANDOM_HUMAN_RANGES.items():
            values = [p[k] for p in rolls]
            assert all(v == 0 or (lo <= v <= hi and round(v, 1) == v) for v in values)
            zeros = values.count(0) / len(values)
            if k in AI_RANDOM_HUMAN_NEVER_ZERO:
                assert zeros == 0
            else:  # the 50%, plus rolling a zero in the range
                rolled = 1 / (round(hi * 10) - round(lo * 10) + 1)
                assert abs(zeros - (0.5 + 0.5 * rolled)) < 0.06
        zeroable = set(AI_RANDOM_HUMAN_RANGES) - AI_RANDOM_HUMAN_NEVER_ZERO
        all_zero = sum(all(p[k] == 0 for k in zeroable) for p in rolls)
        assert all_zero < 400  # each option gets its own 50% chance, not one chance for all of them (~140 expected)
        for chance, pick, expected in [(0.49, min, "zero"), (0.5, min, "lo"), (0.99, max, "hi")]:
            monkeypatch.setattr(random, "random", lambda chance=chance: chance)
            monkeypatch.setattr(random, "randint", lambda a, b, pick=pick: pick(a, b))
            for k, v in random_human_personality(fake_game(), "B").items():
                lo, hi = AI_RANDOM_HUMAN_RANGES[k]
                if expected == "zero" and k not in AI_RANDOM_HUMAN_NEVER_ZERO:
                    assert v == 0.0
                else:
                    assert v == (hi if expected == "hi" else lo)

    def test_personality_per_player_and_game(self, monkeypatch):
        rng = random.Random(1)  # with zeros common, two rolls could match by chance
        monkeypatch.setattr(random, "random", rng.random)
        monkeypatch.setattr(random, "randint", rng.randint)
        game = fake_game()
        black = random_human_personality(game, "B")
        assert random_human_personality(game, "B") is black  # fixed for the game
        others = [random_human_personality(game, "W")] + [random_human_personality(fake_game(), "B") for _ in range(3)]
        assert all(p != black for p in others)  # the other player, and new games, roll their own

    def test_settings(self):
        game = fake_game("W")
        settings = {"kyu_rank": 7, "pick_scale": 1.0, "static_score_utility": 0.5, "settled_weight": 99}
        strategy = RandomRankedHumanStrategy(game, settings)
        strategy.use_personality()
        assert strategy.personality is game.ai_personalities["W"]
        assert strategy.settings == {**settings, **strategy.personality, "human_kyu_rank": 7}
        strategy.human_engine = SimpleNamespace(query_generation=0)
        search = strategy.own_search_settings()
        assert search["humanSLProfile"] == "rank_7k"
        assert search["dynamicScoreUtilityFactor"] == strategy.personality["dynamic_score_utility"]
        assert search["staticScoreUtilityFactor"] == 0.5

    def test_reported_in_thoughts(self, monkeypatch):
        from katrain.core.ai import CalibratedHumanStrategy

        monkeypatch.setattr(CalibratedHumanStrategy, "generate_move", lambda self: ("move", "Thoughts."))
        game = fake_game()
        game.ai_personalities["B"] = {
            "attach_penalty": -2.0,
            "dynamic_score_utility": 1.0,
            "opponent_fac": 0.5,
            "settled_weight": -0.3,
            "tenuki_penalty": 0.0,
        }
        move, thoughts = RandomRankedHumanStrategy(game, {"kyu_rank": 4.0}).generate_move()
        assert thoughts == (
            "Personality: attach penalty -2.0, dynamic score utility 1.0, opponent fac 0.5, settled weight -0.3, "
            "tenuki penalty 0.0, human kyu rank 4 (rank_4k). Thoughts."
        )

    def test_katago_settings_keep_the_personality(self):
        katago = {"dynamicScoreUtilityFactor": 0.0, "humanSLProfile": "rank_9d", "cpuctExploration": 2.0}
        strategy = RandomRankedHumanStrategy(fake_game(), {"kyu_rank": 4, "static_score_utility": 0.5}, katago)
        assert strategy.katago_overrides == {"cpuctExploration": 2.0}
        assert strategy.katago_settings == {"cpuctExploration": 2.0}

    def test_real_game_has_personalities(self):
        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        game = Game(katrain, SimpleNamespace(on_new_game=lambda: None, request_analysis=lambda *a, **k: None))
        assert game.ai_personalities == {}

    @pytest.mark.skipif(not os.environ.get("KATRAIN_HUMAN_MODEL"), reason="needs KATRAIN_HUMAN_MODEL")
    def test_ai_vs_ai(self):
        from katrain.core.ai import generate_ai_move, shutdown_human_model_engine

        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        katrain._config["engine"]["humanlike_model"] = os.environ["KATRAIN_HUMAN_MODEL"]
        engine = KataGoEngine(katrain, katrain.config("engine"))
        game = Game(katrain, engine)
        settings = katrain.config(f"ai/{AI_RANDOM_HUMAN}")
        katago = katrain.config(f"{KATAGO_SETTINGS_CONFIG_SECTION}/{AI_RANDOM_HUMAN}")
        thoughts = {"B": set(), "W": set()}
        try:
            for _ in range(6):
                player = game.current_node.next_player
                move, node = generate_ai_move(game, AI_RANDOM_HUMAN, settings, katago)
                katrain.log(f"Random Ranked Human -> {move}: {node.ai_thoughts}", 0)
                assert move.coords is not None
                personality = node.ai_thoughts.split(". ")[0]
                assert personality.startswith("Personality: attach penalty ")
                assert "human kyu rank 13 (rank_13k). Human-like model at rank_13k." in node.ai_thoughts
                assert "winLossUtilityFactor=0.7, playoutDoublingAdvantage=0.6" in node.ai_thoughts
                thoughts[player].add(personality)
            assert len(thoughts["B"]) == 1 and len(thoughts["W"]) == 1  # the same all game
            # rolled separately (with zeros common they can match by chance, so not compared here)
            assert game.ai_personalities["B"] is not game.ai_personalities["W"]
            assert set(game.ai_personalities) == {"B", "W"}
        finally:
            shutdown_human_model_engine(katrain)
            engine.shutdown(finish=False)
