import os
import random
from types import SimpleNamespace

import pytest

from katrain.core.ai import RandomRankedHumanStrategy, ai_rank_estimation, random_human_personality
from katrain.core.base_katrain import KaTrainBase
from katrain.core.constants import (
    AI_CALIBRATED_HUMAN,
    AI_RANDOM_HUMAN,
    AI_RANDOM_HUMAN_MAX_WIN_LOSS_UTILITY,
    AI_RANDOM_HUMAN_NEVER_ZERO,
    AI_RANDOM_HUMAN_OLD_KATAGO_DEFAULTS,
    AI_RANDOM_HUMAN_RANGES,
    AI_RANDOM_HUMAN_UTILITY_TOTAL,
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
ROLLED_KATAGO = {"playout_doubling_advantage", "win_loss_utility"}  # KataGo settings, not AI settings


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
        # winLossUtilityFactor and playoutDoublingAdvantage are now in the personality
        assert katrain.config(f"{KATAGO_SETTINGS_CONFIG_SECTION}/{AI_RANDOM_HUMAN}", {}) == {}
        assert clean_katago_overrides(AI_RANDOM_HUMAN_OLD_KATAGO_DEFAULTS) == AI_RANDOM_HUMAN_OLD_KATAGO_DEFAULTS
        assert AI_RANDOM_HUMAN in AI_STRATEGIES and AI_RANDOM_HUMAN in AI_STRATEGIES_RECOMMENDED_ORDER
        assert ai_rank_estimation(AI_RANDOM_HUMAN, settings) == -12  # 13k

    def test_old_katago_defaults_dropped_from_existing_configs(self):
        katrain = KaTrainBase(force_package_config=True, debug_level=0)
        section = katrain._config.setdefault(KATAGO_SETTINGS_CONFIG_SECTION, {})
        section[AI_RANDOM_HUMAN] = dict(AI_RANDOM_HUMAN_OLD_KATAGO_DEFAULTS)  # a config from PR 6 or 7
        katrain._add_missing_ai_settings()
        assert katrain.config(f"{KATAGO_SETTINGS_CONFIG_SECTION}/{AI_RANDOM_HUMAN}") == {}
        own = {"winLossUtilityFactor": 0.7, "playoutDoublingAdvantage": 0.6, "cpuctExploration": 2.0}
        katrain._config[KATAGO_SETTINGS_CONFIG_SECTION][AI_RANDOM_HUMAN] = dict(own)  # changed by the user: kept
        katrain._add_missing_ai_settings()
        assert katrain.config(f"{KATAGO_SETTINGS_CONFIG_SECTION}/{AI_RANDOM_HUMAN}") == own
        katrain._config[KATAGO_SETTINGS_CONFIG_SECTION] = "corrupt"
        katrain._add_missing_ai_settings()
        assert katrain.config(KATAGO_SETTINGS_CONFIG_SECTION) == {}

    def test_personality_ranges(self, monkeypatch):
        assert set(AI_RANDOM_HUMAN_RANGES) | {"human_kyu_rank", "win_loss_utility"} == HIDDEN | ROLLED_KATAGO
        assert AI_RANDOM_HUMAN_RANGES == {
            "attach_penalty": (-2.0, 2.0),
            "dynamic_score_utility": (0.1, 1.0),
            "opponent_fac": (-1.0, 1.0),
            "settled_weight": (-0.3, 1.0),
            "tenuki_penalty": (-3.0, 3.0),
            "playout_doubling_advantage": (-0.5, 0.5),
        }
        assert AI_RANDOM_HUMAN_ZERO_CHANCE == 0.5 and AI_RANDOM_HUMAN_NEVER_ZERO == {"dynamic_score_utility"}
        assert AI_RANDOM_HUMAN_UTILITY_TOTAL == 1.3 and AI_RANDOM_HUMAN_MAX_WIN_LOSS_UTILITY == 1.0
        rolls = [random_human_personality(fake_game(), "B") for _ in range(2000)]
        for p in rolls:  # rolled first, then win/loss utility follows it, to one decimal place, at most KataGo's 1
            assert list(p) == list(AI_RANDOM_HUMAN_RANGES) + ["win_loss_utility"]
            assert p["win_loss_utility"] == min(1.0, round(1.3 - p["dynamic_score_utility"], 1))
            search = {
                "winLossUtilityFactor": p["win_loss_utility"],
                "playoutDoublingAdvantage": p["playout_doubling_advantage"],
            }
            assert clean_katago_overrides(search) == search  # within what KataGo accepts
        assert {p["win_loss_utility"] for p in rolls} == {x / 10 for x in range(3, 11)}
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
        assert all_zero < 200  # each option gets its own 50% chance, not one chance for all of them (~80 expected)
        for chance, pick, expected in [(0.49, min, "zero"), (0.5, min, "lo"), (0.99, max, "hi")]:
            monkeypatch.setattr(random, "random", lambda chance=chance: chance)
            monkeypatch.setattr(random, "randint", lambda a, b, pick=pick: pick(a, b))
            personality = random_human_personality(fake_game(), "B")
            win_loss = personality.pop("win_loss_utility")
            assert win_loss == (0.3 if expected == "hi" else 1.0)  # dynamic score utility 1.0, or 0.1 and capped
            for k, v in personality.items():
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
        assert search["playoutDoublingAdvantage"] == strategy.personality["playout_doubling_advantage"]
        assert search["winLossUtilityFactor"] == strategy.personality["win_loss_utility"]
        assert search["winLossUtilityFactor"] == min(1.0, round(1.3 - search["dynamicScoreUtilityFactor"], 1))

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
            "playout_doubling_advantage": -0.5,
            "win_loss_utility": 0.3,
        }
        move, thoughts = RandomRankedHumanStrategy(game, {"kyu_rank": 4.0}).generate_move()
        assert thoughts == (
            "Personality: attach penalty -2.0, dynamic score utility 1.0, opponent fac 0.5, settled weight -0.3, "
            "tenuki penalty 0.0, playout doubling advantage -0.5, win loss utility 0.3, human kyu rank 4 (rank_4k). "
            "Thoughts."
        )

    def test_katago_settings_keep_the_personality(self):
        katago = {
            "dynamicScoreUtilityFactor": 0.0,
            "humanSLProfile": "rank_9d",
            "playoutDoublingAdvantage": 1.0,
            "winLossUtilityFactor": 0.1,
            "playoutDoublingAdvantagePla": "B",
            "cpuctExploration": 2.0,
        }
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
        katago = {"cpuctExploration": 1.0}
        thoughts = {"B": set(), "W": set()}
        # Black gets the extremes KataGo must accept, White a random roll
        game.ai_personalities["B"] = {
            **{k: lo for k, (lo, hi) in AI_RANDOM_HUMAN_RANGES.items()},
            "win_loss_utility": 1.0,
        }
        try:
            for _ in range(6):
                player = game.current_node.next_player
                move, node = generate_ai_move(game, AI_RANDOM_HUMAN, settings, katago)
                katrain.log(f"Random Ranked Human -> {move}: {node.ai_thoughts}", 0)
                assert move.coords is not None
                personality = node.ai_thoughts.split(". ")[0]
                assert personality.startswith("Personality: attach penalty ")
                assert "human kyu rank 13 (rank_13k). Human-like model at rank_13k." in node.ai_thoughts
                assert "KataGo settings: cpuctExploration=1.0" in node.ai_thoughts
                assert "failed" not in node.ai_thoughts  # KataGo took every setting
                thoughts[player].add(personality)
            assert len(thoughts["B"]) == 1 and len(thoughts["W"]) == 1  # the same all game
            # rolled separately (with zeros common they can match by chance, so not compared here)
            assert game.ai_personalities["B"] is not game.ai_personalities["W"]
            assert set(game.ai_personalities) == {"B", "W"}
        finally:
            shutdown_human_model_engine(katrain)
            engine.shutdown(finish=False)
