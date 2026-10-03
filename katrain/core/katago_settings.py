"""Catalog of the KataGo search parameters an AI opponent can override.

Every entry is a parameter the analysis engine accepts per query (as `maxVisits` on the query itself, or in
`overrideSettings`), so each opponent can search with its own settings without restarting KataGo. Parameters that
are fixed when KataGo starts (threads per analysis, GPU and cache settings, the eval cache), that only apply to GTP
play (time management, pondering, chosenMoveTemperature and the humanSLChosenMove settings, resignation), or that KaTrain itself depends on
(reportAnalysisWinratesAs, rules) are deliberately left out.

Overrides are stored per opponent in the `ai_katago` config section as {strategy: {param: value}}. A parameter that
is missing from that dict uses KaTrain's engine setting or KataGo's own default.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

KATAGO_SETTINGS_CONFIG_SECTION = "ai_katago"

# Parameters sent on the query itself rather than in overrideSettings.
QUERY_LEVEL_PARAMS = {"maxVisits"}

AUTO_TRUE_FALSE = ("auto", "true", "false")
PLAYER_CHOICES = ("B", "W")


@dataclass(frozen=True)
class KataGoParam:
    name: str
    kind: str  # "int", "float", "bool", "choice" or "str"
    default: Any  # KataGo's own default, for display. None when KaTrain's engine settings decide it.
    group: str
    description: str
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Sequence[str] = ()

    def parse(self, raw: Any) -> Any:
        """Convert a value from the UI or config into what KataGo expects. Raises ValueError when invalid."""
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            if text in ("true", "1", "yes"):
                return True
            if text in ("false", "0", "no"):
                return False
            raise ValueError(f"{self.name} must be true or false")
        if self.kind == "choice":
            text = str(raw).strip()
            if text not in self.choices:
                raise ValueError(f"{self.name} must be one of {', '.join(self.choices)}")
            return text
        if self.kind == "str":
            text = str(raw).strip()
            if not text:
                raise ValueError(f"{self.name} must not be empty")
            return text
        if isinstance(raw, bool):
            raise ValueError(f"{self.name} must be a number")
        value = float(raw)
        if not math.isfinite(value):  # NaN is not valid JSON, and would leave the query unanswered
            raise ValueError(f"{self.name} must be a finite number")
        if self.kind == "int":
            if value != int(value):
                raise ValueError(f"{self.name} must be a whole number")
            value = int(value)
        if (self.min is not None and value < self.min) or (self.max is not None and value > self.max):
            raise ValueError(f"{self.name} must be between {self.min:g} and {self.max:g}")
        return value

    def range_text(self) -> str:
        if self.kind in ("bool", "choice"):
            return " / ".join(("true", "false") if self.kind == "bool" else self.choices)
        if self.kind == "str":
            return "text"
        return f"{self.min:g} or more" if self.max >= MAX_INT else f"{self.min:g} to {self.max:g}"

    def default_text(self) -> str:
        if self.default is None:
            return ""
        if isinstance(self.default, bool):
            return str(self.default).lower()
        if isinstance(self.default, float):
            return f"{self.default:g}"
        return str(self.default)


MAX_INT = float(1 << 50)

KATAGO_PARAMS: List[KataGoParam] = [
    # Search limits
    KataGoParam("maxVisits", "int", None, "Search limits",
                "Visits per move. Unset uses the engine's max visits setting.", 1, MAX_INT),
    KataGoParam("maxPlayouts", "int", None, "Search limits",
                "Playouts per move, counting only new playouts and not reused tree. Unlimited when unset.", 1, MAX_INT),
    KataGoParam("maxTime", "float", None, "Search limits",
                "Seconds per move. Unset uses the engine's max time setting.", 0, 1.0e20),
    KataGoParam("futileVisitsThreshold", "float", None, "Search limits",
                "Stop searching moves that cannot catch up to this fraction of the best move's visits. Off when unset.",
                0.01, 1.0),
    KataGoParam("numSearchThreads", "int", None, "Search limits",
                "Threads searching this position. Unset uses the analysis config.", 1, 4096),
    KataGoParam("minPlayoutsPerThread", "float", 8.0, "Search limits",
                "Use fewer threads on small searches so each thread gets at least this many playouts.", 0, 1.0e20),
    KataGoParam("numVirtualLossesPerThread", "float", 1.0, "Search limits",
                "Virtual losses each thread adds to keep threads from searching the same node.", 0.01, 1000),
    # Utility: what the AI tries to maximise
    KataGoParam("winLossUtilityFactor", "float", 1.0, "Utility",
                "How much the AI values winning the game.", 0, 1),
    KataGoParam("staticScoreUtilityFactor", "float", 0.1, "Utility",
                "How much the AI values score, centred on zero.", 0, 1),
    KataGoParam("dynamicScoreUtilityFactor", "float", 0.3, "Utility",
                "How much the AI values score, centred on the expected score.", 0, 1),
    KataGoParam("dynamicScoreCenterZeroWeight", "float", 0.2, "Utility",
                "How far the dynamic score centre is pulled back towards zero.", 0, 1),
    KataGoParam("dynamicScoreCenterScale", "float", 0.75, "Utility",
                "Scale of the dynamic score utility curve.", 0.2, 5),
    KataGoParam("noResultUtilityForWhite", "float", 0.0, "Utility",
                "Utility of a no-result game for White.", -1, 1),
    KataGoParam("drawEquivalentWinsForWhite", "float", 0.5, "Utility",
                "How many wins a draw is worth to White.", 0, 1),
    # Exploration
    KataGoParam("cpuctExploration", "float", 1.0, "Exploration",
                "How strongly the search explores moves the policy likes.", 0, 10),
    KataGoParam("cpuctExplorationLog", "float", 0.45, "Exploration",
                "Extra exploration that grows with the number of visits.", 0, 10),
    KataGoParam("cpuctExplorationBase", "float", 500.0, "Exploration",
                "Visit count at which the extra exploration starts to matter.", 10, 100000),
    KataGoParam("cpuctUtilityStdevPrior", "float", 0.4, "Exploration",
                "Prior for the spread of utilities, used to scale exploration.", 1e-8, 10),
    KataGoParam("cpuctUtilityStdevPriorWeight", "float", 2.0, "Exploration",
                "Weight of that prior.", 0, 100),
    KataGoParam("cpuctUtilityStdevScale", "float", 0.85, "Exploration",
                "How much the observed spread of utilities scales exploration.", 0, 1),
    KataGoParam("fpuReductionMax", "float", 0.2, "Exploration",
                "How pessimistically unvisited moves are valued below the root.", 0, 2),
    KataGoParam("fpuLossProp", "float", 0.0, "Exploration",
                "Blend unvisited moves' value towards a loss.", 0, 1),
    KataGoParam("fpuParentWeightByVisitedPolicy", "bool", True, "Exploration",
                "Weight the parent's value for unvisited moves by how much policy is already visited."),
    KataGoParam("fpuParentWeightByVisitedPolicyPow", "float", 2.0, "Exploration",
                "Exponent for that weighting.", 0, 5),
    KataGoParam("fpuParentWeight", "float", 0.0, "Exploration",
                "Fixed weight of the parent's value for unvisited moves. Only used when fpuParentWeightByVisitedPolicy is false.",
                0, 1),
    KataGoParam("policyOptimism", "float", 1.0, "Exploration",
                "Blend in the optimistic policy head below the root.", 0, 1),
    KataGoParam("nnPolicyTemperature", "float", 1.0, "Exploration",
                "Temperature applied to the neural net policy everywhere in the search.", 0.01, 5),
    # Value estimation
    KataGoParam("valueWeightExponent", "float", 0.25, "Value estimation",
                "How much more reliable evaluations are weighted when averaging.", 0, 1),
    KataGoParam("useNoisePruning", "bool", True, "Value estimation",
                "Down-weight noisy, low-value visits when averaging."),
    KataGoParam("noisePruneUtilityScale", "float", 0.15, "Value estimation",
                "Utility scale used by noise pruning.", 0.001, 10),
    KataGoParam("noisePruningCap", "float", 1e50, "Value estimation",
                "Maximum weight noise pruning can remove.", 0, 1e50),
    KataGoParam("useUncertainty", "bool", True, "Value estimation",
                "Weight evaluations by the network's own uncertainty."),
    KataGoParam("uncertaintyCoeff", "float", 0.25, "Value estimation",
                "Strength of uncertainty weighting.", 0.0001, 1),
    KataGoParam("uncertaintyExponent", "float", 1.0, "Value estimation",
                "Exponent of uncertainty weighting.", 0, 2),
    KataGoParam("uncertaintyMaxWeight", "float", 8.0, "Value estimation",
                "Maximum weight a confident evaluation can get.", 1, 100),
    KataGoParam("subtreeValueBiasFactor", "float", 0.45, "Value estimation",
                "Correct evaluations using errors seen in similar local shapes.", 0, 1),
    KataGoParam("subtreeValueBiasFreeProp", "float", 0.8, "Value estimation",
                "Proportion of that correction that is applied freely.", 0, 1),
    KataGoParam("subtreeValueBiasWeightExponent", "float", 0.85, "Value estimation",
                "Exponent for weighting that correction by visits.", 0, 1),
    KataGoParam("useGraphSearch", "bool", True, "Value estimation",
                "Share search results between transpositions."),
    KataGoParam("graphSearchRepBound", "int", 11, "Value estimation",
                "Repetition bound used to make graph search safe.", 3, 50),
    KataGoParam("graphSearchCatchUpLeakProb", "float", 0.0, "Value estimation",
                "Probability of letting a transposition catch up on visits.", 0, 1),
    # Root of the search
    KataGoParam("wideRootNoise", "float", None, "Root",
                "Spread visits over more moves at the root. Unset uses the engine's wide root noise setting.", 0, 5),
    KataGoParam("rootPolicyTemperature", "float", 1.0, "Root",
                "Temperature of the policy at the root. Above 1 explores more varied moves.", 0.01, 100),
    KataGoParam("rootPolicyTemperatureEarly", "float", 1.0, "Root",
                "Root policy temperature early in the game.", 0.01, 100),
    KataGoParam("rootFpuReductionMax", "float", 0.1, "Root",
                "How pessimistically unvisited moves are valued at the root.", 0, 2),
    KataGoParam("rootFpuLossProp", "float", 0.0, "Root",
                "Blend unvisited root moves' value towards a loss.", 0, 1),
    KataGoParam("rootNoiseEnabled", "bool", False, "Root",
                "Add Dirichlet noise to the root policy, as in self-play training."),
    KataGoParam("rootDirichletNoiseTotalConcentration", "float", 10.83, "Root",
                "Concentration of the root Dirichlet noise.", 0.001, 10000),
    KataGoParam("rootDirichletNoiseWeight", "float", 0.25, "Root",
                "Weight of the root Dirichlet noise.", 0, 1),
    KataGoParam("rootNumSymmetriesToSample", "int", 1, "Root",
                "Board symmetries averaged for the root policy. More gives a slightly better policy.", 1, 8),
    KataGoParam("rootSymmetryPruning", "bool", True, "Root",
                "Skip moves that are symmetric duplicates of other moves."),
    KataGoParam("rootDesiredPerChildVisitsCoeff", "float", 0.0, "Root",
                "Force a minimum number of visits to each root move.", 0, 100),
    KataGoParam("rootPolicyOptimism", "float", 0.2, "Root",
                "Blend in the optimistic policy head at the root.", 0, 1),
    KataGoParam("rootEndingBonusPoints", "float", 0.5, "Root",
                "Bonus for moves that help end the game cleanly.", -1, 1),
    KataGoParam("rootPruneUselessMoves", "bool", True, "Root",
                "Prune pointless moves at the end of the game."),
    # Move selection order
    KataGoParam("useLcbForSelection", "bool", True, "Move selection",
                "Rank moves by lower confidence bound instead of visits."),
    KataGoParam("lcbStdevs", "float", 5.0, "Move selection",
                "Standard deviations used for the lower confidence bound.", 1, 12),
    KataGoParam("minVisitPropForLCB", "float", 0.15, "Move selection",
                "Minimum share of visits a move needs before its LCB counts.", 0, 1),
    KataGoParam("useNonBuggyLcb", "bool", True, "Move selection",
                "Use the corrected LCB calculation."),
    # Passing and rules handling
    KataGoParam("conservativePass", "bool", True, "Passing and rules",
                "Never assume a pass ends the game."),
    KataGoParam("fillDameBeforePass", "bool", True, "Passing and rules",
                "Fill neutral points before passing."),
    KataGoParam("enablePassingHacks", "bool", True, "Passing and rules",
                "Extra search on passing to avoid passing too early or too late."),
    KataGoParam("enableMorePassingHacks", "bool", True, "Passing and rules",
                "More of those passing safeguards."),
    KataGoParam("alwaysComputePassAliveUnderSuicideRules", "choice", "auto", "Passing and rules",
                "Compute pass-alive areas even when suicide is legal.", choices=AUTO_TRUE_FALSE),
    KataGoParam("excludeTerritoryAdjacentToAtari", "choice", "auto", "Passing and rules",
                "Do not count points next to stones in atari as territory.", choices=AUTO_TRUE_FALSE),
    # Modelling the opponent
    KataGoParam("playoutDoublingAdvantage", "float", 0.0, "Opponent modelling",
                "Assume this side searches 2^x times more than its opponent. Positive plays more aggressively.",
                -3, 3),
    KataGoParam("playoutDoublingAdvantagePla", "choice", None, "Opponent modelling",
                "Which side the playout doubling advantage is for. Unset means the side to move.",
                choices=PLAYER_CHOICES),
    KataGoParam("avoidRepeatedPatternUtility", "float", 0.0, "Opponent modelling",
                "Penalty for repeating the same local patterns.", -3, 3),
    KataGoParam("antiMirror", "bool", False, "Opponent modelling",
                "Detect and punish an opponent who mirrors every move."),
    # History
    KataGoParam("ignorePreRootHistory", "bool", True, "History",
                "Ignore the moves that led to this position when evaluating it."),
    KataGoParam("ignoreAllHistory", "bool", False, "History",
                "Ignore all move history, including within the search."),
    # Human-like play (needs the human-like model set in the engine settings)
    KataGoParam("humanSLProfile", "str", None, "Human-like model",
                "Human profile such as rank_5k, preaz_2d or proyear_1990. Needs the human-like model."),
    KataGoParam("humanSLCpuctExploration", "float", 1.0, "Human-like model",
                "Exploration towards moves the human policy likes.", 0, 1000),
    KataGoParam("humanSLCpuctPermanent", "float", 0.0, "Human-like model",
                "Exploration towards human moves that does not fade with visits.", 0, 1000),
    KataGoParam("humanSLRootExploreProbWeightless", "float", 0.0, "Human-like model",
                "Share of root visits spent exploring human moves, without affecting values.", 0, 1),
    KataGoParam("humanSLRootExploreProbWeightful", "float", 0.0, "Human-like model",
                "Share of root visits spent exploring human moves, affecting values.", 0, 1),
    KataGoParam("humanSLPlaExploreProbWeightless", "float", 0.0, "Human-like model",
                "Same for the AI's own later moves, without affecting values.", 0, 1),
    KataGoParam("humanSLPlaExploreProbWeightful", "float", 0.0, "Human-like model",
                "Same for the AI's own later moves, affecting values.", 0, 1),
    KataGoParam("humanSLOppExploreProbWeightless", "float", 0.0, "Human-like model",
                "Same for the opponent's moves, without affecting values.", 0, 1),
    KataGoParam("humanSLOppExploreProbWeightful", "float", 0.0, "Human-like model",
                "Same for the opponent's moves, affecting values. Models a human opponent.", 0, 1),
]  # fmt: skip

KATAGO_PARAMS_BY_NAME: Dict[str, KataGoParam] = {p.name: p for p in KATAGO_PARAMS}


def katago_param_groups() -> List[Tuple[str, List[KataGoParam]]]:
    groups: Dict[str, List[KataGoParam]] = {}
    for param in KATAGO_PARAMS:
        groups.setdefault(param.group, []).append(param)
    return list(groups.items())


def clean_katago_overrides(overrides: Optional[Dict], log=None) -> Dict[str, Any]:
    """Drop unknown or invalid entries from stored overrides, so a hand-edited config cannot break the AI."""
    cleaned = {}
    if not isinstance(overrides, dict):
        if overrides and log:
            log(f"Ignoring KataGo settings {overrides!r}: not a dictionary")
        return cleaned
    for name, raw in overrides.items():
        param = KATAGO_PARAMS_BY_NAME.get(name)
        if param is None:
            if log:
                log(f"Ignoring unknown KataGo setting {name}")
            continue
        try:
            cleaned[name] = param.parse(raw)
        except (TypeError, ValueError) as e:
            if log:
                log(f"Ignoring KataGo setting {name}={raw!r}: {e}")
    return cleaned


def split_katago_overrides(overrides: Dict[str, Any]) -> Tuple[Optional[int], Dict[str, Any]]:
    """Split overrides into the query's visits and the overrideSettings dict."""
    settings = {k: v for k, v in overrides.items() if k not in QUERY_LEVEL_PARAMS}
    # KataGo only reads these when the setting they depend on is off or on, and warns about them otherwise.
    if settings.get("fpuParentWeightByVisitedPolicy", True):
        settings.pop("fpuParentWeight", None)
    else:
        settings.pop("fpuParentWeightByVisitedPolicyPow", None)
    return overrides.get("maxVisits"), settings
