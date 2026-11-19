"""Forecast GTA VI trophy earn rates on Exophase.

Model (own design, standard library only, deterministic):

Every trophy's earn rate is earners / players_tracked.  Both move: the
player base grows as the game sells and Exophase scans profiles, and
earners accumulate as players progress.  The program therefore forecasts
counts, not rates, and divides at the target.

Two regimes, picked from the case's history:

* Frozen history -- every read identical for hours while the game just
  launched (the practice cases).  Exophase refreshes its counts in
  bursts, so the page is stale and a refresh is pending: players tracked
  jumps by a large factor while earners grow only a little (new players
  have barely played).  The median rate is the last rate times
  earners-growth / players-jump (about 1.26 / 6), with a wide spread,
  because the jump's size is uncertain.

* Moving history -- the final case.  A per-trophy age model: players
  join over time (join mass from the players-tracked trajectory) and a
  player who joined tau hours ago has earned trophy i with probability
  S_i * F(tau / kappa_i), where F is a progression curve.  Two curve
  families are fitted per trophy (Weibull and lognormal, the second with
  the heavier late tail) and their target counts averaged, so the
  forecast does not depend on one family's shape.  (S_i, kappa_i) come
  from least squares on the case's reads.  Players tracked is
  extrapolated to the target with a power law and a saturating
  exponential, averaged in log space.  The spread combines the fit
  residual, the parameter uncertainty across near-best fits, the
  disagreement between the families, the players-tracked uncertainty
  and a horizon term.  A trophy with (near-)zero observed earners gets
  a tier prior instead of ~0: the page cannot show what nobody has
  earned yet.

Quantiles are logistic(logit(median) + z * spread), rounded to 4
decimals half-up.  A trophy at 100% in the last read stays 100%.
Any failure while forecasting one case falls back to the last-rate
forecast, so a case is never left without output.
No randomness, no clock, no network: two runs write identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- frozen-history regime (launch-night refresh) -------------------------
# Players-tracked jump at the pending refresh, and earners growth over
# the same window.  Calibrated on the launch night the rules describe:
# the page sat at 638 players, then refreshed to 3,868 (a ~6x jump),
# while earners of most trophies barely moved and early, quick trophies
# roughly doubled.  The ratio below is their combination.
FROZEN_PLAYERS_JUMP = 6.0
FROZEN_EARNERS_GROWTH = 1.26
FROZEN_SPREAD = 0.60

# --- moving-history regime -------------------------------------------------
# Weibull shape of the per-trophy progression curve.
WEIBULL_SHAPE = 1.5
# Lognormal spread (in log hours) of the second progression family.
LOGNORM_SIGMA = 0.8
# Timescales (hours) tried for a trophy's progression curve.
KAPPA_GRID = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0,
              144.0, 233.0, 377.0, 610.0, 1000.0, 1600.0]
# Timescales (hours) tried for the saturating players-tracked curve.
TAU_P_GRID = [12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0]
# Offsets (hours) tried for the power-law players-tracked curve.
POWER_OFFSETS = [0.2, 1.0, 4.0, 12.0]
# Prior day-7 rate (percent) for a trophy with (near-)zero observed
# earners: the page cannot show what nobody has earned yet, so the
# model falls back to a tier prior instead of forecasting ~0.
TIER_FLOOR = {"platinum": 0.01, "gold": 0.02, "silver": 0.04,
              "bronze": 0.06}
# The age model's spread is honest but a little wide on simulated
# finals (coverage near 1.0); scale it down, keeping the floors.
SPREAD_SCALE = 0.80


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


def least_squares(xs, ys):
    """Slope and intercept of y ~ a + b x; returns (b, a)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def weibull_cdf(tau: float, kappa: float) -> float:
    """Share of players who earned the trophy within tau hours of joining."""
    if tau <= 0.0:
        return 0.0
    return 1.0 - math.exp(-((tau / kappa) ** WEIBULL_SHAPE))


def lognorm_cdf(tau: float, kappa: float) -> float:
    """Same share under a lognormal progression curve (heavier late tail)."""
    if tau <= 0.0:
        return 0.0
    z = (math.log(tau) - math.log(kappa)) / LOGNORM_SIGMA
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def hourly_bins(reads):
    """Collapse reads to one per hour (the last read in each hour bin).

    The trajectory is smooth on this scale; hourly bins keep the age
    model's kernel cheap without losing the shape.
    """
    bins = {}
    for t, p, rates in reads:
        bins[int(math.floor(t))] = (t, p, rates)
    out = [bins[k] for k in sorted(bins)]
    return out


# ---------------------------------------------------------------------------
# Case preparation
# ---------------------------------------------------------------------------

def prepare(case: dict):
    """Return (release, checkpoint, target_hours, reads).

    reads: list of (t_hours_since_release, players, rates) with distinct
    (players, rates) only, oldest first; rates keep None entries.
    """
    game = case.get("game") or {}
    release = None
    if game.get("release_date"):
        try:
            release = datetime.fromisoformat(game["release_date"]).replace(
                tzinfo=timezone.utc)
        except ValueError:
            release = None
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    first_at = parse_time(history[0]["at"])
    if release is None:
        release = first_at
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    t_check = hours_between(release, checkpoint)
    t_target = hours_between(release, target)
    reads = []
    seen = set()
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key in seen:
            continue
        seen.add(key)
        reads.append((hours_between(release, parse_time(h["at"])),
                      float(h["players_tracked"]),
                      list(h["rates"])))
    return release, t_check, t_target, reads


def last_rates(case: dict):
    history = case["history"]
    return list(history[-1]["rates"]), float(history[-1]["players_tracked"])


# ---------------------------------------------------------------------------
# Frozen-history regime
# ---------------------------------------------------------------------------

def frozen_forecast(case: dict, t_check: float, horizon_min: float):
    """One pending refresh: players jump, earners creep, rates fall."""
    rates, _players = last_rates(case)
    jump = FROZEN_PLAYERS_JUMP if horizon_min >= 60 else 2.0
    ratio = FROZEN_EARNERS_GROWTH / jump
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = rates[i]
        if r is None:
            r = 0.0
        if r >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        median = clamp(r * ratio)
        lm = logit(median)
        rows.append({"id": trophy["id"],
                     "q": [round4(logistic(lm + z * FROZEN_SPREAD))
                           for z in Z]})
    return {"trophies": rows}


# ---------------------------------------------------------------------------
# Moving-history regime: players-tracked extrapolation
# ---------------------------------------------------------------------------

def extrapolate_players(reads, t_check: float, t_target: float):
    """Return (players_at_target, log-spread).

    Two curves fitted on distinct reads, averaged in log space: a power
    law in hours since the first read (its offset chosen by fit), and a
    saturating exponential.
    """
    t0 = reads[0][0]
    # Power law: try several offsets, keep the one fitting best.
    best_power = None
    for off in POWER_OFFSETS:
        xs = []
        ys = []
        for t, p, _rates in reads:
            tau = t - t0 + off
            if tau >= 1.5 and p > 0:
                xs.append(math.log(tau))
                ys.append(math.log(p))
        if len(xs) < 3:
            continue
        b, a = least_squares(xs, ys)
        b = min(2.0, max(0.05, b))
        sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        if best_power is None or sse < best_power[0]:
            best_power = (sse, a, b, off, xs, ys)
    if best_power is None:
        # Too little movement: hold the last count, wide spread.
        return reads[-1][1], 0.9
    _sse, a, b, off, xs, ys = best_power
    tau_t = t_target - t0 + off
    p_power = math.exp(a + b * math.log(max(tau_t, 1.0)))

    # Saturating exponential: P = Psat * (1 - exp(-tau / tp)).
    best = None
    for tp in TAU_P_GRID:
        num = 0.0
        den = 0.0
        for t, p, _r in reads:
            tau = t - t0 + off
            w = 1.0 - math.exp(-tau / tp)
            num += w * p
            den += w * w
        if den <= 0:
            continue
        psat = num / den
        sse = 0.0
        for t, p, _r in reads:
            tau = t - t0 + off
            w = 1.0 - math.exp(-tau / tp)
            sse += (p - psat * w) ** 2
        if best is None or sse < best[0]:
            best = (sse, psat, tp)
    _sse, psat, tp = best
    tau_t = t_target - t0 + off
    p_sat = psat * (1.0 - math.exp(-tau_t / tp))

    p_med = math.exp(0.5 * (math.log(max(p_power, 1.0))
                            + math.log(max(p_sat, 1.0))))
    # Spread from the two curves disagreeing, plus fit noise.
    resid = 0.0
    for x, y in zip(xs, ys):
        resid += (y - (a + b * x)) ** 2
    resid = math.sqrt(resid / len(xs))
    spread = 0.5 * abs(math.log(max(p_power, 1.0) / max(p_sat, 1.0)))
    spread = math.sqrt(spread ** 2 + resid ** 2)
    spread = min(0.9, max(0.15, spread))
    # Never forecast fewer players than the last read.
    p_med = max(p_med, reads[-1][1])
    return p_med, spread


# ---------------------------------------------------------------------------
# Moving-history regime: per-trophy age model
# ---------------------------------------------------------------------------

def join_buckets(reads, t_target: float, p_target: float):
    """(join_time, mass) buckets: observed joins, then extrapolated joins.

    Observed joins are the players-tracked steps between hourly reads;
    future joins follow the extrapolated players curve on an hourly grid
    in log space, landing exactly on the target count.
    """
    buckets = []
    prev_p = 0.0
    prev_t = None
    for t, p, _rates in reads:
        mass = p - prev_p
        if mass > 0:
            buckets.append((t, mass))
        prev_p = p
        prev_t = t
    t = prev_t
    p = prev_p
    log_p0 = math.log(max(prev_p, 1.0))
    log_pT = math.log(max(p_target, 1.0))
    span = max(t_target - prev_t, 1e-9)
    n_steps = int(math.ceil(t_target - t))
    for k in range(1, n_steps + 1):
        t_next = t_target if k == n_steps else t + 1.0
        w = (t_next - prev_t) / span
        p_next = p_target if k == n_steps else math.exp(
            log_p0 + w * (log_pT - log_p0))
        mass = p_next - p
        if mass > 0:
            buckets.append((0.5 * (t + t_next), mass))
        t, p = t_next, p_next
    return buckets


def kernel(buckets, kappa: float, times, family: str):
    """K_kappa(t) for each t: players-times-progression summed over joins."""
    cdf = weibull_cdf if family == "weibull" else lognorm_cdf
    out = []
    for t in times:
        total = 0.0
        for u, mass in buckets:
            age = t - u
            if age > 0:
                total += mass * cdf(age, kappa)
        out.append(total)
    return out


def fit_family(buckets, kappa_grid, times, k_target, obs, family):
    """Best (sse, kappa, S) and near-best fits for one family."""
    best = None
    fits = []
    for kappa in kappa_grid:
        k = kernel(buckets, kappa, times, family)
        num = 0.0
        den = 0.0
        for c, kv in zip(obs, k):
            if c is None or kv <= 0:
                continue
            num += kv * c
            den += kv * kv
        if den <= 0:
            continue
        s_est = num / den
        sse = 0.0
        cnt = 0
        for c, kv in zip(obs, k):
            if c is None:
                continue
            sse += (c - s_est * kv) ** 2
            cnt += 1
        if cnt == 0:
            continue
        fits.append((sse, kappa, s_est))
        if best is None or sse < best[0]:
            best = (sse, kappa, s_est)
    return best, fits


def age_forecast(case: dict, t_check: float, t_target: float, reads):
    rates_last, p_last = last_rates(case)
    horizon_days = max((t_target - t_check) / 24.0, 0.0)
    bins = hourly_bins(reads)
    p_target, p_spread = extrapolate_players(bins, t_check, t_target)
    buckets = join_buckets(bins, t_target, p_target)
    times = [t for t, _p, _r in bins]

    # Kernels per family and kappa, shared across trophies.
    kernels = {}
    k_target = {}
    for family in ("weibull", "lognorm"):
        for kappa in KAPPA_GRID:
            kernels[(family, kappa)] = kernel(buckets, kappa, times, family)
            k_target[(family, kappa)] = kernel(buckets, kappa, [t_target],
                                               family)[0]

    s_horizon = 0.22 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r_last = rates_last[i]
        if r_last is not None and r_last >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        obs = []
        for (_t, p, rates) in bins:
            r = rates[i]
            obs.append(None if r is None else r * p / 100.0)
        c_last = (r_last * p_last / 100.0) if r_last is not None else 0.0
        # Fit both progression families; average their target counts.
        preds = {}
        bests = {}
        for family in ("weibull", "lognorm"):
            best, fits = fit_family(buckets, KAPPA_GRID, times, k_target,
                                    obs, family)
            bests[family] = (best, fits)
            if best is None:
                preds[family] = None
                continue
            _sse, kappa_best, s_best = best
            preds[family] = max(s_best * k_target[(family, kappa_best)], 0.0)
        if preds["weibull"] is None and preds["lognorm"] is None:
            # No usable history for this trophy: persist, wide spread.
            med = clamp(r_last if r_last is not None else 0.0)
            spread = 1.2
            rows.append({"id": trophy["id"],
                         "q": [round4(logistic(logit(med) + z * spread))
                               for z in Z]})
            continue
        avail = [v for v in preds.values() if v is not None]
        if len(avail) == 2:
            c_med = math.sqrt(max(preds["weibull"], 0.0)
                              * max(preds["lognorm"], 0.0))
            s_family = 0.5 * abs(math.log(
                max(preds["weibull"], 0.05) / max(preds["lognorm"], 0.05)))
        else:
            c_med = avail[0]
            s_family = 0.35
        # Parameter uncertainty: near-best fits of the better family.
        best_fam = min(
            (f for f in ("weibull", "lognorm") if bests[f][0] is not None),
            key=lambda f: bests[f][0][0])
        sse_best, kappa_best, s_best = bests[best_fam][0]
        near = []
        for sse, kappa, s_est in bests[best_fam][1]:
            if sse <= 1.5 * sse_best + 1.0:
                near.append(max(s_est * k_target[(best_fam, kappa)], 0.0))
        if len(near) >= 2 and c_med > 0:
            logs = [math.log(max(p, 0.05)) for p in near]
            mean_log = sum(logs) / len(logs)
            var = sum((x - mean_log) ** 2 for x in logs) / len(logs)
            s_param = min(0.9, max(0.1, math.sqrt(var)))
        else:
            s_param = 0.25
        # Fit residual on the log-count scale (better family).
        sq = 0.0
        cnt = 0
        for c, kv in zip(obs, kernels[(best_fam, kappa_best)]):
            if c is None or kv <= 0:
                continue
            pred = s_best * kv
            sq += (math.log(max(c, 0.5)) - math.log(max(pred, 0.5))) ** 2
            cnt += 1
        s_fit = min(1.0, max(0.2, math.sqrt(sq / cnt))) if cnt else 0.6
        # A trophy nobody has earned yet: fall back to a tier prior
        # instead of forecasting ~0; its day-7 rate is genuinely unknown.
        spread_floor = 0.35
        median = clamp(100.0 * c_med / max(p_target, 1.0))
        if c_last < 5.0:
            floor = TIER_FLOOR.get(trophy.get("tier"), 0.1)
            median = max(median, floor)
            spread_floor = 1.1
        # Near saturation the eventual share is uncertain: widen up.
        s_sat = 0.5 * (median / 100.0) ** 2
        spread = math.sqrt(s_fit ** 2 + s_param ** 2 + s_family ** 2
                           + p_spread ** 2 + s_horizon ** 2 + s_sat ** 2
                           + 0.15 ** 2)
        spread = min(1.6, max(spread_floor, SPREAD_SCALE * spread))
        lm = logit(median)
        rows.append({"id": trophy["id"],
                     "q": [round4(logistic(lm + z * spread)) for z in Z]})
    return {"trophies": rows}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def forecast_case(case: dict) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError("case asks for levels %r, not %r"
                         % (case["levels"], LEVELS))
    if not case["history"]:
        raise ValueError("case has no history")
    _release, t_check, t_target, reads = prepare(case)
    horizon_min = float(case["horizon_minutes"])
    history = case["history"]
    span_h = hours_between(parse_time(history[0]["at"]),
                           parse_time(history[-1]["at"]))
    frozen = (len(reads) == 1 and span_h >= 1.0 and t_check < 36.0)
    if frozen:
        return frozen_forecast(case, t_check, horizon_min)
    return age_forecast(case, t_check, t_target, reads)


def persistence_fallback(case: dict) -> dict:
    """Last-rate forecast with the starter's spread: never crash a case."""
    rates, _p = last_rates(case)
    horizon_days = max(float(case.get("horizon_minutes", 0)) / 1440.0, 0.0)
    spread = 0.5 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = rates[i]
        if r is None:
            r = 0.0
        if r >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        lm = logit(r)
        rows.append({"id": trophy["id"],
                     "q": [round4(logistic(lm + z * spread)) for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            try:
                forecast = persistence_fallback(case)
            except Exception:
                forecast = {"trophies": [
                    {"id": t["id"], "q": [1, 1, 1, 1, 1]}
                    for t in case.get("trophies", [])]}
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
