# main.py — GPT webhook for XAUUSD_Pro_GM_ORB_AVWAP_GPT.mq5
# FastAPI server that ingests EA payload and returns: open_long/open_short/modify/close/hold
# Decision = tech readiness + AB=CD waves + optional sentiment, with hard risk guards.

from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
import os, time, math

# ---------------- Env / knobs ----------------
SYMBOL_ALLOW = set(os.getenv("SYMBOL_ALLOW", "XAUUSD,XAUUSD.m,XAUUSD_i").split(","))

# weights (0..1). They are normalized implicitly by thresholds below.
SENTI_WEIGHT = float(os.getenv("SENTI_WEIGHT", "0.35"))
WAVE_WEIGHT  = float(os.getenv("WAVE_WEIGHT",  "0.40"))
TECH_WEIGHT  = float(os.getenv("TECH_WEIGHT",  "0.25"))

# thresholds
OPEN_TREND_OR_BREAKOUT = float(os.getenv("OPEN_TREND_OR_BREAKOUT", "0.55"))
OPEN_RANGE             = float(os.getenv("OPEN_RANGE",             "0.70"))
OPEN_INDECISION        = float(os.getenv("OPEN_INDECISION",        "0.75"))
MIN_ADX_SHORT          = float(os.getenv("MIN_ADX_SHORT",          "24.0"))
HARD_STOP_R_MULT       = float(os.getenv("HARD_STOP_R_MULT",       "1.30"))  # close if r_now <= -1.3R
TRAIL_START_R          = float(os.getenv("TRAIL_START_R",          "1.40"))  # mirrors EA

# simple in-memory sentiment (optional)
GLOBAL_SENTIMENT = {"bull": 0.5, "bear": 0.5, "ts": time.time()}

# ---------------- API scaffolding -------------
app = FastAPI(title="GM GPT Trade Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

# ---------------- Outgoing schema -------------
class OrdersOut(BaseModel):
    sl: Optional[float] = None
    tp: Optional[float] = None
    close_partial_volume: Optional[float] = None

class DecisionOut(BaseModel):
    action: str = Field(..., regex=r"^(open_long|open_short|modify|close|hold)$")
    reason: str
    orders: OrdersOut = OrdersOut()

# ---------------- Helpers --------------------
def getf(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    try: return float(d.get(k, default))
    except Exception: return default

def geti(d: Dict[str, Any], k: str, default: int = 0) -> int:
    try: return int(d.get(k, default))
    except Exception: return default

def getb(d: Dict[str, Any], k: str, default: bool = False) -> bool:
    v = d.get(k, default)
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return v != 0
    if isinstance(v, str): return v.lower() in ("true","1","yes","y")
    return bool(v)

def position_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = payload.get("position") or {}
    return {
        "has": getb(p, "has", False),
        "dir": p.get("dir"),
        "volume": getf(p, "volume", 0.0),
        "entry": getf(p, "entry", 0.0),
        "sl": getf(p, "sl", 0.0),
        "tp": getf(p, "tp", 0.0),
        "r_now": getf(p, "r_now", 0.0),
        "unrealized": getf(p, "unrealized", 0.0),
        "age_minutes": geti(p, "age_minutes", 0),
        "partial_done": getb(p, "partial_done", False),
    }

def sentiment_score(payload: Dict[str, Any]) -> float:
    s = payload.get("sentiment") or {}
    bull = float(s.get("bull", GLOBAL_SENTIMENT["bull"]))
    bear = float(s.get("bear", GLOBAL_SENTIMENT["bear"]))
    tot  = bull + bear
    if tot <= 0: return 0.5
    return max(0.0, min(1.0, bull / tot))

def waves_bias_conf(payload: Dict[str, Any]) -> tuple[str, float]:
    w = payload.get("waves")
    if not isinstance(w, dict): return "neutral", 0.0
    bias = str(w.get("bias", "neutral")).lower()
    conf = float(w.get("confidence", 0.0))
    return bias, max(0.0, min(1.0, conf))

def tech_readiness(payload: Dict[str, Any]) -> Dict[str, bool]:
    # booleans emitted by EA
    return {
        "pull_long":  getb(payload, "pullback_long_ready"),
        "brk_long":   getb(payload, "breakout_long_ready"),
        "orb_long":   getb(payload, "orb_long_ready"),
        "avwap_long": getb(payload, "avwap_long_ready"),
        "pull_short": getb(payload, "pullback_short_ready"),
        "brk_short":  getb(payload, "breakout_short_ready"),
        "orb_short":  getb(payload, "orb_short_ready"),
    }

def combine_scores(payload: Dict[str, Any]) -> Dict[str, float]:
    senti = sentiment_score(payload)           # 0..1
    wave_b, wave_c = waves_bias_conf(payload)  # "bullish"/"bearish"/"neutral", 0..1
    tech = tech_readiness(payload)

    # technical subscores
    tl = sum(1 for k in ("pull_long","brk_long","orb_long","avwap_long") if tech[k]) / 4.0
    ts = sum(1 for k in ("pull_short","brk_short","orb_short")           if tech[k]) / 3.0

    # wave directional confidence
    wave_long  = wave_c if wave_b == "bullish" else (0.0 if wave_b == "bearish" else wave_c*0.25)
    wave_short = wave_c if wave_b == "bearish" else (0.0 if wave_b == "bullish" else wave_c*0.25)

    # sentiment → centered scores
    senti_long  = max(0.0, (senti - 0.5) * 2.0)   # 0..1
    senti_short = max(0.0, (0.5 - senti) * 2.0)

    long_score  = SENTI_WEIGHT*senti_long  + WAVE_WEIGHT*wave_long  + TECH_WEIGHT*tl
    short_score = SENTI_WEIGHT*senti_short + WAVE_WEIGHT*wave_short + TECH_WEIGHT*ts

    return {"long": max(0.0, min(1.0, long_score)),
            "short": max(0.0, min(1.0, short_score))}

def entry_bias_ok(payload: Dict[str, Any], side: str) -> bool:
    # respect EA H1 bias (when it’s active on EA side)
    b = str(payload.get("bias_dir", "neutral")).lower()
    if side == "long"  and b not in ("long","neutral"):  return False
    if side == "short" and b not in ("short","neutral"): return False
    return True

def want_open(side: str, payload: Dict[str, Any], scores: Dict[str, float]) -> bool:
    regime = str(payload.get("regime", "indecision")).lower()
    score  = scores[side]

    # shorts stricter for XAU unless ADX strong
    if side == "short":
        adx = getf(payload, "adx", 0.0)
        if adx < MIN_ADX_SHORT:
            return False

    if regime in ("trend","breakout"):
        return score >= OPEN_TREND_OR_BREAKOUT and entry_bias_ok(payload, side)
    if regime == "range":
        return score >= OPEN_RANGE and entry_bias_ok(payload, side)
    return score >= OPEN_INDECISION and entry_bias_ok(payload, side)

def proposed_orders(side: str, payload: Dict[str, Any]) -> OrdersOut:
    if side == "long":
        sl = getf(payload, "long_proposed_sl", 0.0)
        tp = getf(payload, "long_proposed_tp", 0.0)
    else:
        sl = getf(payload, "short_proposed_sl", 0.0)
        tp = getf(payload, "short_proposed_tp", 0.0)
    return OrdersOut(sl=sl if sl>0 else None, tp=tp if tp>0 else None)

def hard_guards(payload: Dict[str, Any]) -> Optional[DecisionOut]:
    # symbol allowlist
    sym = (payload.get("symbol") or "").upper()
    if SYMBOL_ALLOW and all(s.upper() != sym for s in SYMBOL_ALLOW):
        return DecisionOut(action="hold", reason=f"Symbol not allowed: {sym}", orders=OrdersOut())

    # spread and DD lock from EA
    if not getb(payload, "max_spread_pips_ok", True):
        return DecisionOut(action="hold", reason="Spread filter (EA)", orders=OrdersOut())

    if getb(payload, "daily_dd_locked", False):
        pos = position_from(payload)
        if pos["has"]:
            return DecisionOut(action="close", reason="Daily DD lock (EA) — closing", orders=OrdersOut())
        return DecisionOut(action="hold", reason="Daily DD lock (EA) — flat", orders=OrdersOut())

    return None

def manage_open_position(payload: Dict[str, Any]) -> DecisionOut:
    """Light-touch management. EA already handles BE/partial/trailing."""
    pos = position_from(payload)
    # hard stop on adverse R multiple
    if pos["r_now"] <= -HARD_STOP_R_MULT:
        return DecisionOut(action="close", reason=f"Hard stop: r_now={pos['r_now']:.2f}", orders=OrdersOut())

    # optional: tighten TP near AB=CD completion once > trail start R
    waves = payload.get("waves") or {}
    completion = getf(waves, "completion_price", 0.0)
    if pos["dir"] == "long" and pos["r_now"] >= TRAIL_START_R and completion > 0:
        tp_now = pos["tp"]
        if tp_now and tp_now > completion:
            return DecisionOut(action="modify", reason="Tighten TP to AB=CD completion", orders=OrdersOut(tp=completion))
    if pos["dir"] == "short" and pos["r_now"] >= TRAIL_START_R and completion > 0:
        tp_now = pos["tp"]
        if tp_now and tp_now < completion:
            return DecisionOut(action="modify", reason="Tighten TP to AB=CD completion", orders=OrdersOut(tp=completion))

    return DecisionOut(action="hold", reason="EA managing (BE/partial/trail)", orders=OrdersOut())

# ---------------- Endpoint -------------------
@app.post("/gpt/manage", response_model=DecisionOut)
async def gpt_manage(req: Request):
    payload: Dict[str, Any] = await req.json()

    # hard guards
    hg = hard_guards(payload)
    if hg is not None:
        return hg

    # if position open, keep decisions minimal
    pos = position_from(payload)
    if pos["has"]:
        return manage_open_position(payload)

    # flat → evaluate entries
    scores = combine_scores(payload)
    long_ok  = want_open("long",  payload, scores)
    short_ok = want_open("short", payload, scores)

    if long_ok or short_ok:
        if scores["long"] >= scores["short"]:
            od = proposed_orders("long", payload)
            return DecisionOut(action="open_long",
                               reason=f"Open long (score={scores['long']:.2f})",
                               orders=od)
        else:
            od = proposed_orders("short", payload)
            return DecisionOut(action="open_short",
                               reason=f"Open short (score={scores['short']:.2f})",
                               orders=od)

    return DecisionOut(action="hold", reason="No edge (scores below thresholds / bias/regime mismatch)", orders=OrdersOut())

# ---- Simple sentiment setter for testing ----
@app.get("/sentiment/set")
async def set_sentiment(bull: float, bear: float):
    bull = max(0.0, min(1.0, float(bull)))
    bear = max(0.0, min(1.0, float(bear)))
    if bull == 0 and bear == 0:
        bull, bear = 0.5, 0.5
    GLOBAL_SENTIMENT.update({"bull": bull, "bear": bear, "ts": time.time()})
    return {"ok": True, "sentiment": GLOBAL_SENTIMENT}

@app.get("/")
async def root():
    return {"ok": True, "msg": "GM GPT Trade Server — regime/bias + tech + AB=CD + sentiment, with prop-safe guards."}
