# main.py — GPT decision server for XAUUSD_Pro_GM_ORB_AVWAP_GPT.mq5
#
# Inputs:  EA JSON payload (see BuildPayloadJSON in your EA)
# Output:  {"action": "open_long|open_short|modify|close|hold",
#           "reason": "...",
#           "orders": {"sl": float|null, "tp": float|null, "close_partial_volume": float|null}}
#
# GPT is the decision maker. Local hard-guards still apply (spread/DD lock/-R kill switch).
# Sentiment: uses payload.sentiment plus server-side aggregations you can set via endpoints.

from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os, json, time, math

# ------------- OpenAI -------------
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # e.g., gpt-4o, gpt-4o-mini, gpt-4.1
OAI = OpenAI(api_key=OPENAI_API_KEY)

# ------------- Server knobs -------------
SYMBOL_ALLOW = set(os.getenv("SYMBOL_ALLOW", "XAUUSD,XAUUSD.m,XAUUSD_i").split(","))
REQ_TIMEOUT_MS = int(os.getenv("REQ_TIMEOUT_MS", "5000"))

# Prop/guard rails
HARD_STOP_R_MULT = float(os.getenv("HARD_STOP_R_MULT", "1.30"))   # close if r_now <= -1.30
MIN_ADX_SHORT    = float(os.getenv("MIN_ADX_SHORT", "24.0"))      # GPT will see ADX, but we hard-veto weak shorts

# ------------- Sentiment store (optional external sources) -------------
# You can push external signals (news, risk-on/off, DXY/inverse, gold ETF flows) here via API endpoints below.
SENTIMENT = {
    "bull": 0.5,       # normalized 0..1 (bullish)
    "bear": 0.5,       # normalized 0..1 (bearish)
    "components": {},  # store raw inputs for transparency
    "ts": time.time(),
}

def combined_sentiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Blend EA-provided sentiment and server-side sentiment into one normalized view."""
    p_sent = payload.get("sentiment") or {}
    bull_p = float(p_sent.get("bull", 0.5))
    bear_p = float(p_sent.get("bear", 0.5))
    tot_p = bull_p + bear_p
    bull_p = (bull_p / tot_p) if tot_p > 0 else 0.5

    bull_s = SENTIMENT["bull"]
    # weighted average: payload gets 60%, server store 40%
    bull = 0.6 * bull_p + 0.4 * bull_s
    bull = max(0.0, min(1.0, bull))
    bear = 1.0 - bull
    return {
        "bull": bull,
        "bear": bear,
        "details": {
            "payload_bull": bull_p,
            "server_bull": bull_s,
            "components": SENTIMENT.get("components", {}),
        }
    }

# ------------- API + schema -------------
app = FastAPI(title="GM GPT Trade Server", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

class OrdersOut(BaseModel):
    sl: Optional[float] = None
    tp: Optional[float] = None
    close_partial_volume: Optional[float] = None

class DecisionOut(BaseModel):
    action: str = Field(..., regex=r"^(open_long|open_short|modify|close|hold)$")
    reason: str
    orders: OrdersOut = OrdersOut()

# ------------- Helpers -------------
def getf(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    try: return float(d.get(k, default))
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
        "age_minutes": int(getf(p, "age_minutes", 0.0)),
        "partial_done": getb(p, "partial_done", False),
    }

def hard_guards(payload: Dict[str, Any]) -> Optional[DecisionOut]:
    # allowlist
    sym = (payload.get("symbol") or "").upper()
    if SYMBOL_ALLOW and all(s.upper() != sym for s in SYMBOL_ALLOW):
        return DecisionOut(action="hold", reason=f"Symbol not allowed: {sym}", orders=OrdersOut())

    # EA-side risk rails we always honor
    if not getb(payload, "max_spread_pips_ok", True):
        return DecisionOut(action="hold", reason="Spread filter (EA)", orders=OrdersOut())

    if getb(payload, "daily_dd_locked", False):
        pos = position_from(payload)
        if pos["has"]:
            return DecisionOut(action="close", reason="Daily DD lock (EA) — closing", orders=OrdersOut())
        return DecisionOut(action="hold", reason="Daily DD lock (EA) — flat", orders=OrdersOut())

    return None

def last_resort_management(payload: Dict[str, Any]) -> Optional[DecisionOut]:
    """If position open, we still enforce a last-resort hard stop by R."""
    pos = position_from(payload)
    if pos["has"] and pos["r_now"] <= -HARD_STOP_R_MULT:
        return DecisionOut(action="close", reason=f"Hard stop: r_now={pos['r_now']:.2f}", orders=OrdersOut())
    return None

def safe_parse_json(txt: str) -> Optional[dict]:
    try:
        return json.loads(txt)
    except Exception:
        try:
            # remove trailing commas
            repaired = json.loads(txt.replace(",}", "}").replace(",]", "]"))
            return repaired
        except Exception:
            return None

# ------------- GPT call -------------
SYSTEM_PROMPT = (
    "You are a disciplined, risk-aware trading decision engine for XAUUSD.\n"
    "Return STRICT JSON ONLY (no prose). Keys:\n"
    "action: one of open_long, open_short, modify, close, hold\n"
    "reason: concise explanation referencing key evidence used\n"
    "orders: { sl: number|null, tp: number|null, close_partial_volume: number|null }\n"
    "Rules:\n"
    "- Respect the provided risk context: do not suggest trades that violate an obvious EA guard (e.g., massive spread, DD lock). "
    "Those will be filtered anyway, but your reasoning should be consistent.\n"
    "- Shorts require stronger trend/ADX evidence than longs.\n"
    "- Prefer SL based on swing/ATR context in the payload; TP ≥ 1.8R by default unless a nearer strong S/R or AB=CD completion is present.\n"
    "- If a position is already open, prefer 'modify' or 'hold' unless a close is clearly justified.\n"
    "- If no edge: action='hold'.\n"
    "Output nothing but the JSON object."
)

def build_user_prompt(payload: Dict[str, Any]) -> str:
    # Condense candles to a tiny summary to keep token load sane
    candles_txt = "null"
    if isinstance(payload.get("candles"), list) and payload["candles"]:
        try:
            last = payload["candles"][-5:]
            # keep only t/o/h/l/c
            slim = [{"t": c.get("t"), "o": c.get("o"), "h": c.get("h"), "l": c.get("l"), "c": c.get("c")} for c in last]
            candles_txt = json.dumps(slim)
        except Exception:
            candles_txt = "null"

    waves = payload.get("waves") or {}
    senti = combined_sentiment(payload)

    compact = {
        "symbol": payload.get("symbol"),
        "time": payload.get("server_time"),
        "session": payload.get("session"),
        "regime": payload.get("regime"),
        "bias_dir": payload.get("bias_dir"),
        "adx": payload.get("adx"),
        "bb_width": payload.get("bb_width"),
        "ema_sep_pct": payload.get("ema_sep_pct"),
        "lrc_angle_deg": payload.get("lrc_angle_deg"),
        "readiness": {
            "pullback_long": payload.get("pullback_long_ready"),
            "breakout_long": payload.get("breakout_long_ready"),
            "orb_long": payload.get("orb_long_ready"),
            "avwap_long": payload.get("avwap_long_ready"),
            "pullback_short": payload.get("pullback_short_ready"),
            "breakout_short": payload.get("breakout_short_ready"),
            "orb_short": payload.get("orb_short_ready"),
        },
        "prices": {"bid": payload.get("bid"), "ask": payload.get("ask")},
        "atr": payload.get("atr"),
        "swings": {"high": payload.get("swing_high"), "low": payload.get("swing_low")},
        "proposed": {
            "long": {"sl": payload.get("long_proposed_sl"),  "tp": payload.get("long_proposed_tp")},
            "short":{"sl": payload.get("short_proposed_sl"), "tp": payload.get("short_proposed_tp")}
        },
        "position": payload.get("position"),
        "results": payload.get("today_pl"),
        "waves": {
            "valid": (waves.get("valid_abcd") if isinstance(waves.get("valid_abcd"), bool) else False),
            "pattern": waves.get("pattern"),
            "bias": waves.get("bias"),
            "completion_price": waves.get("completion_price"),
            "confidence": waves.get("confidence"),
        },
        "sentiment": senti,  # blended sentiment (payload + server)
        "candles_tail": candles_txt,
        "guards_echo": {
            "spread_ok": payload.get("max_spread_pips_ok", True),
            "dd_locked": payload.get("daily_dd_locked", False)
        }
    }
    return json.dumps(compact, separators=(",", ":"))  # dense JSON user message

def call_gpt(payload: Dict[str, Any]) -> Optional[dict]:
    user_content = build_user_prompt(payload)
    rsp = OAI.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.15,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )
    txt = (rsp.choices[0].message.content or "").strip()
    return safe_parse_json(txt)

# ------------- Endpoint -------------
@app.post("/gpt/manage", response_model=DecisionOut)
async def gpt_manage(req: Request):
    payload: Dict[str, Any] = await req.json()

    # 1) Hard guards (symbol/spread/DD lock)
    hg = hard_guards(payload)
    if hg is not None:
        return hg

    # 2) If a position is open, we still enforce last-resort -R close locally
    lr = last_resort_management(payload)
    if lr is not None:
        return lr

    # 3) Let GPT decide
    g = call_gpt(payload)
    if not isinstance(g, dict):
        return DecisionOut(action="hold", reason="Model returned no/invalid JSON.", orders=OrdersOut())

    # Normalize GPT output into the contract
    action = str(g.get("action", "hold")).strip().lower()
    if action not in ("open_long","open_short","modify","close","hold"):
        action = "hold"

    orders_in = g.get("orders") or {}
    orders = OrdersOut(
        sl = orders_in.get("sl"),
        tp = orders_in.get("tp"),
        close_partial_volume = orders_in.get("close_partial_volume"),
    )

    # 4) Final safety checks (shorts w/ weak ADX veto)
    if action == "open_short":
        adx = getf(payload, "adx", 0.0)
        if adx < MIN_ADX_SHORT:
            return DecisionOut(action="hold", reason=f"Veto weak short (ADX {adx:.1f}<{MIN_ADX_SHORT})", orders=OrdersOut())

    # 5) Return GPT’s decision
    return DecisionOut(action=action, reason=str(g.get("reason","")).strip(), orders=orders)

# ------------- Sentiment endpoints -------------
@app.get("/sentiment/set")
async def sentiment_set(bull: float, bear: float):
    bull = max(0.0, min(1.0, float(bull)))
    bear = max(0.0, min(1.0, float(bear)))
    if bull == 0 and bear == 0:
        bull, bear = 0.5, 0.5
    SENTIMENT.update({"bull": bull, "bear": bear, "ts": time.time()})
    return {"ok": True, "sentiment": SENTIMENT}

@app.post("/sentiment/ingest")
async def sentiment_ingest(component: str, score: float):
    """
    POST arbitrary sub-sentiment signals you compute elsewhere (0..1 bull).
    We'll keep a simple average with existing components and update the headline score.
    body (JSON): { "component": "news_risk", "score": 0.62 }
    """
    score = max(0.0, min(1.0, float(score)))
    comp = SENTIMENT.setdefault("components", {})
    comp[component] = {"score": score, "ts": time.time()}
    # headline = average of components (if any) blended 70% with existing headline
    if comp:
        avg = sum(v["score"] for v in comp.values()) / len(comp)
        new_bull = 0.7 * SENTIMENT["bull"] + 0.3 * avg
        SENTIMENT["bull"] = max(0.0, min(1.0, new_bull))
        SENTIMENT["bear"] = 1.0 - SENTIMENT["bull"]
        SENTIMENT["ts"] = time.time()
    return {"ok": True, "sentiment": SENTIMENT}

@app.get("/")
async def root():
    return {"ok": True, "msg": "GM GPT Trade Server — GPT makes decisions w/ sentiment, waves, and tech context (prop-guarded)."}
