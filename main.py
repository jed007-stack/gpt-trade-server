# main.py — GPT decision server for XAUUSD_Pro_GM_ORB_AVWAP_GPT.mq5
from __future__ import annotations
from typing import Any, Dict, Optional, Literal
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, json, time, logging

# ---------- OpenAI ----------
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # e.g. gpt-4o, gpt-4o-mini
OAI = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Server knobs ----------
SYMBOL_ALLOW = set(filter(None, (os.getenv("SYMBOL_ALLOW", "XAUUSD,XAUUSD.m,XAUUSD_i").split(","))))
HARD_STOP_R_MULT = float(os.getenv("HARD_STOP_R_MULT", "1.30"))   # close if r_now <= -1.30
MIN_ADX_SHORT    = float(os.getenv("MIN_ADX_SHORT", "24.0"))      # shorts need stronger ADX
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gpt-trade-server")

# ---------- Sentiment store (simple rolling blend) ----------
SENTIMENT = {
    "bull": 0.5,
    "bear": 0.5,
    "components": {},   # {name: {score, ts}}
    "ts": time.time(),
}

def combined_sentiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    p_sent = payload.get("sentiment") or {}
    bull_p = float(p_sent.get("bull", 0.5))
    bear_p = float(p_sent.get("bear", 0.5))
    tot_p = bull_p + bear_p
    bull_p = (bull_p / tot_p) if tot_p > 0 else 0.5

    bull_s = float(SENTIMENT.get("bull", 0.5))
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

# ---------- FastAPI ----------
app = FastAPI(title="GM GPT Trade Server", version="2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

# ---------- Schemas ----------
class OrdersOut(BaseModel):
    sl: Optional[float] = None
    tp: Optional[float] = None
    close_partial_volume: Optional[float] = None

class DecisionOut(BaseModel):
    action: Literal["open_long","open_short","modify","close","hold"]
    reason: str
    orders: OrdersOut = OrdersOut()

# ---------- Helpers ----------
def getf(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    try: return float(d.get(k, default))
    except Exception: return default

def getb(d: Dict[str, Any], k: str, default: bool = False) -> bool:
    v = d.get(k, default)
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return v != 0
    if isinstance(v, str): return v.strip().lower() in ("true","1","yes","y","t")
    return bool(v)

def position_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = payload.get("position") or {}
    return {
        "has": getb(p, "has", False),
        "dir": (p.get("dir") or "").lower() if isinstance(p.get("dir"), str) else None,
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
    sym = (payload.get("symbol") or "").upper()
    if SYMBOL_ALLOW and all(s.upper() != sym for s in SYMBOL_ALLOW):
        return DecisionOut(action="hold", reason=f"Symbol not allowed: {sym}", orders=OrdersOut())

    if not getb(payload, "max_spread_pips_ok", True):
        return DecisionOut(action="hold", reason="Spread filter (EA)", orders=OrdersOut())

    if getb(payload, "daily_dd_locked", False):
        pos = position_from(payload)
        if pos["has"]:
            return DecisionOut(action="close", reason="Daily DD lock (EA) — closing", orders=OrdersOut())
        return DecisionOut(action="hold", reason="Daily DD lock (EA) — flat", orders=OrdersOut())
    return None

def last_resort_management(payload: Dict[str, Any]) -> Optional[DecisionOut]:
    pos = position_from(payload)
    if pos["has"] and pos["r_now"] <= -HARD_STOP_R_MULT:
        return DecisionOut(action="close", reason=f"Hard stop: r_now={pos['r_now']:.2f}", orders=OrdersOut())
    return None

def safe_parse_json(txt: str) -> Optional[dict]:
    # tolerate trailing commas if they ever sneak in
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(txt.replace(",}", "}").replace(",]", "]"))
        except Exception:
            return None

# ---------- GPT calls ----------
SYSTEM_PROMPT = (
    "You are a disciplined, risk-aware trading decision engine for XAUUSD.\n"
    "Return STRICT JSON ONLY (no prose). Keys:\n"
    "  action: one of open_long, open_short, modify, close, hold\n"
    "  reason: concise explanation referencing key evidence used\n"
    "  orders: { sl: number|null, tp: number|null, close_partial_volume: number|null }\n"
    "Rules:\n"
    "- Respect provided guards; be consistent with them in your rationale.\n"
    "- Shorts require stronger trend/ADX than longs.\n"
    "- Prefer SL via swing/ATR; TP ≥ 1.8R unless strong S/R or AB=CD completion closer.\n"
    "- If a position is open, prefer modify/hold unless close is clearly justified.\n"
    "- If no edge: action='hold'.\n"
    "Output nothing but the JSON object."
)

def build_user_prompt(payload: Dict[str, Any]) -> str:
    # down-sample candles for token economy
    candles_txt = "null"
    if isinstance(payload.get("candles"), list) and payload["candles"]:
        try:
            last = payload["candles"][-8:]  # up to 8 most recent for context
            slim = [{"t": c.get("t"), "o": c.get("o"), "h": c.get("h"), "l": c.get("l"), "c": c.get("c")} for c in last]
            candles_txt = json.dumps(slim, separators=(",", ":"))
        except Exception:
            candles_txt = "null"

    waves = payload.get("waves") or {}
    senti = combined_sentiment(payload)

    compact = {
        "symbol": payload.get("symbol"),
        "server_time": payload.get("server_time"),
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
            "valid": bool(waves.get("valid_abcd")) if waves.get("valid_abcd") is not None else False,
            "pattern": waves.get("pattern"),
            "bias": waves.get("bias"),
            "completion_price": waves.get("completion_price"),
            "confidence": waves.get("confidence"),
        },
        "sentiment": senti,
        "candles_tail": candles_txt,
        "guards_echo": {
            "spread_ok": payload.get("max_spread_pips_ok", True),
            "dd_locked": payload.get("daily_dd_locked", False)
        }
    }
    return json.dumps(compact, separators=(",", ":"))

def call_gpt(payload: Dict[str, Any]) -> Optional[dict]:
    try:
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
        log.debug(f"GPT raw: {txt[:1000]}")
        return safe_parse_json(txt)
    except Exception as e:
        log.error(f"OpenAI error: {e}")
        return None

# ---------- Endpoint ----------
@app.post("/gpt/manage", response_model=DecisionOut)
async def gpt_manage(req: Request):
    try:
        payload: Dict[str, Any] = await req.json()
    except Exception:
        # EA always sends JSON, but be defensive
        raw = await req.body()
        log.warning(f"Non-JSON body received: {raw[:200]!r}")
        return DecisionOut(action="hold", reason="Bad request body", orders=OrdersOut())

    log.debug(f"payload: {json.dumps(payload)[:1200]}")

    # 1) Hard guards (spread/DD lock/symbol)
    hg = hard_guards(payload)
    if hg is not None:
        return hg

    # 2) Emergency risk guard if already in a trade
    lr = last_resort_management(payload)
    if lr is not None:
        return lr

    # 3) GPT decides
    g = call_gpt(payload)
    if not isinstance(g, dict):
        return DecisionOut(action="hold", reason="Model returned no/invalid JSON.", orders=OrdersOut())

    action_in = str(g.get("action", "hold")).strip().lower()
    action: Literal["open_long","open_short","modify","close","hold"]
    if action_in in ("open_long","open_short","modify","close","hold"):
        action = action_in  # type: ignore
    else:
        action = "hold"     # type: ignore

    orders_in = g.get("orders") or {}
    orders = OrdersOut(
        sl = orders_in.get("sl"),
        tp = orders_in.get("tp"),
        close_partial_volume = orders_in.get("close_partial_volume"),
    )

    # 4) Final safety: shorts need ADX strength
    if action == "open_short":
        adx = getf(payload, "adx", 0.0)
        if adx < MIN_ADX_SHORT:
            return DecisionOut(action="hold", reason=f"Veto weak short (ADX {adx:.1f}<{MIN_ADX_SHORT})", orders=OrdersOut())

    # 5) Done
    return DecisionOut(action=action, reason=str(g.get("reason","")).strip(), orders=orders)

# ---------- Sentiment endpoints ----------
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
    score = max(0.0, min(1.0, float(score)))
    comp = SENTIMENT.setdefault("components", {})
    comp[component] = {"score": score, "ts": time.time()}
    if comp:
        avg = sum(v["score"] for v in comp.values()) / len(comp)
        new_bull = 0.7 * float(SENTIMENT["bull"]) + 0.3 * avg
        SENTIMENT["bull"] = max(0.0, min(1.0, new_bull))
        SENTIMENT["bear"] = 1.0 - SENTIMENT["bull"]
        SENTIMENT["ts"] = time.time()
    return {"ok": True, "sentiment": SENTIMENT}

@app.get("/")
async def root():
    return {"ok": True, "msg": "GM GPT Trade Server — GPT makes decisions w/ sentiment, waves, and tech context (prop-guarded)."}
