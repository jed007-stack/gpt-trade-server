# main.py — GPT decision server (actions only; EA manages SL/TP)
from __future__ import annotations
from typing import Any, Dict, Optional, Literal, Tuple
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, time, logging

# ---------- OpenAI ----------
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # gpt-4o / gpt-4o-mini etc.
OAI = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Server knobs ----------
SYMBOL_ALLOW   = set(filter(None, (os.getenv("SYMBOL_ALLOW", "XAUUSD,XAUUSD.M,XAUUSD_I").split(","))))
HARD_STOP_R    = float(os.getenv("HARD_STOP_R_MULT", "1.30"))   # close if r_now <= -1.30
MIN_ADX_SHORT  = float(os.getenv("MIN_ADX_SHORT", "24.0"))      # shorts need stronger ADX
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO").upper()

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
app = FastAPI(title="GM GPT Trade Server (Actions Only)", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

# ---------- Schemas ----------
class DecisionOut(BaseModel):
    action: Literal["open_long","open_short","close","hold"]
    reason: str
    # Optional extras (EA can use them for gating)
    align: Optional[bool] = None
    confidence: Optional[float] = None

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
        "r_now": getf(p, "r_now", 0.0),
        "unrealized": getf(p, "unrealized", 0.0),
        "partial_done": getb(p, "partial_done", False),
    }

def guards_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    g = payload.get("guards") or {}
    # Back-compat fallbacks if EA ever sends flat keys:
    if not g:
        g = {
            "spread_ok": getb(payload, "max_spread_pips_ok", True),
            "dd_locked": getb(payload, "daily_dd_locked", False),
            "active_window": True,
            "news_block": False,
        }
    return {
        "spread_ok": getb(g, "spread_ok", True),
        "dd_locked": getb(g, "dd_locked", False),
        "active_window": getb(g, "active_window", True),
        "news_block": getb(g, "news_block", False),
    }

def constraints_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    c = payload.get("constraints") or {}
    return {
        "require_align": getb(c, "require_align", True),
        "min_confidence": float(c.get("min_confidence", 0.65)),
    }

def readiness_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = payload.get("readiness") or {}
    # Back-compat fallbacks from older flat payloads:
    if not r:
        r = {
            "pullback_long": getb(payload, "pullback_long_ready", False),
            "breakout_long": getb(payload, "breakout_long_ready", False),
            "orb_long":      getb(payload, "orb_long_ready", False),
            "avwap_long":    getb(payload, "avwap_long_ready", False),
            "pullback_short": getb(payload, "pullback_short_ready", False),
            "breakout_short": getb(payload, "breakout_short_ready", False),
            "orb_short":      getb(payload, "orb_short_ready", False),
        }
    return r

def hard_guards(payload: Dict[str, Any]) -> Optional[DecisionOut]:
    sym = (payload.get("symbol") or "").upper()
    if SYMBOL_ALLOW and all(s.upper() != sym for s in SYMBOL_ALLOW):
        return DecisionOut(action="hold", reason=f"Symbol not allowed: {sym}", align=False, confidence=0.0)

    g = guards_from(payload)
    if not g["spread_ok"]:
        return DecisionOut(action="hold", reason="Spread filter (EA)", align=False, confidence=0.0)

    if g["dd_locked"]:
        pos = position_from(payload)
        if pos["has"]:
            return DecisionOut(action="close", reason="Daily DD lock (EA) — closing", align=True, confidence=1.0)
        return DecisionOut(action="hold", reason="Daily DD lock (EA) — flat", align=True, confidence=1.0)

    # Simple news block: never open during window (EA also guards)
    if g["news_block"]:
        pos = position_from(payload)
        if not pos["has"]:
            return DecisionOut(action="hold", reason="News window — no new trades", align=False, confidence=0.0)

    return None

def last_resort_management(payload: Dict[str, Any]) -> Optional[DecisionOut]:
    pos = position_from(payload)
    if pos["has"] and pos["r_now"] <= -HARD_STOP_R:
        return DecisionOut(action="close", reason=f"Hard stop: r_now={pos['r_now']:.2f}", align=True, confidence=1.0)
    return None

def safe_parse_json(txt: str) -> Optional[dict]:
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(txt.replace(",}", "}").replace(",]", "]"))
        except Exception:
            return None

# ---------- Alignment & confidence ----------
def compute_alignment_and_confidence(payload: Dict[str, Any], proposed: str) -> Tuple[bool, float, str]:
    """Return (align_ok, confidence_0_1, note) using readiness + regime/ADX."""
    r = readiness_from(payload)
    g = guards_from(payload)
    allow_longs  = getb(payload, "allow_longs", True)
    allow_shorts = getb(payload, "allow_shorts", False)

    adx = getf(payload, "adx", 0.0)
    bbw = getf(payload, "bb_width", 0.0)
    ema_sep = getf(payload, "ema_sep_pct", 0.0)
    regime = (payload.get("regime") or "").lower()

    # Base readiness votes
    long_vote  = any([getb(r, "pullback_long"), getb(r, "breakout_long"), getb(r, "orb_long"), getb(r, "avwap_long")])
    short_vote = any([getb(r, "pullback_short"), getb(r, "breakout_short"), getb(r, "orb_short")])

    # Confidence signals (normalized-ish)
    adx_score = min(1.0, max(0.0, (adx - 10.0) / 20.0))         # ~0 @10, ~1 @30
    bbw_score = min(1.0, max(0.0, (bbw - 0.02) / 0.06))         # stronger expansion => higher
    ema_score = min(1.0, max(0.0, abs(ema_sep) / 0.25))         # 0–0.25%+
    base_conf = 0.25*adx_score + 0.25*bbw_score + 0.25*ema_score + 0.25*(1.0 if regime in ("trend","breakout") else 0.0)

    # Regime bias: shorts need stronger environment
    if proposed == "open_short":
        base_conf *= 0.9
        if adx < MIN_ADX_SHORT:
            return (False, 0.0, f"Weak ADX for short ({adx:.1f}<{MIN_ADX_SHORT})")

    # Check hard environment constraints
    if not g["active_window"] or g["news_block"] or not g["spread_ok"] or g["dd_locked"]:
        return (False, 0.0, "Environment block (session/news/spread/dd)")

    # Side permissions
    if proposed == "open_long" and not allow_longs:
        return (False, 0.0, "Longs not allowed")
    if proposed == "open_short" and not allow_shorts:
        return (False, 0.0, "Shorts not allowed")

    # Alignment by readiness votes
    if proposed == "open_long" and not long_vote:
        return (False, base_conf*0.6, "No long setup ready")
    if proposed == "open_short" and not short_vote:
        return (False, base_conf*0.6, "No short setup ready")

    # Position-state alignment: if already in trade, we never open (EA also blocks)
    pos = position_from(payload)
    if pos["has"] and proposed in ("open_long","open_short"):
        return (False, 0.0, "Position already open")

    # Close/hold always aligned, derive mild confidence
    if proposed in ("close","hold"):
        return (True, 0.8 if proposed=="close" else 0.6, "Non-open action")

    # If we reached here, aligned open
    return (True, max(0.0, min(1.0, base_conf)), "Aligned with readiness & regime")

# ---------- GPT calls ----------
SYSTEM_PROMPT = (
    "You are a disciplined, risk-aware trading decision engine for XAUUSD.\n"
    "Return STRICT JSON ONLY (no prose). Keys:\n"
    '  action: one of "open_long", "open_short", "close", "hold"\n'
    "  reason: concise explanation referencing key evidence used\n"
    "Guidance:\n"
    "- Respect EA guards (spread/dd/news/session) — if blocked, hold.\n"
    "- Shorts require stronger trend/ADX than longs.\n"
    "- If a position exists, prefer hold/close (do not suggest SL/TP changes).\n"
    "- If no edge: action='hold'.\n"
    "Output only the JSON object."
)

def build_user_prompt(payload: Dict[str, Any]) -> str:
    # Compact user content aligned to new EA payload
    candles_txt = "null"
    if isinstance(payload.get("candles"), list) and payload["candles"]:
        try:
            last = payload["candles"][-8:]
            slim = [{"t": c.get("t"), "o": c.get("o"), "h": c.get("h"), "l": c.get("l"), "c": c.get("c")} for c in last]
            candles_txt = json.dumps(slim, separators=(",", ":"))
        except Exception:
            candles_txt = "null"

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
        "readiness": readiness_from(payload),
        "market": payload.get("market") or {"bid": None, "ask": None, "atr": None},
        "swings": {"high": getf(payload, "swing_high", 0.0), "low": getf(payload, "swing_low", 0.0)},
        "position": payload.get("position"),
        "today_pl": payload.get("today_pl"),
        "sentiment": senti,
        "guards": guards_from(payload),
        "constraints": constraints_from(payload),
        "allow_longs": getb(payload, "allow_longs", True),
        "allow_shorts": getb(payload, "allow_shorts", False),
        "candles_tail": candles_txt,
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
        raw = await req.body()
        log.warning(f"Non-JSON body received: {raw[:200]!r}")
        return DecisionOut(action="hold", reason="Bad request body", align=False, confidence=0.0)

    log.debug(f"payload: {json.dumps(payload)[:1500]}")

    # 1) Hard guards (spread/DD/news/symbol)
    hg = hard_guards(payload)
    if hg is not None:
        return hg

    # 2) Emergency risk guard if already in a trade (does not touch SL/TP)
    lr = last_resort_management(payload)
    if lr is not None:
        return lr

    # 3) GPT proposes
    g = call_gpt(payload)
    if not isinstance(g, dict):
        return DecisionOut(action="hold", reason="Model returned no/invalid JSON.", align=False, confidence=0.0)

    proposed = str(g.get("action", "hold")).strip().lower()
    if proposed not in ("open_long","open_short","close","hold"):
        proposed = "hold"

    # 4) Alignment + confidence
    align_ok, conf, note = compute_alignment_and_confidence(payload, proposed)

    # Honor EA-declared constraints (it will also gate on its side)
    cons = constraints_from(payload)
    if cons["require_align"] and not align_ok:
        return DecisionOut(action="hold", reason=f"Alignment veto: {note}", align=False, confidence=conf)
    if conf < cons["min_confidence"] and proposed in ("open_long","open_short"):
        return DecisionOut(action="hold", reason=f"Confidence {conf:.2f} < min {cons['min_confidence']:.2f}", align=align_ok, confidence=conf)

    # 5) Extra safety: shorts need ADX strength
    if proposed == "open_short":
        adx = getf(payload, "adx", 0.0)
        if adx < MIN_ADX_SHORT:
            return DecisionOut(action="hold", reason=f"Veto weak short (ADX {adx:.1f}<{MIN_ADX_SHORT})", align=False, confidence=conf)

    # Pass through with GPT's reason (if any)
    reason = str(g.get("reason", "")).strip() or note
    return DecisionOut(action=proposed, reason=reason, align=align_ok, confidence=conf)

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
    return {"ok": True, "msg": "GM GPT Trade Server — Actions only (EA owns SL/TP)."}
