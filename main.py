# main.py — SmartGPT EA Server (gpt-4o only, no retries) + Low-Volatility Trend Trap Filters
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError, conint, confloat
from typing import List, Optional, Dict, Any, Literal
from fastapi.responses import JSONResponse
import os
import logging
import json
import re
from datetime import datetime, time
import pytz
from openai import OpenAI

# === Setup ===
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

# Model config: primary gpt-4o ONLY (no fallback, no retries)
MODEL_ID = os.getenv("OPENAI_MODEL", "gpt-4o")

openai_client = OpenAI(api_key=api_key)

app = FastAPI()
logging.basicConfig(level=logging.INFO)

REASON_MAXLEN = 900  # clamp to avoid oversized JSON


# === Data Models (incoming) ===
class MACD(BaseModel):
    main: Optional[float] = None
    signal: Optional[float] = None

class Ichimoku(BaseModel):
    tenkan: Optional[float] = None
    kijun: Optional[float] = None
    senkou_a: Optional[float] = None
    senkou_b: Optional[float] = None

class Indicators(BaseModel):
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    stoch_j: Optional[float] = None
    macd: Optional[MACD] = None
    ema: Optional[float] = None
    ema_period: Optional[int] = None
    ema_array: Optional[List[float]] = None
    lwma: Optional[float] = None
    lwma_period: Optional[int] = None
    adx: Optional[float] = None
    mfi: Optional[float] = None
    williams_r: Optional[float] = None
    ichimoku: Optional[Ichimoku] = None
    rsi_array: Optional[List[float]] = None
    price_array: Optional[List[float]] = None
    support_resistance: Optional[Dict[str, List[float]]] = None
    fibonacci: Optional[Dict[str, Any]] = None
    candlestick_patterns: Optional[List[str]] = None
    atr: Optional[float] = None

class Candle(BaseModel):
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None

class Position(BaseModel):
    direction: Optional[str] = None
    open_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    lot: Optional[float] = None
    pnl: Optional[float] = None

class Account(BaseModel):
    balance: Optional[float] = None
    equity: Optional[float] = None
    margin: Optional[float] = None

class Timeframes(BaseModel):
    main: Optional[List[Candle]] = None
    tf2: Optional[List[Candle]] = None
    tf3: Optional[List[Candle]] = None

class Regime(BaseModel):
    label: Optional[str] = None      # "trend" | "range" | "breakout"
    bb_width: Optional[float] = None
    slope_pct: Optional[float] = None
    adx: Optional[float] = None

class TradeData(BaseModel):
    symbol: str
    timeframe: str
    update_type: Optional[str] = None       # e.g., "ema_over_lwma", "session_check_19"
    cross_signal: Optional[str] = None
    cross_meaning: Optional[str] = None

    indicators: Optional[Indicators] = None       # main TF
    tf2_indicators: Optional[Indicators] = None   # tf2
    tf3_indicators: Optional[Indicators] = None   # tf3
    timeframes: Optional[Timeframes] = None
    tf_names: Optional[Dict[str, str]] = None     # {"main":"M5","tf2":"M15","tf3":"H1"}

    position: Optional[Position] = None
    account: Optional[Account] = None

    # Session/news & recovery
    news_override: Optional[bool] = False
    last_trade_was_loss: Optional[bool] = False
    unrecovered_loss: Optional[float] = 0.0

    # Crossover signal from EA
    strong_crossover: Optional[bool] = False

    # microstructure context from EA
    regime: Optional[Regime] = None
    spread_pips: Optional[float] = None
    spread_to_sl: Optional[float] = None
    require_sr_tp: Optional[bool] = None

class TradeWrapper(BaseModel):
    data: TradeData


# === Model for outgoing decision (strict) ===
AllowedAction = Literal["buy", "sell", "hold", "close"]

class EMAContext(BaseModel):
    period: Optional[int] = None
    price_vs_ema: Optional[Literal["above","below","near","unknown"]] = "unknown"
    aligns_all_tfs: Optional[bool] = False
    slopes: Optional[Dict[str, str]] = None

class StructureSummary(BaseModel):
    sr_levels: List[float] = []
    last_reversal: Optional[str] = "none"

class FibSummary(BaseModel):
    high: float = 0.0
    low: float = 0.0
    active_levels: List[str] = []

class VolatilityContext(BaseModel):
    atr: float = 0.0
    bb_state: Optional[Literal["squeeze","expansion","neutral"]] = "neutral"

class VolumeContext(BaseModel):
    mfi: float = 0.0
    state: Optional[Literal["low","normal","high","unknown"]] = "unknown"

class ModelSelfAudit(BaseModel):
    data_used: List[str] = []
    data_ignored: List[str] = []
    confidence_rationale: str = ""

class DecisionOut(BaseModel):
    action: AllowedAction
    reason: str
    confidence: conint(ge=0, le=10) = 0
    lot: Optional[confloat(ge=0)] = None
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    categories: List[str] = []

    missing_categories: List[str] = []
    needed_to_enter: List[str] = []
    disqualifiers: List[str] = []

    session_block: bool = False
    god_mode_used: bool = False
    force_close: bool = False

    ema_context: EMAContext = EMAContext()
    structure_summary: StructureSummary = StructureSummary()
    fib_summary: FibSummary = FibSummary()
    volatility_context: VolatilityContext = VolatilityContext()
    volume_context: VolumeContext = VolumeContext()

    risk_notes: str = ""
    policy: Optional[Literal["none","session_exit","friday_exit","news_conflict"]] = "none"
    recovery_mode: bool = False

    model_self_audit: ModelSelfAudit = ModelSelfAudit()


# === Helpers ===
ALLOWED_ACTIONS = {"buy", "sell", "hold", "close"}
ALLOWED_POLICIES = {"none", "session_exit", "friday_exit", "news_conflict"}

def _coerce_dict(x, default=None):
    """Ensure nested objects are dicts (handle '', '{}', None)."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return default if default is not None else {}
        try:
            j = json.loads(s)
            return j if isinstance(j, dict) else (default if default is not None else {})
        except Exception:
            return default if default is not None else {}
    if x is None:
        return default if default is not None else {}
    return default if default is not None else {}

def _coerce_policy(p):
    p = (p or "none")
    if isinstance(p, str):
        p = p.strip().lower()
    else:
        p = "none"
    return p if p in ALLOWED_POLICIES else "none"

def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in {"1","true","t","yes","y"}
    return False

def _as_str(x):
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)

def _as_list_str(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        return [x]
    return [str(x)]

def flatten_action(decision: Any) -> Dict[str, Any]:
    # Single-pass decode; no second GPT call
    if isinstance(decision, dict) and "action" in decision:
        return decision
    if isinstance(decision, str):
        cleaned = re.sub(r"```(\w+)?", "", decision).strip()
        try:
            d = json.loads(cleaned)
            if "action" in d:
                return d
        except Exception:
            pass
    if isinstance(decision, dict) and "raw" in decision:
        cleaned = re.sub(r"```(\w+)?", "", decision["raw"]).strip()
        try:
            d = json.loads(cleaned)
            if "action" in d:
                return d
        except Exception:
            pass
    return {"action": "hold", "reason": "Could not decode action.", "confidence": 0}

def extract_json_object(s: str):
    if not isinstance(s, str):
        return None
    cleaned = re.sub(r"```(?:json|JSON)?", "", s).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = cleaned[start:end+1]
    try:
        return json.loads(chunk)
    except Exception:
        try:
            no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", chunk)
            return json.loads(no_trailing_commas)
        except Exception:
            return None

def uk_time_now():
    utc_now = datetime.utcnow()
    london = pytz.timezone('Europe/London')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(london)

def is_between_uk_time(start_h, end_h):
    now = uk_time_now().time()
    if start_h < end_h:
        return time(start_h, 0) <= now < time(end_h, 0)
    else:
        return now >= time(start_h, 0) or now < time(end_h, 0)

def is_friday_5pm_or_later():
    now = uk_time_now()
    return now.weekday() == 4 and now.time() >= time(17, 0)

def is_uk_at_or_after(hour_24):
    return uk_time_now().time() >= time(hour_24, 0)

def extract_categories(reason):
    m = re.search(r"Confluences?:\s*([^.]+)", reason or "", re.IGNORECASE)
    if not m:
        return set()
    cats = m.group(1)
    found = set()
    for cat in ["Trend", "Momentum", "Volatility", "Volume", "Structure", "ADX"]:
        if re.search(r"\b{}\b".format(cat), cats, re.IGNORECASE):
            found.add(cat.lower())
    return found

# === EMA helpers ===
def ema_trend(ind: Indicators):
    if not ind or not ind.ema_array or len(ind.ema_array) < 2:
        return 0
    prev_ema = ind.ema_array[-2]
    curr_ema = ind.ema_array[-1]
    if curr_ema > prev_ema:
        return 1
    elif curr_ema < prev_ema:
        return -1
    return 0

def ema_slope(ind: Indicators):
    if not ind or not ind.ema_array or len(ind.ema_array) < 2:
        return 0.0
    return (ind.ema_array[-1] or 0.0) - (ind.ema_array[-2] or 0.0)

def ema_slope_desc(slope, threshold=1e-6):
    if slope > threshold:
        return f"sloping up ({slope:.6f})"
    elif slope < -threshold:
        return f"sloping down ({slope:.6f})"
    else:
        return f"flat ({slope:.6f})"

def ema_confirms(ind: Indicators, action: str):
    if not ind or not ind.ema_array or not ind.price_array:
        return False
    price = ind.price_array[-1]
    ema_last = ind.ema_array[-1]
    if action == "buy":
        return (ema_last is not None and price is not None and price > ema_last and ema_trend(ind) == 1)
    if action == "sell":
        return (ema_last is not None and price is not None and price < ema_last and ema_trend(ind) == -1)
    return False

def infer_tick_tol(ind: Indicators) -> float:
    price = None
    if ind and ind.price_array:
        for v in reversed(ind.price_array):
            if v:
                price = v
                break
    if price is None:
        return 1e-4
    s = f"{price:.10f}".rstrip("0")
    decimals = len(s.split(".")[1]) if "." in s else 2
    return 10 ** (-(decimals - 1))

def atr_norm_slope(slope: float, ind: Indicators) -> float:
    atr = ind.atr if ind and ind.atr else None
    ref = atr if (atr and atr > 0) else (ind.price_array[-1] if ind and ind.price_array else 1.0)
    ref = abs(ref) if ref else 1.0
    return slope / ref

# === VOL/MOM/BB helpers for Low-Volatility Trend Trap ===
def bb_width(ind: Indicators) -> Optional[float]:
    if not ind or ind.bb_upper is None or ind.bb_lower is None or ind.bb_middle is None:
        return None
    mid = ind.bb_middle if ind.bb_middle != 0 else 1.0
    return abs(ind.bb_upper - ind.bb_lower) / abs(mid)

def classify_bb_state(ind: Indicators) -> str:
    # If regime.bb_width is supplied upstream, EA can send it; else derive from bands
    w = bb_width(ind)
    if w is None:
        return "neutral"
    # Heuristic thresholds work well M5/M15: tweak upstream if needed
    if w < 0.002:  # very narrow
        return "squeeze"
    if w > 0.005:  # clearly expanding
        return "expansion"
    return "neutral"

def macd_hist(ind: Indicators) -> Optional[float]:
    if not ind or not ind.macd or ind.macd.main is None or ind.macd.signal is None:
        return None
    return ind.macd.main - ind.macd.signal

def rsi_latest(ind: Indicators) -> Optional[float]:
    if not ind or not ind.rsi_array or len(ind.rsi_array) == 0:
        return None
    return ind.rsi_array[-1]

def rsi_prev(ind: Indicators) -> Optional[float]:
    if not ind or not ind.rsi_array or len(ind.rsi_array) < 2:
        return None
    return ind.rsi_array[-2]

def momentum_exhausted_buy(ind_main: Indicators, ind_tf2: Indicators) -> bool:
    # Overbought stoch OR RSI rolling over OR MACD hist declining (if tf2 present as proxy for "was higher")
    stoch_over = (ind_main.stoch_k or 0) >= 75
    rsi_now, rsi_was = rsi_latest(ind_main), rsi_prev(ind_main)
    rsi_roll = (rsi_now is not None and rsi_was is not None and rsi_was >= 55 and rsi_now < rsi_was)
    # Use tf2 MACD as "prior momentum" if main.hist present but cannot compare prev; otherwise skip
    hist_main = macd_hist(ind_main)
    hist_tf2  = macd_hist(ind_tf2)
    macd_fading = (hist_main is not None and hist_tf2 is not None and hist_main <= hist_tf2)
    return stoch_over or rsi_roll or macd_fading

def momentum_exhausted_sell(ind_main: Indicators, ind_tf2: Indicators) -> bool:
    stoch_over = (ind_main.stoch_k or 100) <= 25
    rsi_now, rsi_was = rsi_latest(ind_main), rsi_prev(ind_main)
    rsi_roll = (rsi_now is not None and rsi_was is not None and rsi_was <= 45 and rsi_now > rsi_was)
    hist_main = macd_hist(ind_main)
    hist_tf2  = macd_hist(ind_tf2)
    macd_fading = (hist_main is not None and hist_tf2 is not None and hist_main >= hist_tf2)
    return stoch_over or rsi_roll or macd_fading

def adx_ok_and_rising(action: str, ind_main: Indicators, ind_tf2: Indicators) -> bool:
    a_main = ind_main.adx or 0.0
    a_tf2  = ind_tf2.adx if (ind_tf2 and ind_tf2.adx is not None) else None
    # Need >=25 and (rising vs tf2 if available)
    if a_main < 25.0:
        return False
    if a_tf2 is not None and a_main <= a_tf2:
        return False
    return True

def volume_confirm_buy(ind_main: Indicators, ind_tf2: Indicators) -> bool:
    m_main = ind_main.mfi
    m_tf2  = ind_tf2.mfi if ind_tf2 else None
    if m_main is None:
        return False
    # prefer >60 and rising vs tf2 if available
    if m_main < 60.0:
        return False
    if m_tf2 is not None and m_main <= m_tf2:
        return False
    return True

def volume_confirm_sell(ind_main: Indicators, ind_tf2: Indicators) -> bool:
    m_main = ind_main.mfi
    m_tf2  = ind_tf2.mfi if ind_tf2 else None
    if m_main is None:
        return False
    if m_main > 40.0:
        return False
    if m_tf2 is not None and m_main >= m_tf2:
        return False
    return True

def bb_expanding(ind_main: Indicators, ind_tf2: Indicators) -> bool:
    w_main = bb_width(ind_main)
    w_tf2  = bb_width(ind_tf2)
    if w_main is None:
        return False
    if w_tf2 is None:
        # Fallback: treat "expansion" classification as expanding
        return classify_bb_state(ind_main) == "expansion"
    return w_main > w_tf2  # current width greater than prior timeframe's width -> expansion bias

# === Chat call (single shot, no retry, no fallback) ===
def _gpt_once(prompt: str) -> str:
    chat = openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are an elite, disciplined, risk-aware SCALPER assistant. Reply ONLY in strict JSON. Include 'new_sl' and 'new_tp' for buy/sell. Never include analysis outside the JSON fields requested. For object fields, return JSON objects (e.g., {}) not strings."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,  # gpt-4o uses max_tokens
        temperature=0.15,
        response_format={"type": "json_object"}
    )
    return chat.choices[0].message.content or ""


def call_gpt_messages(prompt: str):
    try:
        return _gpt_once(prompt)
    except Exception as e:
        logging.error(f"GPT error: {e}")
        return ""  # handled downstream as fail-safe HOLD


# === Coercion + Validation ===
def sanitize_and_validate(out_dict: Dict[str, Any], fallback_reason: str, in_recovery_mode: bool) -> DecisionOut:
    # Action clamp
    action = str(out_dict.get("action", "hold")).strip().lower()
    if action not in ALLOWED_ACTIONS:
        action = "hold"
        out_dict["reason"] = (out_dict.get("reason") or "") + " | Action invalid; coerced to HOLD."
    out_dict["action"] = action

    # Defaults
    out_dict.setdefault("reason", fallback_reason)
    out_dict.setdefault("confidence", 0)
    out_dict.setdefault("categories", [])
    out_dict.setdefault("missing_categories", [])
    out_dict.setdefault("needed_to_enter", [])
    out_dict.setdefault("disqualifiers", [])
    out_dict.setdefault("session_block", False)
    out_dict.setdefault("god_mode_used", False)
    out_dict.setdefault("force_close", False)
    out_dict.setdefault("ema_context", {})
    out_dict.setdefault("structure_summary", {})
    out_dict.setdefault("fib_summary", {})
    out_dict.setdefault("volatility_context", {})
    out_dict.setdefault("volume_context", {})
    out_dict.setdefault("risk_notes", "")
    out_dict.setdefault("policy", "none")
    out_dict.setdefault("model_self_audit", {})
    out_dict["recovery_mode"] = in_recovery_mode

    # Clamps / coercions
    try:
        out_dict["confidence"] = max(0, min(10, int(out_dict.get("confidence", 0) or 0)))
    except Exception:
        out_dict["confidence"] = 0

    for k in ("categories","missing_categories","needed_to_enter","disqualifiers"):
        out_dict[k] = _as_list_str(out_dict.get(k, []))

    for k in ("session_block","god_mode_used","force_close"):
        out_dict[k] = _as_bool(out_dict.get(k, False))

    out_dict["risk_notes"] = _as_str(out_dict.get("risk_notes",""))

    for k in ("ema_context","structure_summary","fib_summary","volatility_context","volume_context","model_self_audit"):
        out_dict[k] = _coerce_dict(out_dict.get(k), default={})

    out_dict["policy"] = _coerce_policy(out_dict.get("policy"))

    # Clamp reason to prevent massive payloads
    if isinstance(out_dict.get("reason"), str) and len(out_dict["reason"]) > REASON_MAXLEN:
        out_dict["reason"] = out_dict["reason"][:REASON_MAXLEN] + "..."

    # Final validation
    return DecisionOut(**out_dict)


# === Core endpoint ===
@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind_main = trade.indicators or Indicators()
    ind_tf2  = trade.tf2_indicators or Indicators()
    ind_tf3  = trade.tf3_indicators or Indicators()
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000, margin=None)

    candles_main = (trade.timeframes.main or [])[-5:] if trade.timeframes and trade.timeframes.main else []
    candles_tf2  = (trade.timeframes.tf2  or [])[-5:] if trade.timeframes and trade.timeframes.tf2  else []
    candles_tf3  = (trade.timeframes.tf3  or [])[-5:] if trade.timeframes and trade.timeframes.tf3  else []

    # Safety: inject price_array from candles if EA didn't send it
    def inject_from_candles(ind: Indicators, candles: list):
        if not ind:
            return
        if (not ind.price_array) and candles:
            closes = [c.close for c in candles if c and c.close is not None]
            if closes:
                ind.price_array = closes[-2:] if len(closes) >= 2 else closes

    inject_from_candles(ind_main, candles_main)
    inject_from_candles(ind_tf2,  candles_tf2)
    inject_from_candles(ind_tf3,  candles_tf3)

    cross_signal = trade.cross_signal or "none"
    cross_meaning = trade.cross_meaning or "none"

    ema_period = ind_main.ema_period or 100
    tf_label_main = (trade.tf_names or {}).get("main", "main")
    tf_label_tf2  = (trade.tf_names or {}).get("tf2",  "tf2")
    tf_label_tf3  = (trade.tf_names or {}).get("tf3",  "tf3")

    def _ema_slope(ind: Indicators):
        if not ind or not ind.ema_array or len(ind.ema_array) < 2:
            return 0.0
        return (ind.ema_array[-1] or 0.0) - (ind.ema_array[-2] or 0.0)

    def _ema_slope_desc(slope, threshold=1e-6):
        if slope > threshold:
            return f"sloping up ({slope:.6f})"
        elif slope < -threshold:
            return f"sloping down ({slope:.6f})"
        else:
            return f"flat ({slope:.6f})"

    main_slope = _ema_slope(ind_main); main_slope_txt = _ema_slope_desc(main_slope)
    tf2_slope  = _ema_slope(ind_tf2);  tf2_slope_txt  = _ema_slope_desc(tf2_slope)
    tf3_slope  = _ema_slope(ind_tf3);  tf3_slope_txt  = _ema_slope_desc(tf3_slope)

    def _atr_norm_slope(slope: float, ind: Indicators) -> float:
        atr = ind.atr if ind and ind.atr else None
        ref = atr if (atr and atr > 0) else (ind.price_array[-1] if ind and ind.price_array else 1.0)
        ref = abs(ref) if ref else 1.0
        return slope / ref

    norm_main = _atr_norm_slope(main_slope, ind_main)
    regime = trade.regime or Regime()
    spread_to_sl = trade.spread_to_sl if trade.spread_to_sl is not None else 0.0
    require_sr_tp = bool(trade.require_sr_tp)

    in_recovery_mode = bool(trade.last_trade_was_loss or (trade.unrecovered_loss or 0.0) > 0.0)

    logging.info(f"🔻 RAW PAYLOAD:\n{wrapper.json()}\n---")

    # News override hard-hold
    if trade.news_override:
        return JSONResponse(content={
            "action": "hold",
            "reason": "News conflict — override active",
            "confidence": 0,
            "categories": [],
            "missing_categories": ["trend","momentum","volatility","volume","structure","adx"],
            "needed_to_enter": ["Wait for news window to end."],
            "disqualifiers": ["News override"],
            "session_block": False,
            "god_mode_used": False,
            "ema_context": {},
            "structure_summary": {},
            "fib_summary": {},
            "volatility_context": {},
            "volume_context": {},
            "risk_notes": "",
            "policy": "news_conflict",
            "model_self_audit": {"data_used": [], "data_ignored": [], "confidence_rationale": ""},
            "recovery_mode": in_recovery_mode,
            "force_close": False
        })

    # 19:00 UK profit lock policy
    if trade.update_type == "session_check_19" or is_uk_at_or_after(19):
        if pos and (pos.pnl or 0.0) > 0.0:
            return JSONResponse(content={
                "action": "close",
                "reason": "Session policy: 19:00 UK — locking in profit to avoid spread widening. Confluences: Trend (policy), Momentum (n/a), Volatility (n/a), Volume (n/a), Structure (n/a), ADX (n/a).",
                "confidence": 10,
                "categories": ["trend"],
                "recovery_mode": in_recovery_mode,
                "force_close": True,
                "session_block": True,
                "god_mode_used": False,
                "missing_categories": [],
                "needed_to_enter": [],
                "disqualifiers": ["Session policy exit"],
                "policy": "session_exit",
                "ema_context": {},
                "structure_summary": {},
                "fib_summary": {},
                "volatility_context": {},
                "volume_context": {},
                "risk_notes": "",
                "model_self_audit": {"data_used": [], "data_ignored": [], "confidence_rationale": ""}
            })

    recovery_note = ""
    if in_recovery_mode:
        recovery_note = (
            "\n---\n"
            "RECOVERY MODE: Require ≥5/6 categories and confidence ≥8. "
            "List the exact categories used.\n---\n"
        )

    prompt = f"""{recovery_note}
You are a disciplined prop-firm scalping assistant. Reply in STRICT JSON only.

EXTRA CONTEXT FROM EA:
- regime: {regime.dict()}   # label in ["trend","range","breakout"]
- spread_to_sl: {spread_to_sl:.3f}
- require_sr_tp: {require_sr_tp}  # if True, TP must reference SR/Fib, not just 2R

USE ALL CAPABILITIES:
- Quote numeric values you rely on (MACD, RSI, MFI, ADX, BB width %, ATR multiples, EMA slopes).
- Evidence audit: list data you used vs. ignored (with reason).
- Counterfactuals: minimal changes that would flip HOLD→ENTRY (e.g., "ADX>22 and price retests EMA as support").
- Session rules: honor 19:00–07:00 UK and Friday 17:00+ (session_block=true).
- Risk: state SL method (swing/ATR) and TP method (≥2R OR SR/Fib per require_sr_tp), with numbers.

CATEGORIES (max 1 per cat):
1) TREND: EMA {ema_period} on MAIN TF ONLY (mandatory for entries)
2) MOMENTUM: MACD OR RSI OR Stochastic
3) VOLATILITY: Bollinger Bands OR ATR
4) VOLUME: MFI OR volume spike
5) STRUCTURE: S/R OR Fibonacci OR reversal candle
6) ADX: ADX > 20 and direction

HARD RULES:
- Confluences line is mandatory: "Confluences: Trend (...), Momentum (...), Volatility (...), Volume (...), Structure (...), ADX (...)."
- ABSOLUTE TREND: BUY only if EMA {ema_period} slopes up AND price>EMA; SELL only if slopes down AND price<EMA.
- GOD-MODE: Allowed only if EMA is NOT strongly against. If used, set god_mode_used=true and justify.
- Entries need at least {"5" if in_recovery_mode else "4"} categories and confidence ≥ {"8" if in_recovery_mode else "6"}.
- No new trades 19:00–07:00 UK and after 17:00 UK Friday (set session_block=true).

SL/TP:
- SL beyond last swing or ≥1xATR (state which, with numbers).
- TP ≥2xSL or next S/R/Fib (state which, with numbers). If require_sr_tp=true, TP must be SR/Fib-based.

SLOPES ({tf_label_main}/{tf_label_tf2}/{tf_label_tf3}):
- {tf_label_main} EMA {ema_period}: {main_slope_txt}
- {tf_label_tf2}  EMA {ema_period}: {tf2_slope_txt}
- {tf_label_tf3}  EMA {ema_period}: {tf3_slope_txt}

CONTEXT:
- Indicators (main): {ind_main.dict()}
- Indicators (tf2): {ind_tf2.dict()}
- Indicators (tf3): {ind_tf3.dict()}
- Recent Candles ({tf_label_main}): {[c.dict() for c in candles_main]}
- Cross Signal: {cross_signal} | Meaning: {cross_meaning}
- Position: {pos.dict() if pos else "None"}
- Account:  {acc.dict() if acc else "None"}

Return JSON EXACTLY like:
{{
  "action": "buy|sell|hold|close",
  "reason": "Confluences: Trend (...), Momentum (...), Volatility (...), Volume (...), Structure (...), ADX (...). Then full explanation.",
  "confidence": 0-10,
  "lot": 1,
  "new_sl": 0.0,
  "new_tp": 0.0,
  "categories": ["trend","momentum","volatility","volume","structure","adx"],

  "missing_categories": ["volume","structure"],
  "needed_to_enter": ["ADX>20","price above EMA"],
  "disqualifiers": ["EMA {ema_period} flat","Session block"],

  "session_block": false,
  "god_mode_used": false,

  "ema_context": {{
    "period": {ema_period},
    "price_vs_ema": "above|below|near",
    "aligns_all_tfs": true|false,
    "slopes": {{"{tf_label_main}": "{main_slope_txt}", "{tf_label_tf2}": "{tf2_slope_txt}", "{tf_label_tf3}": "{tf3_slope_txt}"}}
  }},
  "structure_summary": {{"sr_levels": [..], "last_reversal": "hammer|engulfing|none"}},
  "fib_summary": {{"high": 0.0, "low": 0.0, "active_levels": ["38.2","61.8"]}},
  "volatility_context": {{"atr": 0.0, "bb_state": "squeeze|expansion|neutral"}},
  "volume_context": {{"mfi": 0.0, "state": "low|normal|high"}},

  "risk_notes": "1R=..., SL at swing low + ATR buffer...",
  "policy": "none|session_exit|friday_exit|news_conflict",

  "model_self_audit": {{
    "data_used": ["ema_array[-2:]", "price_array[-1]", "macd.main", "mfi", "adx"],
    "data_ignored": ["ichimoku (not part of confluence categories)"],
    "confidence_rationale": "why confidence=X given the evidence"
  }}
}}
"""

    try:
        raw = call_gpt_messages(prompt)
        if not raw or not raw.strip():
            return JSONResponse(content={
                "action":"hold","reason":"Empty model response","confidence":0,
                "categories":[], "recovery_mode": in_recovery_mode,
                "session_block": False, "force_close": False,
                "ema_context":{}, "structure_summary":{}, "fib_summary":{},
                "volatility_context":{}, "volume_context":{}, "risk_notes":"", "policy":"none",
                "model_self_audit":{"data_used":[],"data_ignored":[],"confidence_rationale":""}
            })

        logging.info(f"📩 GPT raw (truncated): {raw[:800]}")
        decision = extract_json_object(raw) or {"raw": raw}
        action = flatten_action(decision)  # NO repair call

        # Build categories set from both JSON list + "Confluences:" line
        claimed = set()
        if isinstance(action.get("categories"), list):
            claimed |= {str(x).lower() for x in action["categories"]}
        claimed |= extract_categories(action.get("reason", ""))
        action["categories"] = sorted(list(claimed))

        # Recovery thresholds
        conf = int(action.get("confidence", 0) or 0)
        in_recovery = in_recovery_mode
        min_cats = 5 if in_recovery else 4
        min_conf = 8 if in_recovery else 6

        # Requirements for entries
        if action.get("action") in {"buy", "sell"}:
            if "confluences:" not in (action.get("reason", "").lower()):
                action["action"] = "hold"
                action["reason"] = (action.get("reason","") + " | Missing mandatory Confluences line.").strip()
            if len(claimed) < min_cats:
                action["action"] = "hold"
                action["reason"] = (action.get("reason","") + f" | Confluence requirement not met ({len(claimed)}/{min_cats}).").strip()

        if action.get("action") in {"buy", "sell"}:
            if "new_sl" not in action or "new_tp" not in action:
                action["action"] = "hold"
                action["reason"] += " | Missing new_sl/new_tp for entry."
            if conf < min_conf:
                action["action"] = "hold"
                action["reason"] += f" | Confidence too low ({conf}<{min_conf})."
            if require_sr_tp and "structure" not in claimed:
                action["action"] = "hold"
                action["reason"] += " | High spread/SL: TP must be SR/Fib-based (Structure)."

        # Session guards
        if action.get("action") in {"buy", "sell"} and is_between_uk_time(19, 7):
            action["action"] = "hold"
            action["reason"] += " | No new trades between 19:00 and 07:00 UK."
            action["session_block"] = True
        if is_friday_5pm_or_later():
            if pos and (pos.pnl or 0.0) > 0 and action.get("action") not in {"close", "hold"}:
                action["action"] = "close"
                action["reason"] += " | Closing profitable trade before weekend."
                action["force_close"] = True
                action["policy"] = "friday_exit"
            elif action.get("action") in {"buy", "sell"}:
                action["action"] = "hold"
                action["reason"] += " | No new trades after 17:00 UK Friday."
                action["session_block"] = True

        # EMA hard guard + soft "god mode"
        def ema_trend_local(ind: Indicators):
            if not ind or not ind.ema_array or len(ind.ema_array) < 2:
                return 0
            prev_ema = ind.ema_array[-2]
            curr_ema = ind.ema_array[-1]
            if curr_ema > prev_ema:
                return 1
            elif curr_ema < prev_ema:
                return -1
            return 0

        def ema_confirms_local(ind: Indicators, action_dir: str):
            if not ind or not ind.ema_array or not ind.price_array:
                return False
            price = ind.price_array[-1]
            ema_last = ind.ema_array[-1]
            if action_dir == "buy":
                return (ema_last is not None and price is not None and price > ema_last and ema_trend_local(ind) == 1)
            if action_dir == "sell":
                return (ema_last is not None and price is not None and price < ema_last and ema_trend_local(ind) == -1)
            return False

        main_trend = ema_trend_local(ind_main)
        strong_thresh = 0.02
        if action.get("action") in {"buy", "sell"}:
            trend_block = False
            if not ema_confirms_local(ind_main, action["action"]):
                if action["action"] == "buy" and (main_trend == -1 and norm_main < -strong_thresh):
                    trend_block = True
                if action["action"] == "sell" and (main_trend == 1 and norm_main > strong_thresh):
                    trend_block = True
            if trend_block:
                action["action"] = "hold"
                action["reason"] += " | EMA trend strongly opposes (hard block)."
                action["god_mode_used"] = False
            else:
                if not ema_confirms_local(ind_main, action["action"]):
                    action["god_mode_used"] = True
                    action["reason"] += " | God-mode: EMA not strongly against — confluence+confidence allow entry."
                agrees_tf2 = ema_trend_local(ind_tf2) == (1 if action["action"] == "buy" else -1)
                agrees_tf3 = ema_trend_local(ind_tf3) == (1 if action["action"] == "buy" else -1)
                if agrees_tf2 and agrees_tf3:
                    action["confidence"] = int(action.get("confidence", 0)) + 1
                    action["reason"] += " | EMA alignment across main/tf2/tf3. Confidence +1."

        # === Low-Volatility Trend Trap (skip late continuation without expansion) ===
        if action.get("action") in {"buy", "sell"}:
            bb_state_main = classify_bb_state(ind_main)
            is_neutral = (bb_state_main == "neutral")
            expanding = bb_expanding(ind_main, ind_tf2)

            # ADX rising + volume rising confirmations
            adx_rising = adx_ok_and_rising(action["action"], ind_main, ind_tf2)
            vol_conf = volume_confirm_buy(ind_main, ind_tf2) if action["action"] == "buy" else volume_confirm_sell(ind_main, ind_tf2)

            # Momentum exhaustion checks
            mom_exh = momentum_exhausted_buy(ind_main, ind_tf2) if action["action"] == "buy" else momentum_exhausted_sell(ind_main, ind_tf2)

            # Trap condition: neutral BB + momentum exhausted + NOT (ADX rising AND BB expanding AND volume confirming)
            if is_neutral and mom_exh and not (adx_rising and expanding and vol_conf):
                action["reason"] += " | Low-Volatility Trend Trap: BB neutral + momentum exhaustion with no expansion/ADX rise/volume push."
                action["disqualifiers"] = _as_list_str(action.get("disqualifiers", [])) + ["low_vol_trend_trap"]
                action["action"] = "hold"

        # Pre-22:00 UK profit lock preference
        if pos and (pos.pnl or 0.0) > 0 and is_between_uk_time(21, 22) and action.get("action") not in {"close", "hold"}:
            action["action"] = "close"
            action["reason"] += " | Pre-22:00 UK: prefer locking profit."
            action["force_close"] = True
            action["policy"] = "session_exit"

        # Breakeven SL: do not move SL backwards
        if pos and action.get("action") and "new_sl" in action and pos.open_price is not None and pos.sl is not None:
            tick_tol = infer_tick_tol(ind_main)
            if abs((pos.sl or 0.0) - (pos.open_price or 0.0)) <= tick_tol:
                action["new_sl"] = pos.sl
                action["reason"] += " | SL at breakeven; not moving SL."

        # Ensure defaults + coercion before validation
        action.setdefault("ema_context", {
            "period": ema_period,
            "price_vs_ema": "unknown",
            "aligns_all_tfs": False,
            "slopes": {tf_label_main: main_slope_txt, tf_label_tf2: tf2_slope_txt, tf_label_tf3: tf3_slope_txt}
        })
        action.setdefault("structure_summary", {"sr_levels": [], "last_reversal": "none"})
        action.setdefault("fib_summary", {"high": 0.0, "low": 0.0, "active_levels": []})
        action.setdefault("volatility_context", {"atr": float(ind_main.atr or 0.0), "bb_state": classify_bb_state(ind_main)})
        action.setdefault("volume_context", {"mfi": float(ind_main.mfi or 0.0), "state": "unknown"})
        action.setdefault("risk_notes", "")
        action.setdefault("policy", "none")
        action.setdefault("force_close", False)
        action.setdefault("model_self_audit", {"data_used": [], "data_ignored": [], "confidence_rationale": ""})
        action["recovery_mode"] = in_recovery_mode

        try:
            validated = sanitize_and_validate(action, "Validated output", in_recovery_mode)
        except ValidationError as ve:
            logging.warning(f"ValidationError (no repair): {ve}")
            validated = DecisionOut(
                action="hold",
                reason="Validation failed.",
                confidence=0,
                categories=[],
                recovery_mode=in_recovery_mode
            )

        logging.info(f"📝 Decision: {validated.model_dump_json()[:1200]}")
        return JSONResponse(content=json.loads(validated.model_dump_json()))

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        in_recovery_mode = bool(trade.last_trade_was_loss or (trade.unrecovered_loss or 0.0) > 0.0)
        return JSONResponse(content={
            "action": "hold",
            "reason": str(e)[:REASON_MAXLEN] + ("..." if len(str(e)) > REASON_MAXLEN else ""),
            "confidence": 0,
            "categories": [],
            "missing_categories": [],
            "needed_to_enter": [],
            "disqualifiers": [],
            "session_block": False,
            "god_mode_used": False,
            "ema_context": {},
            "structure_summary": {},
            "fib_summary": {},
            "volatility_context": {},
            "volume_context": {},
            "risk_notes": "",
            "policy": "none",
            "model_self_audit": {"data_used": [], "data_ignored": [], "confidence_rationale": ""},
            "recovery_mode": in_recovery_mode,
            "force_close": False
        })


@app.get("/")
async def root():
    return {
        "message": "SmartGPT EA SCALPER — gpt-4o only (no fallback, no retries), regime+spread aware, tf2/tf3, evidence audit & counterfactuals, ATR-normalized EMA guard, 19:00 UK profit close, strict confluences, soft God-mode, session guards, recovery mode, JSON hard-validate, plus Low-Volatility Trend Trap filters."
    }
