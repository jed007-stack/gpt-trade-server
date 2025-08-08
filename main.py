from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import JSONResponse
import openai
import os
import logging
import json
import re
from datetime import datetime, time
import pytz

# === Setup ===
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")
openai_client = openai.OpenAI(api_key=api_key)

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# === Data Models ===
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

class TradeData(BaseModel):
    symbol: str
    timeframe: str
    update_type: Optional[str] = None  # "ema_over_lwma", "session_check_19", etc.
    cross_signal: Optional[str] = None
    cross_meaning: Optional[str] = None
    indicators: Optional[Indicators] = None         # main TF
    tf2_indicators: Optional[Indicators] = None     # tf2
    tf3_indicators: Optional[Indicators] = None     # tf3
    timeframes: Optional[Timeframes] = None
    tf_names: Optional[Dict[str, str]] = None       # e.g., {"main":"M5","tf2":"M15","tf3":"H1"}
    position: Optional[Position] = None
    account: Optional[Account] = None
    news_override: Optional[bool] = False
    live_candle1: Optional[Candle] = None
    live_candle2: Optional[Candle] = None
    last_trade_was_loss: Optional[bool] = False
    unrecovered_loss: Optional[float] = 0.0
    strong_crossover: Optional[bool] = False

class TradeWrapper(BaseModel):
    data: TradeData

# === Helpers ===
def flatten_action(decision):
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
    m = re.search(r"Confluences?:\s*([^.]+)", reason, re.IGNORECASE)
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
    # Guess tolerance from price decimals or fallback
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
    return 10 ** (-(decimals - 1))  # a bit looser than 1 tick

def atr_norm_slope(slope: float, ind: Indicators) -> float:
    atr = ind.atr if ind and ind.atr else None
    ref = atr if (atr and atr > 0) else (ind.price_array[-1] if ind and ind.price_array else 1.0)
    ref = abs(ref) if ref else 1.0
    return slope / ref

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
    cross_signal = trade.cross_signal or "none"
    cross_meaning = trade.cross_meaning or "none"

    ema_period = ind_main.ema_period or 100  # dynamic EMA period from payload (EA)
    tf_label_main = (trade.tf_names or {}).get("main", "main")
    tf_label_tf2  = (trade.tf_names or {}).get("tf2",  "tf2")
    tf_label_tf3  = (trade.tf_names or {}).get("tf3",  "tf3")

    # Slopes
    main_slope = ema_slope(ind_main); main_slope_txt = ema_slope_desc(main_slope)
    tf2_slope  = ema_slope(ind_tf2);  tf2_slope_txt  = ema_slope_desc(tf2_slope)
    tf3_slope  = ema_slope(ind_tf3);  tf3_slope_txt  = ema_slope_desc(tf3_slope)

    # ATR-normalized slope to judge "strongly opposes"
    norm_main = atr_norm_slope(main_slope, ind_main)

    # Recovery mode
    in_recovery_mode = bool(trade.last_trade_was_loss or (trade.unrecovered_loss or 0.0) > 0.0)

    logging.info(f"🔻 RAW PAYLOAD:\n{wrapper.json()}\n---")
    if trade.news_override:
        return JSONResponse(content={
            "action": "hold",
            "reason": "News conflict — override active",
            "confidence": 0,
            "categories": [],
            "recovery_mode": in_recovery_mode,
            "force_close": False,
            "session_block": False,
            "god_mode_used": False,
            "missing_categories": ["trend","momentum","volatility","volume","structure","adx"],
            "needed_to_enter": ["Wait for news window to end."],
            "disqualifiers": ["News override"]
        })

    # 7pm UK policy: if EA pings at 19:00 or it's 19:00+, close profitable positions (force_close)
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
                "disqualifiers": ["Session policy exit"]
            })

    # Recovery note
    recovery_note = ""
    if in_recovery_mode:
        recovery_note = (
            "\n---\n"
            "RECOVERY MODE: The last trade was a loss and has not been fully recovered. "
            "Recommend a trade ONLY if at least 5/6 unique categories align and confidence ≥ 8. "
            "List the exact categories used (Trend, Momentum, Volatility, Volume, Structure, ADX).\n---\n"
        )

    # Prompt — ask for rich, structured JSON
    prompt = f"""{recovery_note}
You are a decisive, disciplined prop-firm trading assistant. Reply in STRICT JSON matching the schema shown below.

GOAL:
- Always explain: what you HAVE now (by category), what you're MISSING, and what you WOULD NEED to take a trade (the minimal missing conditions).
- If you recommend HOLD, specify disqualifiers and the exact missing items.
- If you recommend CLOSE, justify using categories or session/policy reasons.
- If you recommend BUY/SELL, include 'new_sl' and 'new_tp', and detail SL/TP rationale (swing/ATR/Fib/SR).

CATEGORIES (max 1 indicator per category):
1) TREND: EMA {ema_period} on main TF ONLY (mandatory for entries)
2) MOMENTUM: MACD or RSI or Stochastic
3) VOLATILITY: Bollinger Bands or ATR
4) VOLUME: MFI or clear volume spike
5) STRUCTURE: Support/Resistance OR Fibonacci alignment OR reversal candle
6) ADX: ADX > 20 and direction

RULES:
- Confluences line is mandatory, every time: "Confluences: Trend (...), Momentum (...), Volatility (...), Volume (...), Structure (...), ADX (...)."
- ABSOLUTE TREND RULE: BUY only if EMA {ema_period} slopes up AND price>EMA; SELL only if EMA {ema_period} slopes down AND price<EMA.
- GOD-MODE: Allowed only if EMA is NOT strongly against. If strongly opposite, HOLD. If used, set "god_mode_used": true and explain.
- Entries require at least {(5 if in_recovery_mode else 4)} unique categories aligned and confidence ≥ {(8 if in_recovery_mode else 6)}.
- No new trades 19:00–07:00 UK, and none after 17:00 UK Friday. If blocked by session, set "session_block": true and HOLD unless closing profits.

SL/TP:
- SL beyond last swing or ≥ 1xATR (state which).
- TP ≥ 2xSL or next S/R/Fib level (state which).
- Return numeric new_sl/new_tp for entries.

SLOPES ({tf_label_main}/{tf_label_tf2}/{tf_label_tf3}):
- {tf_label_main} EMA {ema_period}: {main_slope_txt}
- {tf_label_tf2}  EMA {ema_period}: {tf2_slope_txt}
- {tf_label_tf3}  EMA {ema_period}: {tf3_slope_txt}

CONTEXT TO USE:
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
  "reason": "Confluences: Trend (...), Momentum (...), Volatility (...), Volume (...), Structure (...), ADX (...). Then explanation.",
  "confidence": 0-10,
  "lot": 1,
  "new_sl": 0.0,
  "new_tp": 0.0,
  "categories": ["trend","momentum","volatility","volume","structure","adx"],
  "missing_categories": ["volume","structure"],                # what’s missing RIGHT NOW
  "needed_to_enter": ["ADX>20","price above EMA"],             # exact minimal requirements you want
  "disqualifiers": ["EMA {ema_period} flat", "Session block"], # hard blockers if any
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
  "policy": "none|session_exit|friday_exit|news_conflict"
}}
"""

    try:
        chat = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are an elite, disciplined, risk-aware SCALPER assistant. Reply ONLY in strict JSON. Include 'new_sl' and 'new_tp' for buy/sell."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=900,
            response_format={"type": "json_object"}
        )
        decision = chat.choices[0].message.content.strip()
        action = flatten_action(decision)

        # Collect categories from JSON + from reason line
        claimed = set()
        if isinstance(action.get("categories"), list):
            claimed |= {str(x).lower() for x in action["categories"]}
        claimed |= extract_categories(action.get("reason", ""))
        action["categories"] = sorted(list(claimed))

        conf = int(action.get("confidence", 0) or 0)
        in_recovery = in_recovery_mode
        min_cats = 5 if in_recovery else 4
        min_conf = 8 if in_recovery else 6

        # Confluence line enforcement for entries
        if action.get("action") in {"buy", "sell"}:
            if "confluences:" not in action.get("reason", "").lower() or len(claimed) < min_cats:
                action["action"] = "hold"
                action["reason"] = (action.get("reason","") + f" | Confluence requirement not met ({len(claimed)}/{min_cats}).").strip()

        # SL/TP requirement for entries
        if action.get("action") in {"buy", "sell"}:
            if "new_sl" not in action or "new_tp" not in action:
                action["action"] = "hold"
                action["reason"] += " | Missing new_sl/new_tp for entry."
            if conf < min_conf:
                action["action"] = "hold"
                action["reason"] += f" | Confidence too low ({conf}<{min_conf})."

        # Session guards (no new entries overnight / Friday late)
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

        # God-mode vs EMA rule (ATR-normalized)
        if action.get("action") in {"buy", "sell"}:
            main_trend = ema_trend(ind_main)
            main_slope_val = main_slope
            norm = norm_main
            # threshold in ATR units (tunable)
            strong_thresh = 0.02  # ~2% of ATR per bar change; adjust if needed
            trend_block = False
            if not ema_confirms(ind_main, action["action"]):
                if action["action"] == "buy" and (main_trend == -1 and norm < -strong_thresh):
                    trend_block = True
                if action["action"] == "sell" and (main_trend == 1 and norm > strong_thresh):
                    trend_block = True
            if trend_block:
                action["action"] = "hold"
                action["reason"] += " | EMA trend strongly opposes (hard block)."
                action["god_mode_used"] = False
            else:
                # If not confirming but not strongly opposite -> allow God-mode
                if not ema_confirms(ind_main, action["action"]):
                    action["god_mode_used"] = True
                    action["reason"] += " | God-mode: EMA not strongly against — confluence+confidence allow entry."
                # Bonus if tf2 & tf3 align
                agrees_tf2 = ema_trend(ind_tf2) == (1 if action["action"] == "buy" else -1)
                agrees_tf3 = ema_trend(ind_tf3) == (1 if action["action"] == "buy" else -1)
                if agrees_tf2 and agrees_tf3:
                    action["confidence"] = int(action.get("confidence", 0)) + 1
                    action["reason"] += " | EMA alignment across main/tf2/tf3. Confidence +1."

        # Pre-22:00 UK: prefer locking profit on existing positions
        if pos and (pos.pnl or 0.0) > 0 and is_between_uk_time(21, 22) and action.get("action") not in {"close", "hold"}:
            action["action"] = "close"
            action["reason"] += " | Pre-22:00 UK: prefer locking profit."
            action["force_close"] = True
            action["policy"] = "session_exit"

        # Prevent SL move when at breakeven
        if pos and action.get("action") and "new_sl" in action and pos.open_price is not None and pos.sl is not None:
            tick_tol = infer_tick_tol(ind_main)
            if abs(pos.sl - pos.open_price) <= tick_tol:
                action["new_sl"] = pos.sl
                action["reason"] += " | SL at breakeven; not moving SL."

        # Ensure rich fields exist even if model skipped them
        action.setdefault("missing_categories", [])
        action.setdefault("needed_to_enter", [])
        action.setdefault("disqualifiers", [])
        action.setdefault("session_block", False)
        action.setdefault("god_mode_used", False)
        action.setdefault("ema_context", {
            "period": ema_period,
            "price_vs_ema": "unknown",
            "aligns_all_tfs": False,
            "slopes": {tf_label_main: main_slope_txt, tf_label_tf2: tf2_slope_txt, tf_label_tf3: tf3_slope_txt}
        })
        action.setdefault("structure_summary", {"sr_levels": [], "last_reversal": "none"})
        action.setdefault("fib_summary", {"high": 0.0, "low": 0.0, "active_levels": []})
        action.setdefault("volatility_context", {"atr": float(ind_main.atr or 0.0), "bb_state": "neutral"})
        action.setdefault("volume_context", {"mfi": float(ind_main.mfi or 0.0), "state": "unknown"})
        action.setdefault("risk_notes", "")
        action.setdefault("policy", "none")
        action.setdefault("force_close", False)
        action["recovery_mode"] = in_recovery

        return JSONResponse(content=action)

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={
            "action": "hold",
            "reason": str(e),
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
            "recovery_mode": in_recovery_mode,
            "force_close": False
        })

@app.get("/")
async def root():
    return {
        "message": "SmartGPT EA SCALPER — tf2/tf3, rich explanations (what’s missing/needed), ATR-normalized EMA guard, 19:00 UK profit close, strict confluences, God-mode (soft), session guards, recovery mode."
    }
