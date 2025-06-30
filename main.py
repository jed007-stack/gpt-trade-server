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

# === Data Models (Match EA Payload) ===

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
    lwma: Optional[float] = None
    lwma_period: Optional[int] = None
    smma: Optional[float] = None
    smma_period: Optional[int] = None
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

class TradeData(BaseModel):
    symbol: str
    timeframe: str
    update_type: Optional[str] = None
    cross_signal: Optional[str] = None
    cross_meaning: Optional[str] = None
    indicators: Optional[Indicators] = None     # 1m
    h1_indicators: Optional[Indicators] = None  # 5m
    h4_indicators: Optional[Indicators] = None  # 15m
    position: Optional[Position] = None
    account: Optional[Account] = None
    candles1: Optional[List[Candle]] = None     # 1m
    candles2: Optional[List[Candle]] = None     # 5m
    candles3: Optional[List[Candle]] = None     # 15m
    news_override: Optional[bool] = False
    live_candle1: Optional[Candle] = None
    live_candle2: Optional[Candle] = None

class TradeWrapper(BaseModel):
    data: TradeData

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
    return time(start_h, 0) <= now < time(end_h, 0)

def is_friday_5pm_or_later():
    now = uk_time_now()
    return now.weekday() == 4 and now.time() >= time(17, 0)

@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind_1m = trade.indicators or Indicators()
    ind_5m = trade.h1_indicators or Indicators()
    ind_15m = trade.h4_indicators or Indicators()
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000, margin=None)
    candles_1m = (trade.candles1 or [])[-5:]
    candles_5m = (trade.candles2 or [])[-5:]
    candles_15m = (trade.candles3 or [])[-5:]
    cross_signal = trade.cross_signal or "none"
    cross_meaning = trade.cross_meaning or "none"

    logging.info(f"🔻 RAW PAYLOAD:\n{wrapper.json()}\n---")

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active", "confidence": 0})

    logging.info(f"✅ {trade.symbol} | 1m Dir: {getattr(pos, 'direction', None)} | {getattr(pos, 'open_price', None)} → {getattr(pos, 'pnl', None)}")
    logging.info(
        f"📊 1m BB: ({ind_1m.bb_upper}, {ind_1m.bb_middle}, {ind_1m.bb_lower}) | "
        f"Stoch: K={ind_1m.stoch_k}, D={ind_1m.stoch_d}, J={ind_1m.stoch_j} | "
        f"MACD: {getattr(ind_1m.macd, 'main', None)}/{getattr(ind_1m.macd, 'signal', None)} | "
        f"EMA: {ind_1m.ema} LWMA: {ind_1m.lwma} SMMA: {ind_1m.smma} | "
        f"ADX: {ind_1m.adx} | MFI: {ind_1m.mfi} | WillR: {ind_1m.williams_r}"
    )

    # ========== GPT PROMPT ==========
    prompt = f"""
You are a decisive, disciplined prop firm trading assistant.

- The EA HANDLES ALL partial profits and moves the stop loss (SL) to breakeven.
- IF THE STOP LOSS IS AT BREAKEVEN (SL == entry price), YOU MUST NOT SUGGEST OR MOVE THE SL.
- DO NOT open any new trades after 17:00 UK time on Friday. Only close or manage existing trades.
- DO NOT open any new trades between 21:00 and 23:00 UK time.
- ALWAYS try to close profitable trades before 22:00 UK time or before the weekend.
- You CAN suggest a new take profit (TP) or a full close if necessary.
- You MUST require at least 3 confluences for a new entry.
- If at least three confluences are present and there is no direct conflict, you should generally favor taking the trade, unless there is a clear reversal or major uncertainty.
- ONLY reply in VALID JSON using the example format.
- If you are not certain, or if the entry rules are not met, reply:
  {{
    "action": "hold",
    "reason": "Explain in detail why you are not taking a trade. Mention if the market is choppy, range-bound, unclear trend, indicators are not aligned, or if specific confluences are missing.",
    "confidence": 2
  }}

IMPORTANT:
- DO NOT move or suggest a new SL if the SL is already at breakeven.
- DO NOT suggest new entries after 17:00 UK Friday or between 21:00 and 23:00 UK time.
- ONLY take entries when 1m EMA/LWMA cross matches the trend of 5m or 15m AND you have at least three confluences.

ENTRY RULES:
- The latest cross_signal from the EA is: {cross_signal}
- The latest cross_meaning from the EA is: {cross_meaning}
- Only take trades if the 1m EMA/LWMA cross matches the trend of at least one higher timeframe (5m or 15m).
- If you detect 3 or more of the following ("confluences") with no direct conflicts, issue a trade ("buy" or "sell"):
  - MACD
  - SMMA
  - RSI or Stochastic
  - ADX > 20
  - Ichimoku agrees
  - Bollinger Bands breakout or squeeze
  - Candlestick reversal at a key level (S/R/fibonacci)
- If ALL indicators align (1m, 5m, 15m), lot size should be 2. Otherwise, use 1.

EXIT/SCALP:
- Do NOT suggest or move SL if SL is at breakeven (SL == entry price).
- Suggest a new TP or a full close if a strong reversal/confluence break occurs.
- Always try to close profitable trades before the weekend.

SL/TP:
- Always suggest new SL and TP based on high timeframe, but only if SL is not at breakeven.
- SL: Just beyond nearest 1m or 5m swing high/low (or min 1xATR)
- TP: At least 2xSL or next major S/R.

EXAMPLES (JSON):
{{
  "action": "buy",
  "reason": "1m EMA over LWMA, 5m uptrend, MACD, ADX, and BB breakout. Three confluences, strong entry.",
  "confidence": 9,
  "lot": 2,
  "new_sl": 2301.5,
  "new_tp": 2310.0
}}
{{
  "action": "close",
  "reason": "Reversal on 5m, confluence breakdown.",
  "confidence": 9
}}
{{
  "action": "hold",
  "reason": "Hold, market is range-bound and Stochastic/MACD are not aligned.",
  "confidence": 3
}}

Current Cross Signal: {cross_signal}
Current Cross Meaning: {cross_meaning}
Current Position: {pos.dict() if pos else "None"}
Current Account: {acc.dict() if acc else "None"}
SL: {getattr(pos, 'sl', None)} | Open: {getattr(pos, 'open_price', None)}
Recent 1m Candles: {[candle.dict() for candle in candles_1m]}
Indicators (1m): {ind_1m.dict()}
Indicators (5m): {ind_5m.dict()}
Indicators (15m): {ind_15m.dict()}
"""

    logging.info(f"\n========= SENDING PROMPT TO GPT =========\n{prompt}\n========================================\n")

    try:
        chat = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an elite, disciplined, risk-aware SCALPER trade assistant. Reply ONLY in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.11,
            response_format={"type": "json_object"}
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        action = flatten_action(decision)
        allowed = {"hold", "close", "trail_sl", "trail_tp", "buy", "sell"}

        conf = action.get("confidence", 0)
        if action.get("action") in {"buy", "sell"} and conf < 6:
            action["action"] = "hold"
            action["reason"] = (action.get("reason") or "") + " (confidence too low for entry)"
        if action.get("action") in {"buy", "sell"} and "lot" not in action:
            action["lot"] = 2 if "double" in (action.get("reason") or "").lower() or conf >= 9 else 1

        if "reason" not in action or not action["reason"]:
            action["reason"] = "No reasoning returned by GPT."

        # === SL Breakeven Guard ===
        if pos and pos.open_price and pos.sl is not None:
            if abs(pos.sl - pos.open_price) < 1e-5:
                if "new_sl" in action and action["new_sl"] != pos.sl:
                    action["reason"] += " | SL at breakeven, not allowed to move."
                    action["new_sl"] = pos.sl

        # === FRIDAY 5PM+ GUARD: No new trades after 17:00 Friday, close profits ===
        if is_friday_5pm_or_later():
            if pos and pos.pnl and pos.pnl > 0 and action.get("action") not in {"close", "hold"}:
                action["action"] = "close"
                action["reason"] += " | Closing profitable trade before weekend."
            elif action.get("action") in {"buy", "sell"}:
                action["action"] = "hold"
                action["reason"] += " | No new trades after 17:00 UK time Friday (weekend risk)."

        # === TIME GUARD: Block new entries after 21:00, force closing profits before 22:00 ===
        if pos and pos.pnl and acc:
            if is_between_uk_time(21, 23) and action.get("action") in {"buy", "sell"}:
                action["action"] = "hold"
                action["reason"] += " | No new trades between 21:00 and 23:00 UK time."
            if is_between_uk_time(21, 22) and pos.pnl > 0 and action.get("action") not in {"close", "hold"}:
                action["action"] = "close"
                action["reason"] += " | Closing profitable trade before 22:00 UK to avoid spread widening."

        logging.info(f"📝 GPT Action: {action.get('action')} | Lot: {action.get('lot', 1)} | Confidence: {action.get('confidence', 0)} | Reason: {action.get('reason','(none)')}")
        if action.get("action") in allowed:
            return JSONResponse(content=action)
        else:
            return JSONResponse(content={"action": "hold", "reason": f"Could not decode: {decision}", "confidence": 0})

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={"action": "hold", "reason": str(e), "confidence": 0})

@app.get("/")
async def root():
    return {"message": "SmartGPT EA SCALPER - All Sessions, EMA/LWMA/SMMA confluence, 3-confluence filter, confidence 6+, 1m/5m/15m logic, prop firm weekend safety, partial profits & BE handled by EA"}
