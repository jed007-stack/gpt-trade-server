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
openai.api_key = os.getenv("OPENAI_API_KEY")
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
    indicators: Optional[Indicators] = None       # 1m
    h1_indicators: Optional[Indicators] = None    # 5m
    h4_indicators: Optional[Indicators] = None    # 15m
    position: Optional[Position] = None
    account: Optional[Account] = None
    candles1: Optional[List[Candle]] = []
    candles2: Optional[List[Candle]] = []
    candles3: Optional[List[Candle]] = []
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

def in_london_ny_session(utc_dt=None):
    utc_now = utc_dt or datetime.utcnow()
    london = pytz.timezone('Europe/London')
    london_now = utc_now.replace(tzinfo=pytz.utc).astimezone(london)
    l_start = time(7, 0)
    l_end   = time(16, 30)
    n_start = time(13, 0)
    n_end   = time(22, 0)
    session = (l_start <= london_now.time() <= l_end) or (n_start <= london_now.time() <= n_end)
    weekday = london_now.weekday()
    return session and (weekday < 5)

@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind_1m = trade.indicators or Indicators()
    ind_5m = trade.h1_indicators or Indicators()
    ind_15m = trade.h4_indicators or Indicators()
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000, margin=None)
    candles_1m = trade.candles1[-5:] if trade.candles1 else []
    candles_5m = trade.candles2[-5:] if trade.candles2 else []
    candles_15m = trade.candles3[-5:] if trade.candles3 else []
    cross_signal = trade.cross_signal or "none"
    cross_meaning = trade.cross_meaning or "none"

    # LOG incoming payload
    logging.info(f"🔻 RAW PAYLOAD:\n{wrapper.json()}\n---")

    # SESSION FILTER
    if not in_london_ny_session():
        logging.info("⏳ Out of London/NY session, no new trades.")
        return JSONResponse(content={"action": "hold", "reason": "Outside London/New York session", "confidence": 0})

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active", "confidence": 0})

    # LOGGING SUMMARY
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
You are an elite, decisive prop firm scalper trade assistant. 
***You must take every setup that meets the entry rules.*** 
Only reply "hold" if there is a direct conflict or a clear lack of confluence.

**Trade at all times except between 21:00 and 23:00 UK time.**

**Entry:**
- The latest cross_signal from the EA is: {cross_signal}
- The latest cross_meaning from the EA is: {cross_meaning}
- Only take trades if the 1m EMA/LWMA cross matches the trend of at least one higher timeframe (5m or 15m).
- If you detect 3 or more of the following ("confluences") with no direct conflicts, issue a trade ("buy" or "sell"). 
  - MACD
  - RSI or Stochastic
  - ADX > 20
  - Ichimoku agrees
  - Bollinger Bands breakout or squeeze
  - Candlestick reversal at a key level (S/R/fibonacci)
- If ALL indicators align (1m, 5m, 15m), lot size should be 2. Otherwise, use 1.

**Exit/scalp:**
- Move SL to breakeven after 0.40% account profit.
- At 1% profit, use a 0.30% trailing stop.
- Take partial profit if unrealized profit > 0.10% of balance. Take another partial at 0.20%.
- Exit the rest if 2+ indicators reverse or structure breaks.
(Unrealized profit = pos.pnl, balance = account.balance)


**SL/TP:**  
- always suggest New SL and TP based on high timeframe. 
- SL: Just beyond nearest 1m or 5m swing high/low (or min 1xATR)
- TP: At least 2xSL or next major S/R.

**Quality filter:**
- If at least 3 confluences and the main signal matches a higher timeframe, always take a trade.
- Default to "hold" only if clear conflict or no setup.
- Always include "confidence" (1-10).
- Do NOT allow trades that risk being held after 21:00 UK or over the weekend.
- Always give all confluences in 'reason'.

Example JSON reply:
{{
  "action": "buy",
  "reason": "1m EMA over LWMA, 5m uptrend, MACD and ADX > 20, BB breakout. Aggressive trailing stop set, partial close triggered.",
  "confidence": 9,
  "lot": 2,
  "new_sl": 2301.5,
  "new_tp": 2310.0,
  "partial_close": 0.5
}}
{{
  "action": "hold",
  "reason": "5m trend flat, or conflict between MACD and price action.",
  "confidence": 3
}}

Current Cross Signal: {cross_signal}
Current Cross Meaning: {cross_meaning}
Current Position: {pos.dict() if pos else "None"}
Current Account: {acc.dict() if acc else "None"}
Recent 1m Candles: {[candle.dict() for candle in candles_1m]}
Indicators (1m): {ind_1m.dict()}
Indicators (5m): {ind_5m.dict()}
Indicators (15m): {ind_15m.dict()}
"""

    logging.info(f"\n========= SENDING PROMPT TO GPT =========\n{prompt}\n========================================\n")

    try:
        client = openai.OpenAI()
        chat = client.chat.completions.create(
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

        # Confidence filter: only trade if 7+
        conf = action.get("confidence", 0)
        if action.get("action") in {"buy", "sell"} and conf < 7:
            action["action"] = "hold"
            action["reason"] = (action.get("reason") or "") + " (confidence too low for entry)"
        # Lot sizing for high confluence
        if action.get("action") in {"buy", "sell"} and "lot" not in action:
            action["lot"] = 2 if "double" in (action.get("reason") or "").lower() or conf >= 9 else 1

        if "reason" not in action or not action["reason"]:
            action["reason"] = "No reasoning returned by GPT."

        # Log decision and return
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
    return {"message": "SmartGPT EA SCALPER - London/NY session, EMA/LWMA/SMMA confluence, aggressive trailing stop, partial profits"}
