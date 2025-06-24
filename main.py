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
    sma: Optional[float] = None
    ema: Optional[float] = None
    sma_period: Optional[int] = None
    ema_period: Optional[int] = None
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
    direction: Optional[str] = None
    open_price: Optional[float] = None
    current_price: Optional[float] = None
    news_override: Optional[bool] = False
    indicators: Optional[Indicators] = None  # 1m
    h1_indicators: Optional[Indicators] = None  # 5m
    h4_indicators: Optional[Indicators] = None  # 15m
    position: Optional[Position] = None
    account: Optional[Account] = None
    candles1: Optional[List[Candle]] = []
    candles2: Optional[List[Candle]] = []
    candles3: Optional[List[Candle]] = []
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

    # === SESSION FILTER ===
    if not in_london_ny_session():
        logging.info("⏳ Out of London/NY session, no new trades.")
        return JSONResponse(content={"action": "hold", "reason": "Outside London/New York session", "confidence": 0})

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active", "confidence": 0})

    # === LOGGING ===
    logging.info(f"✅ {trade.symbol} | 1m: {trade.direction} | {trade.open_price} → {trade.current_price}")

    # === GPT PROMPT ===
    prompt = f"""
You are a sniper, scalping-focused trading assistant for prop firm challenges.

**Time/Session filter:** Trade ONLY during London or New York session, never overnight or on weekends.

**Entry rules:**  
- Only consider a trade if 1m EMA/SMA cross matches the trend of at least one higher timeframe (5m or 15m).  
- Require confluence: At least 3 of these confirm for entry:  
  - MACD
  - RSI or Stochastic
  - ADX > 20
  - Ichimoku agrees
  - Bollinger Bands breakout or squeeze
  - Candlestick reversal at a key level (S/R/fibonacci)
- If ALL indicators align on 1m and 5m or 15m, upsize to "lot":2 (otherwise lot 1).
- Skip all ambiguous, low-confidence, or non-session signals.

**Exit/scalp rules:**  
- Prefer fast trailing stop as soon as >1xSL in profit.
- Exit fully if at least 2 indicators warn of a reversal, or structure breaks.
- Always provide "new_sl" if trailing, and "new_tp" if next S/R is close.

**SL/TP:**  
- SL: Just beyond nearest 1m or 5m swing high/low (or min 1xATR)
- TP: At least 2xSL or next major S/R.

**Quality filter:**  
- Always state all confluences in 'reason'.
- Default to “hold” unless a real edge is present.
- Always include "confidence" (1-10).
- Do NOT allow trades that risk being held after 21:50 London time or over the weekend.

Example JSON reply:
{{
  "action": "buy",
  "reason": "1m EMA cross up, 5m uptrend, MACD and ADX > 20, BB breakout. All M1/M5 align.",
  "confidence": 9,
  "lot": 2,
  "new_sl": 2301.5,
  "new_tp": 2310.0
}}
{{
  "action": "hold",
  "reason": "Stochastic and BB squeeze, 5m trend flat. No edge.",
  "confidence": 3
}}

Current Position: {pos.dict() if pos else "None"}
Current Account: {acc.dict() if acc else "None"}
Recent 1m Candles: {[candle.dict() for candle in candles_1m]}
Indicators (1m): {ind_1m.dict()}
Indicators (5m): {ind_5m.dict()}
Indicators (15m): {ind_15m.dict()}
"""

    try:
        client = openai.OpenAI()
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an elite, disciplined, risk-aware SCALPER trade assistant. Reply ONLY in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.13,
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

        # Ensure reasoning present
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
    return {"message": "SmartGPT EA SCALPER - London/NY session, 1m/5m/15m confluence, edge only"}
