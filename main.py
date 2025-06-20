from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import JSONResponse
import openai
import os
import logging
import json
import re

# === Setup ===
openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# === Data Models ===

class MACD(BaseModel):
    main: float
    signal: float

class Ichimoku(BaseModel):
    tenkan: float
    kijun: float
    senkou_a: float
    senkou_b: float

class Indicators(BaseModel):
    bb_upper: float
    bb_middle: float
    bb_lower: float
    stoch_k: float
    stoch_d: float
    stoch_j: Optional[float]
    macd: MACD
    sma: Optional[float] = None
    ema: Optional[float] = None
    sma_period: Optional[int] = None
    ema_period: Optional[int] = None
    adx: Optional[float]
    mfi: Optional[float]
    williams_r: Optional[float]
    ichimoku: Optional[Ichimoku]
    rsi_array: Optional[List[float]]
    price_array: Optional[List[float]]
    support_resistance: Optional[Dict[str, List[float]]] = None
    fibonacci: Optional[Dict[str, Any]] = None
    candlestick_patterns: Optional[List[str]] = None
    atr: Optional[float] = None

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class Position(BaseModel):
    direction: Optional[str]
    open_price: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    lot: Optional[float]
    pnl: Optional[float]

class Account(BaseModel):
    balance: float
    equity: float
    margin: Optional[float]

class TradeData(BaseModel):
    symbol: str
    timeframe: str
    direction: Optional[str] = None
    open_price: Optional[float] = None
    current_price: Optional[float] = None
    news_override: Optional[bool] = False
    indicators: Indicators
    h1_indicators: Optional[Indicators] = None
    h4_indicators: Optional[Indicators] = None
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
    return {"action": "hold", "reason": "Could not decode action."}

@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind = trade.indicators
    h1 = trade.h1_indicators or Indicators(**{})
    h4 = trade.h4_indicators or Indicators(**{})
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000, margin=None)
    candles = trade.candles1[-5:] if trade.candles1 else []

    # Use the actual periods from the EA
    sma_period = getattr(ind, "sma_period", "unknown")
    ema_period = getattr(ind, "ema_period", "unknown")

    logging.info(f"✅ {trade.symbol} | Dir: {trade.direction} | {trade.open_price} → {trade.current_price}")
    logging.info(
        f"📊 BB: ({ind.bb_upper}, {ind.bb_middle}, {ind.bb_lower}) | "
        f"Stoch: K={ind.stoch_k}, D={ind.stoch_d}, J={getattr(ind, 'stoch_j', None)} | "
        f"MACD: {ind.macd.main}/{ind.macd.signal} | "
        f"SMA({sma_period}): {ind.sma}, EMA({ema_period}): {ind.ema} | "
        f"ADX: {getattr(ind, 'adx', None)} | "
        f"MFI: {getattr(ind, 'mfi', None)} | Williams %R: {getattr(ind, 'williams_r', None)} | "
        f"Ichimoku: {ind.ichimoku.dict() if ind.ichimoku else None}"
    )

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active"})

    prompt = f"""
You are a sniper-like algorithmic trading assistant for prop firm challenges.
Your core values: Quality over quantity. Only enter or exit when there is true, strong confluence.
Be active and responsive, but never force trades unless all signals are lined up.

**Entry Rules (Sniper Mode):**
- Only trade if the primary MA/EMA cross aligns with **at least 2 additional indicators** (MACD, ADX, Ichimoku, Stoch, RSI, etc).
- If MA/EMA cross is ambiguous or not present, require **at least 4 indicators** in agreement before any trade.
- If all major signals (MA, MACD, ADX, Ichimoku, candlestick structure, and support/resistance) are in perfect alignment, take the trade with double the usual lot ("lot": 2).
- If you don't see a high-probability edge, reply "hold" and wait.

**Exit Rules (High Standard):**
- Exit only if:
  - There is a clear structure break and at least 2 indicators warn.
  - A true higher timeframe (HTF) trend reversal or multi-indicator flip occurs.
  - You have already taken a partial profit (after a strong move away from entry) and technicals signal loss of edge.
- Do NOT close for minor profit if quality confluence remains—**always ride strong trends until the real structure breaks**.
- Move stop loss (or trail) only after a significant move and confirmed by structure/HTF, never pre-emptively.
- Partial profit is allowed, but only after strong move and if structure or indicator warning appears.
- If a position is already open, only "close", "hold", "trail_sl", or suggest better stop-loss/take-profit unless a strong reversal appears.
- Do NOT add to positions in the same direction if one is already open; just "hold" or "trail_sl".

**SL/TP Management:**
- On every response, if a better stop-loss or take-profit is possible, include "new_sl":<price> and/or "new_tp":<price> (absolute price level).
- You may update SL or TP even on hold, to manage risk or lock in profit.
- If you want to move the stop-loss to breakeven after 50 pips, or trail the stop, suggest "new_sl" at the best price.

Always reply in strict JSON.
Example: {{"action":"buy","reason":"All confluence: MA, MACD, ADX, Ichimoku up. Entering with confidence."}}
Example: {{"action":"close","reason":"Structure break, MACD and ADX flipped down, H1 reversal."}}
Example: {{"action":"hold","hold_duration":6,"reason":"Waiting for higher quality setup."}}

Current Position: {pos.dict() if pos else "None"}
Current Account: {acc.dict() if acc else "None"}
Recent Candles: {[candle.dict() for candle in candles]}
Indicators (M5): {ind.dict()}
"""

    try:
        client = openai.OpenAI()
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an elite, disciplined, risk-aware trade assistant. Reply ONLY in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.17
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        allowed = {"hold", "close", "trail_sl", "trail_tp", "martingale", "buy", "sell"}
        action = flatten_action(decision)
        if action.get("action") in {"buy", "sell", "martingale"} and "lot" not in action:
            action["lot"] = 1
        if action.get("action") in allowed or action.get("action") == "hold":
            logging.info(f"📝 GPT Action: {action.get('action')} | Lot: {action.get('lot', 1)} | Reason: {action.get('reason','(none)')}")
            return JSONResponse(content=action)
        else:
            return JSONResponse(content={"action": "hold", "reason": f"Could not decode: {decision}"})

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={"action": "hold", "reason": str(e)})

@app.get("/")
async def root():
    return {"message": "SmartGPT EA Sniper Mode Trade Server running!"}
