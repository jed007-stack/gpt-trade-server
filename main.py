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
    return {"action": "hold", "reason": "Could not decode action.", "confidence": 0}

@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind = trade.indicators
    h1 = trade.h1_indicators or Indicators(**{})
    h4 = trade.h4_indicators or Indicators(**{})
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000, margin=None)
    candles = trade.candles1[-5:] if trade.candles1 else []

    sma_period = getattr(ind, "sma_period", "unknown")
    ema_period = getattr(ind, "ema_period", "unknown")
    atr = ind.atr or 20

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
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active", "confidence": 0})

    prompt = f"""
You are a sniper-style, disciplined, algorithmic trading assistant for prop firm challenges.
Your core values: Quality over quantity, consistency, high-confluence trades, and robust risk management.

**Entry Rules:**
- Only trade if M5 SMA/EMA cross is clear, AND at least 2 additional indicators (MACD, RSI, Stoch, ADX, Ichimoku, Bollinger Bands, candlestick pattern, or support/resistance) confirm.
- H1 OR H4 trend (by MA/EMA or Ichimoku) must AGREE with M5 signal direction.
- If MA/EMA cross is ambiguous or missing, require 4 total indicators in agreement before ANY trade.
- If all major signals (SMA/EMA, MACD, ADX, Ichimoku, S/R, and price action) are perfectly aligned, take the trade with double lot ("lot": 2).
- In ranging conditions (ADX < 20 or BB squeeze), avoid breakouts; only take reversal setups at key S/R with strong confluence.
- If confidence is below 6/10, reply “hold” and explain what is missing in ‘reason’.

**Exit Rules:**
- Exit ONLY if:
  - Clear structure break with at least 2 indicators warning,
  - A true higher timeframe (HTF) trend reversal or multi-indicator flip,
  - Loss of edge after a strong move and technicals warn.
- Don’t close for small profits if quality confluence remains—ride trends until the real structure breaks.
- If a position is open, only "close", "hold", "trail_sl", or suggest better SL/TP unless strong reversal.

**SL/TP Management:**
- Stop-loss must be just beyond nearest swing high/low or S/R, NEVER tighter than 1x ATR.
- Take-profit must target the next major S/R, or at least 2x stop distance (R:R >= 2:1).
- On every response, if a better stop-loss or take-profit is possible, include "new_sl": <price> and/or "new_tp": <price>.
- Suggest breakeven stop if trade is up >1x SL, or trail if strong move.

**Overtrading and Reasoning:**
- Default to “hold” unless all rules are met—better to skip marginal trades.
- Always include a confidence score (1–10), and a clear summary of your reasoning and what confluences are (or are not) present in the ‘reason’ field.

Always reply in strict JSON, like:
{{
  "action": "buy",
  "reason": "M5 EMA cross up, H1 trend up, MACD and RSI confirm, price at support, all signals align.",
  "confidence": 8,
  "new_sl": 1.2345,
  "new_tp": 1.2410
}}
{{
  "action": "hold",
  "reason": "ADX < 20 and mixed signals: EMA cross up but MACD is flat, H1 neutral, wait for better confluence.",
  "confidence": 4
}}

Current Position: {pos.dict() if pos else "None"}
Current Account: {acc.dict() if acc else "None"}
Recent Candles: {[candle.dict() for candle in candles]}
Indicators (M5): {ind.dict()}
Indicators (H1): {h1.dict()}
Indicators (H4): {h4.dict()}
"""

    try:
        client = openai.OpenAI()
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an elite, disciplined, risk-aware trade assistant. Reply ONLY in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.13,
            response_format={"type": "json_object"}
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        action = flatten_action(decision)
        allowed = {"hold", "close", "trail_sl", "trail_tp", "martingale", "buy", "sell"}

        # Confidence filter: only trade if 7+
        conf = action.get("confidence", 0)
        if action.get("action") in {"buy", "sell", "martingale"} and conf < 7:
            action["action"] = "hold"
            action["reason"] = (action.get("reason") or "") + " (confidence too low for entry)"
        # Lot sizing for high confluence
        if action.get("action") in {"buy", "sell", "martingale"} and "lot" not in action:
            action["lot"] = 2 if "double" in (action.get("reason") or "").lower() else 1

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
    return {"message": "SmartGPT EA Sniper Mode Trade Server running!"}
