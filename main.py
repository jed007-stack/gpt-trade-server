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
    sma100: Optional[float]
    ema40: Optional[float]
    adx: Optional[float]
    mfi: Optional[float]
    williams_r: Optional[float]
    ichimoku: Optional[Ichimoku]
    rsi_array: Optional[List[float]]
    price_array: Optional[List[float]]
    support_resistance: Optional[Dict[str, List[float]]] = None
    fibonacci: Optional[Dict[str, Any]] = None
    candlestick_patterns: Optional[List[str]] = None

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
    position: Optional[Position] = None
    account: Optional[Account] = None
    candles1: Optional[List[Candle]] = []
    candles2: Optional[List[Candle]] = []
    candles3: Optional[List[Candle]] = []
    live_candle1: Optional[Candle] = None
    live_candle2: Optional[Candle] = None

class TradeWrapper(BaseModel):
    data: TradeData

# === Helper: Flatten GPT response ===
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

# === GPT Manager ===
@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind = trade.indicators
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000, margin=None)
    candles = trade.candles1[-5:] if trade.candles1 else []

    logging.info(f"✅ {trade.symbol} | Dir: {trade.direction} | {trade.open_price} → {trade.current_price}")
    logging.info(
        f"📊 BB: ({ind.bb_upper}, {ind.bb_middle}, {ind.bb_lower}) | "
        f"Stoch: K={ind.stoch_k}, D={ind.stoch_d}, J={getattr(ind, 'stoch_j', None)} | "
        f"MACD: {ind.macd.main}/{ind.macd.signal} | "
        f"SMA100: {ind.sma100}, EMA40: {ind.ema40} | "
        f"ADX: {getattr(ind, 'adx', None)} | "
        f"MFI: {getattr(ind, 'mfi', None)} | Williams %R: {getattr(ind, 'williams_r', None)} | "
        f"Ichimoku: {ind.ichimoku.dict() if ind.ichimoku else None}"
    )

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active"})

    # === Enhanced Prompt ===
    prompt = f"""
You are an expert, disciplined but confident algorithmic trade manager.
You are managing a prop firm challenge account (e.g., E8), where your priority is to **grow the account quickly while respecting strict risk rules**.

**Your Strategy:**
- Trade only when strong confluence exists (e.g. candle + structure + indicators).
- **Overbought/oversold levels are not reversal signals by default.** In strong trends, treat them as signs of momentum — do not wait unnecessarily.
- Favor trades with ADX > 20–25 and Ichimoku support. These indicate trend strength.
- Trust bullish/bearish candle patterns (e.g. hammer, engulfing) at key levels if backed by BB, MACD, and crossover logic.
- EMA/SMA crossovers are strong entry signals when confirmed by price and indicators.
- RSI extremes alone are not reasons to avoid trades. Look for pattern and trend confirmation.
- dont move sl to Break even too soon. note the charges are $7 for every 1 lot lets try and cover that when moving the stop.
**Risk & Execution Rules:**
- Never break prop firm drawdown rules (daily or max).
- Use `"lot": 2` only with overwhelming confluence and clean trend.
- Do not add to an open position. Do not flip unless a true reversal is present.
- Use `"hold"` only if the market is truly choppy, conflicting, or lacking a clear edge.
- Open `"buy"` or `"sell"` confidently when trend, structure, and indicators align.

**SL/TP Rules:**
- Always return `"new_sl"` and/or `"new_tp"` if structure justifies it.
- Use `"trail_sl"` if price is in profit and trend is continuing.
- Raise `"new_sl"` to breakeven after 50+ pips if not already suggested.

**Respond ONLY in strict JSON format:**
Examples:
{{"action":"buy","lot":1,"reason":"Bullish engulfing + BB breakout + ADX 30 + Ichimoku support."}}  
{{"action":"trail_sl","new_sl":1.2350,"reason":"Protecting gains with trend intact."}}  
{{"action":"hold","new_sl":1.2220,"reason":"Raising SL to breakeven while monitoring for continuation."}}  
{{"action":"close","reason":"Structure break and MACD crossover against trend."}}

Current Position: {pos.dict() if pos else "None"}
Current Account: {acc.dict() if acc else "None"}
Recent Candles: {[candle.dict() for candle in candles]}

Market Data:
Symbol: {trade.symbol}
Timeframe: {trade.timeframe}
Open Price: {trade.open_price}
Current Price: {trade.current_price}
Indicators: 
  BB: {ind.bb_upper}/{ind.bb_middle}/{ind.bb_lower}
  Stoch K/D/J: {ind.stoch_k}/{ind.stoch_d}/{getattr(ind, 'stoch_j', None)}
  MACD: {ind.macd.main}/{ind.macd.signal}
  SMA100: {ind.sma100}
  EMA40: {ind.ema40}
  ADX: {getattr(ind, 'adx', None)}
  MFI: {getattr(ind, 'mfi', None)}
  Williams %R: {getattr(ind, 'williams_r', None)}
  Ichimoku: {ind.ichimoku.dict() if ind.ichimoku else None}
  RSI Array: {ind.rsi_array}
  Price Array: {ind.price_array}
  Support/Resistance: {ind.support_resistance}
  Fibonacci: {ind.fibonacci}
  Candlestick Patterns: {ind.candlestick_patterns}
"""

    try:
        client = openai.OpenAI()
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a disciplined, confident, risk-aware trade assistant. Reply ONLY in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.22
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        allowed = {"hold", "close", "trail_sl", "trail_tp", "martingale", "buy", "sell"}
        action = flatten_action(decision)
        if action.get("action") in {"buy", "sell", "martingale"} and "lot" not in action:
            action["lot"] = 1
        if action.get("action") in allowed:
            logging.info(f"📝 GPT Action: {action.get('action')} | Lot: {action.get('lot', 1)} | Reason: {action.get('reason','(none)')}")
            return JSONResponse(content=action)
        else:
            return JSONResponse(content={"action": "hold", "reason": f"Could not decode: {decision}"})

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={"action": "hold", "reason": str(e)})

@app.get("/")
async def root():
    return {"message": "SmartGPT EA Trade Server running!"}
