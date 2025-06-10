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

    # === Balanced, prop firm-specific patient prompt ===
    prompt = f"""
You are an expert, risk-aware, but patient algorithmic trade manager.
You are managing an account in a strict prop firm challenge (e.g., E8).
Your primary goal is to pass the challenge as quickly and smoothly as possible **without breaking any risk rules**.

Your goals:
- Prefer to hold open trades until a strong, multi-indicator reversal is clear.
- Only signal "close" when there is a very strong reversal or risk of breaching rules.
- Avoid closing trades on minor retracements or mixed signals; be patient and trend-following.
- Only trade when there is a valid edge, but act quickly when all strong signals align.
- Never flip directions unless the reversal is clear and strong; otherwise, signal "close", "hold" or "trail_sl".
- Do NOT add to positions in the same direction if one is already open; just "hold" or "trail_sl".
- If a position is already open, only "close", "hold", or suggest better stop-loss/take-profit unless a strong reversal appears.
- Never martingale unless the trend truly resumes after a drawdown (rare).
- Never expose account to over-risk; only use "lot":2 when all evidence is strong.
- Use "hold" if market is choppy, mixed, or low-confidence.
- **Always follow prop firm rules, especially daily and max drawdown.**
- make sure you find and have an edge and stick to it don’t be afraid you’re an expert look for good entries reversals and pass these challenges 
- Look for strong reversal candle patterns at major support/resistance, with confirmation from indicators
- I like crossovers with confirmation from indicators and strong candle pattern
- snipe entries and smash profits a mixture of being aggressive but knowing the market and taking the higher timeframes into account aiming to grow the account fast but safly

IMPORTANT: On every response, **if you think a better stop-loss or take-profit is possible,** include "new_sl":<price> and/or "new_tp":<price> (absolute price level).  
- You may update SL or TP even on hold, to manage risk or lock in profit.
- If you want to move the stop-loss to breakeven after 50 pips, or trail the stop, suggest "new_sl" at the best price.
- Always return a JSON object (not markdown), for example:
{{"action":"hold","new_sl":2311.50,"new_tp":2350.00,"reason":"Raising SL to breakeven, adjusting TP to top of range."}}
{{"action":"trail_sl","new_sl":2312.00,"reason":"Locking in gains as trade moves in profit."}}
{{"action":"close","reason":"Strong reversal and overbought."}}

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
                {"role": "system", "content": "You are a disciplined, but patient, risk-aware trading assistant. Reply with a single valid JSON object, never markdown."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.22
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        allowed = {"hold", "close", "trail_sl", "trail_tp", "martingale", "buy", "sell"}
        action = flatten_action(decision)
        # Default to "lot":1 if not specified on buy/sell
        if action.get("action") in {"buy", "sell", "martingale"} and "lot" not in action:
            action["lot"] = 1

        # Pass through new_sl and new_tp if included
        if "new_sl" in action or "new_tp" in action:
            logging.info(f"🛡️ SL/TP update: SL={action.get('new_sl')} | TP={action.get('new_tp')}")

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
