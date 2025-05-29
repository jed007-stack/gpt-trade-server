from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
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

class Indicators(BaseModel):
    rsi: float
    atr: float
    macd: MACD
    sma100: Optional[float] = None
    ema40: Optional[float] = None

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class Position(BaseModel):
    direction: str
    open_price: float
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
    direction: str
    open_price: float
    current_price: float
    news_override: Optional[bool] = False
    indicators: Indicators
    position: Optional[Position]
    account: Optional[Account]
    candles1: List[Candle]
    candles2: List[Candle]
    live_candle1: Candle
    live_candle2: Candle

class TradeWrapper(BaseModel):
    data: TradeData

# === Helper: Flatten GPT response
def flatten_action(decision):
    # If it's already a dict with 'action'
    if isinstance(decision, dict) and "action" in decision:
        return decision
    # If it's a string in code or markdown
    if isinstance(decision, str):
        cleaned = re.sub(r"```(\w+)?", "", decision).strip()
        try:
            d = json.loads(cleaned)
            if "action" in d:
                return d
        except Exception:
            pass
    # If it has 'raw', try to extract from there
    if isinstance(decision, dict) and "raw" in decision:
        cleaned = re.sub(r"```(\w+)?", "", decision["raw"]).strip()
        try:
            d = json.loads(cleaned)
            if "action" in d:
                return d
        except Exception:
            pass
    return {"action": "hold"}

# === GPT Manager ===
@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind = trade.indicators
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000)
    candles = trade.candles1[-5:] if trade.candles1 else []

    logging.info(f"✅ {trade.symbol} | Dir: {trade.direction} | {trade.open_price} → {trade.current_price}")
    logging.info(
        f"📊 RSI: {ind.rsi}, ATR: {ind.atr}, MACD: {ind.macd.main}/{ind.macd.signal} | "
        f"SMA100: {ind.sma100}, EMA40: {ind.ema40}"
    )

    # If news override, avoid all trading
    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active"})

    prompt = f"""
You are a professional algorithmic trade manager for forex and gold.
Make decisions using the current position, price, up to 50 recent candles, all indicators, and account risk.
If the market is not favorable, you must 'hold'.
If there is an open position and indicators favor reversal, you may 'close'.
If the trade is in profit, you may 'trail_sl' to lock in gains.
If a recovery/martingale entry is safe (losing position, indicators favor reversal or bounce), you may 'martingale' with a specified lot size.
If a new trade should be opened, use 'buy' or 'sell' and lot.
If no action, always reply with 'hold'.
NEVER take risky recovery if account equity is at risk.
Make decisions step by step, and always use indicators AND recent candle structure.

Here is the live data:

Symbol: {trade.symbol}
Timeframe: {trade.timeframe}
Direction: {trade.direction}
Open Price: {trade.open_price}
Current Price: {trade.current_price}

--- Indicators ---
RSI: {ind.rsi}
ATR: {ind.atr}
MACD: main={ind.macd.main}, signal={ind.macd.signal}
SMA100: {ind.sma100}
EMA40: {ind.ema40}

--- Position ---
{f"Direction: {pos.direction}, Lot: {pos.lot}, PnL: {pos.pnl}, SL: {pos.sl}, TP: {pos.tp}" if pos else "None"}

--- Account ---
Balance: {acc.balance}, Equity: {acc.equity}, Margin: {acc.margin}

--- Candles (last 5 shown) ---
{[{'o': c.open, 'h': c.high, 'l': c.low, 'c': c.close, 'v': c.volume} for c in candles]}

Respond ONLY in JSON with one of these:
{{"action": "hold"}}
{{"action": "close"}}
{{"action": "trail_sl", "new_sl": 2345.0}}
{{"action": "martingale", "lot": 0.2}}
{{"action": "buy", "lot": 0.2}}
{{"action": "sell", "lot": 0.2}}
"""

    try:
        client = openai.OpenAI()
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a disciplined, risk-aware trading assistant. Only reply with a single valid JSON object, never markdown."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.2
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        allowed = ["hold", "close", "trail_sl", "martingale", "buy", "sell"]
        action = flatten_action(decision)
        if action.get("action") in allowed:
            return JSONResponse(content=action)
        else:
            return JSONResponse(content={"action": "hold", "raw": decision})

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={"action": "hold", "error": str(e)})

# Health check
@app.get("/")
async def root():
    return {"message": "GPT Trade Server running!"}
