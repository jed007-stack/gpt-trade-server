from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
import openai
import os
import logging

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
    adx: float
    atr: float
    macd: MACD

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

# === GPT Manager ===
@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind = trade.indicators
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000)

    logging.info(f"✅ {trade.symbol} | Dir: {trade.direction} | {trade.open_price} → {trade.current_price}")
    logging.info(f"📊 RSI: {ind.rsi}, ADX: {ind.adx}, ATR: {ind.atr}, MACD: {ind.macd.main}/{ind.macd.signal}")

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active"})

    prompt = f"""
You are an expert forex trade manager AI. Make decisions based on price, indicators, and position risk.

Symbol: {trade.symbol}
Timeframe: {trade.timeframe}
Direction: {trade.direction}
Open Price: {trade.open_price}
Current Price: {trade.current_price}

--- Indicators ---
RSI: {ind.rsi}
ADX: {ind.adx}
ATR: {ind.atr}
MACD: main={ind.macd.main}, signal={ind.macd.signal}

--- Position ---
Direction: {pos.direction if pos else "none"}
Lot Size: {pos.lot if pos else "n/a"}
Floating PnL: {pos.pnl if pos else "n/a"}

--- Account ---
Balance: {acc.balance}
Equity: {acc.equity}

Respond ONLY with one of the following:
{{"action": "hold"}}
{{"action": "close"}}
{{"action": "trail_sl", "new_sl": 2345.0}}
{{"action": "martingale", "lot": 0.2}}
"""

    try:
        client = openai.OpenAI()
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a disciplined, risk-aware trading assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision: {decision}")

        if decision.startswith("{"):
            return JSONResponse(content=eval(decision))
        else:
            return JSONResponse(content={"action": "hold", "raw": decision})

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={"action": "hold", "error": str(e)})
