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
    sma100: Optional[float]
    ema40: Optional[float]

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
    direction: str
    open_price: float
    current_price: float
    news_override: Optional[bool] = False
    indicators: Indicators
    position: Optional[Position] = None
    account: Optional[Account] = None
    candles1: Optional[List[Candle]] = []
    candles2: Optional[List[Candle]] = []
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
        f"📊 RSI: {ind.rsi}, ATR: {ind.atr}, MACD: {ind.macd.main}/{ind.macd.signal} | "
        f"SMA100: {ind.sma100}, EMA40: {ind.ema40}"
    )

    if trade.news_override:
        logging.warning("🛑 News conflict detected. GPT override active.")
        return JSONResponse(content={"action": "hold", "reason": "News conflict — override active"})

    # Build prompt based on SmartGPT_EA's data layout
    prompt = f"""
You are a professional algorithmic trade manager for forex, gold, and crypto.
Decide the best action: 'hold', 'close', 'trail_sl', 'martingale', 'buy', or 'sell'.
Use all available data: position, price, up to 50 candles, indicators, and risk/account info.
Only reply in one JSON object. If market is not favorable, always 'hold'.

Live Data:
Symbol: {trade.symbol}
Timeframe: {trade.timeframe}
Direction: {trade.direction}
Open Price: {trade.open_price}
Current Price: {trade.current_price}
Indicators: RSI {ind.rsi}, ATR {ind.atr}, MACD {ind.macd.main}/{ind.macd.signal}, SMA100 {ind.sma100}, EMA40 {ind.ema40}
Position: {pos.dict() if pos else "None"}
Account: {acc.dict() if acc else "None"}
Recent Candles: {[candle.dict() for candle in candles]}

ALWAYS respond in JSON with a brief reason, e.g.:
{{"action":"hold", "reason":"No valid setup."}}
{{"action":"close", "reason":"Price reversed."}}
{{"action":"trail_sl", "new_sl":2345.0, "reason":"Trailing stop for profit."}}
{{"action":"martingale", "lot":0.2, "reason":"Martingale recovery."}}
{{"action":"buy", "lot":0.2, "reason":"Buy signal."}}
{{"action":"sell", "lot":0.2, "reason":"Sell signal."}}
"""

    try:
        client = openai.OpenAI()
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a disciplined, risk-aware trading assistant. Reply with a single valid JSON object, never markdown."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0.15
        )
        decision = chat.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Decision (raw): {decision}")

        allowed = {"hold", "close", "trail_sl", "martingale", "buy", "sell"}
        action = flatten_action(decision)
        if action.get("action") in allowed:
            logging.info(f"📝 GPT Action: {action.get('action')} | Reason: {action.get('reason','(none)')}")
            return JSONResponse(content=action)
        else:
            return JSONResponse(content={"action": "hold", "reason": f"Could not decode: {decision}"})

    except Exception as e:
        logging.error(f"❌ GPT Error: {str(e)}")
        return JSONResponse(content={"action": "hold", "reason": str(e)})

@app.get("/")
async def root():
    return {"message": "SmartGPT EA Trade Server running!"}
