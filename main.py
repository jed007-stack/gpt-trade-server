from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
import openai
import os
import logging
import requests
from datetime import datetime, timedelta

# === SETUP ===
openai.api_key = os.getenv("OPENAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")  # Get FMP key from environment
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# === DATA MODELS ===
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

# === Helper: Symbol → Currency Pair (for news) ===
def extract_currency(symbol):
    # For XAUUSD, EURUSD, GBPJPY etc, extract main currencies
    majors = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "XAU", "XAG"]
    found = [c for c in majors if c in symbol.upper()]
    return found if found else [symbol.upper()]

# === FMP News Query ===
def check_relevant_news(symbol):
    """Returns True if there is high-impact news in the next 2 hours for either currency in the pair."""
    if not FMP_API_KEY:
        logging.warning("No FMP_API_KEY set, skipping news check.")
        return False  # allow trading if no key
    try:
        currs = extract_currency(symbol)
        now = datetime.utcnow()
        window = now + timedelta(hours=2)
        url = f"https://financialmodelingprep.com/api/v4/economic_calendar?from={now.date()}&to={window.date()}&apikey={FMP_API_KEY}"
        res = requests.get(url)
        if res.status_code != 200:
            logging.warning("FMP API error: " + res.text)
            return False
        events = res.json()
        for ev in events:
            if "country" not in ev or "event" not in ev:
                continue
            event_time = datetime.strptime(ev.get("date"), "%Y-%m-%d %H:%M:%S")
            if now <= event_time <= window:
                # Look for symbol match or major currency match in event name/country
                if any(c in ev.get("event", "") for c in currs) or any(c in ev.get("country", "") for c in currs):
                    if ev.get("impact", "").lower() in ["high", "3"]:  # FMP sometimes uses numbers
                        logging.warning(f"🛑 High-impact news for {currs}: {ev.get('event')}")
                        return True
        return False
    except Exception as e:
        logging.error(f"News API error: {str(e)}")
        return False

# === MAIN ENDPOINT ===
@app.post("/gpt/manage")
async def gpt_manage(wrapper: TradeWrapper):
    trade = wrapper.data
    ind = trade.indicators
    pos = trade.position
    acc = trade.account or Account(balance=10000, equity=10000)

    # Check for high-impact news (returns True = should skip trade)
    if check_relevant_news(trade.symbol):
        return JSONResponse(content={"action": "hold", "reason": "High-impact news detected, no trading"})

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
