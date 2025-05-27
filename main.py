from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import openai
import os
import logging

# Set OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ======= Data Models =======
class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class TradeData(BaseModel):
    symbol: str
    timeframe: str
    direction: str
    open_price: float
    current_price: float
    candles1: List[Candle]
    candles2: List[Candle]
    live_candle1: Candle
    live_candle2: Candle

class TradeWrapper(BaseModel):
    data: TradeData

# ======= Endpoint =======
@app.post("/gpt/manage")
async def gpt_manager(wrapper: TradeWrapper):
    trade = wrapper.data

    # Calculate simple unrealized PnL and ATR(1)
    pnl = trade.current_price - trade.open_price if trade.direction == "buy" else trade.open_price - trade.current_price
    atr = abs(trade.candles1[-1].high - trade.candles1[-1].low)

    logging.info(f"✅ Received: {trade.symbol} {trade.direction} from {trade.open_price} -> {trade.current_price}")
    logging.info(f"🕯️ Candles1: {len(trade.candles1)} | Candles2: {len(trade.candles2)} | PnL: {pnl:.2f} | ATR: {atr:.2f}")

    # === GPT Prompt ===
    prompt = f"""
You are a professional forex position manager.

Evaluate this trade setup:

Symbol: {trade.symbol}
Timeframe: {trade.timeframe}
Direction: {trade.direction}
Open Price: {trade.open_price}
Current Price: {trade.current_price}
Unrealized PnL: {pnl:.2f}
Approx ATR: {atr:.2f}
Candle Count 1: {len(trade.candles1)}
Candle Count 2: {len(trade.candles2)}

Respond strictly with one of:
{{ "action": "hold" }}
{{ "action": "close" }}
{{ "action": "trail_sl", "new_sl": 2350.0 }}
"""

    try:
        # Use legacy or v1 OpenAI client
        client = openai.OpenAI() if hasattr(openai, "OpenAI") else openai

        response = client.ChatCompletion.create(
            model="gpt-4",
            messages=[
                { "role": "system", "content": "You are a disciplined trade manager focused on risk and edge." },
                { "role": "user", "content": prompt }
            ],
            max_tokens=100,
            temperature=0.3
        )

        text = response.choices[0].message.content.strip()
        logging.info(f"🎯 GPT Response: {text}")

        return eval(text) if text.startswith("{") else { "action": "hold" }

    except Exception as e:
        logging.error(f"❌ GPT error: {str(e)}")
        return { "action": "hold", "error": str(e) }
