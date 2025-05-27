from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
import openai
import os
import logging

# ==== CONFIG ====
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ==== MODELS ====
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

# ==== ENDPOINT ====
@app.post("/gpt/manage")
async def gpt_manager(request: Request):
    try:
        raw = await request.body()
        logging.info("📦 Raw Body Length: %d", len(raw))
        logging.info("📦 Raw Preview:\n%s", raw.decode("utf-8")[:500])

        payload = TradeData.parse_raw(raw)
        trade = payload

        logging.info(f"✅ Parsed Trade: {trade.symbol} {trade.direction} at {trade.open_price} -> {trade.current_price}")

    except Exception as e:
        logging.error(f"❌ JSON Parse Failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # === GPT PROMPT ===
    prompt = f"""
You are a professional forex position manager.

Evaluate this trade setup:

Symbol: {trade.symbol}
Timeframe: {trade.timeframe}
Direction: {trade.direction}
Open Price: {trade.open_price}
Current Price: {trade.current_price}
Closed Candles 1: {len(trade.candles1)}
Closed Candles 2: {len(trade.candles2)}
Live Candle 1: {trade.live_candle1}
Live Candle 2: {trade.live_candle2}

Respond with:
{{ "action": "hold" }}
{{ "action": "close" }}
{{ "action": "trail_sl", "new_sl": 1234.5 }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                { "role": "system", "content": "You are a disciplined trade manager focused on risk and edge." },
                { "role": "user", "content": prompt }
            ],
            max_tokens=100,
            temperature=0.3
        )
        text = response['choices'][0]['message']['content']
        logging.info(f"🤖 GPT Raw Response: {text}")
        return eval(text) if text.startswith("{") else { "action": "hold" }

    except Exception as e:
        logging.error(f"❌ GPT error: {str(e)}")
        return { "action": "hold", "error": str(e) }
