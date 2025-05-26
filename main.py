from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class TradeData(BaseModel):
    symbol: str
    timeframe: str
    direction: str
    open_price: float
    current_price: float
    atr: float
    rsi: float
    bb_width: float
    volume: float
    pattern: str
    htf_trend: str

@app.post("/gpt/manage")
async def gpt_manager(trade: TradeData):
    prompt = f"""
You are a professional forex position manager.

Evaluate this trade:
- Symbol: {trade.symbol}
- Timeframe: {trade.timeframe}
- Direction: {trade.direction}
- Open Price: {trade.open_price}
- Current Price: {trade.current_price}
- ATR: {trade.atr}
- RSI: {trade.rsi}
- BB Width: {trade.bb_width}
- Volume: {trade.volume}
- Pattern: {trade.pattern}
- HTF Trend: {trade.htf_trend}

Respond with one of:
{{ "action": "hold" }}
{{ "action": "close" }}
{{ "action": "trail_sl", "new_sl": 2350.0 }}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                { "role": "system", "content": "You are a strict, professional trade manager focused on reducing drawdown and maximizing edge." },
                { "role": "user", "content": prompt }
            ],
            max_tokens=100,
            temperature=0.3
        )
        text = response['choices'][0]['message']['content']
        return eval(text) if text.startswith("{") else { "action": "hold" }

    except Exception as e:
        return { "action": "hold", "error": str(e) }
