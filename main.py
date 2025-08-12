from fastapi import FastAPI
from pydantic import BaseModel, ValidationError, conint, confloat
from typing import List, Optional, Dict, Any, Literal
from fastapi.responses import JSONResponse
from openai import OpenAI
import os, logging, json, re
from datetime import datetime, time
import pytz

# === Setup ===
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

MODEL_ID     = os.getenv("OPENAI_MODEL", "gpt-5")
FALLBACK_ID  = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-5-mini")
openai_client = OpenAI(api_key=api_key)

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ---------- Payload (lean, matches EA) ----------
class MACD(BaseModel):
    main: Optional[float] = None
    signal: Optional[float] = None

class IndicatorsMain(BaseModel):
    adx: Optional[float] = None
    macd: Optional[MACD] = None
    rsi: Optional[float] = None
    mfi: Optional[float] = None
    atr: Optional[float] = None
    bb_state: Optional[Literal["compressed","expanded","neutral"]] = "neutral"

class IndicatorsTF2(BaseModel):
    ema_period: Optional[int] = None
    ema: Optional[float] = None
    ema_slope: Optional[float] = None
    ema_trend: Optional[Literal["up","down","flat"]] = "flat"
    price_vs_ema: Optional[Literal["above","below","at"]] = "at"

class CandleOC(BaseModel):
    open: Optional[float] = None
    close: Optional[float] = None

class Timeframes(BaseModel):
    main: Optional[List[CandleOC]] = None
    tf2: Optional[List[CandleOC]] = None

class Position(BaseModel):
    direction: Optional[Literal["buy","sell"]] = None
    open_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    lot: Optional[float] = None
    pnl: Optional[float] = None

class TradeData(BaseModel):
    symbol: str
    timeframe: str
    update_type: Optional[str] = None      # "ema_over_lwma"/"ema_under_lwma"
    cross_signal: Optional[str] = None
    cross_meaning: Optional[str] = None

    indicators: Optional[IndicatorsMain] = None
    tf2_indicators: Optional[IndicatorsTF2] = None
    timeframes: Optional[Timeframes] = None
    tf_names: Optional[Dict[str,str]] = None

    position: Optional[Position] = None

    # guards & microstructure
    news_override: Optional[bool] = False
    spread_pips: Optional[float] = 0.0
    spread_to_sl: Optional[float] = 0.0
    require_sr_tp: Optional[bool] = False

class TradeWrapper(BaseModel):
    data: TradeData

# ---------- Outgoing decision (lean) ----------
AllowedAction = Literal["buy","sell","hold","close"]

class DecisionOut(BaseModel):
    action: AllowedAction
    reason: str
    confidence: conint(ge=0, le=10) = 0
    categories: List[str] = []
    new_sl: Optional[confloat(ge=0)] = None
    new_tp: Optional[confloat(ge=0)] = None
    lot: Optional[confloat(ge=0)] = None
    session_block: bool = False
    god_mode_used: bool = False
    force_close: bool = False
    policy: Optional[Literal["none","session_exit","friday_exit","news_conflict"]] = "none"

# ---------- Small helpers ----------
def extract_json_object(s: str):
    if not isinstance(s, str):
        return None
    s = re.sub(r"```(?:json|JSON)?", "", s).strip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b <= a: return None
    chunk = s[a:b+1]
    try:    return json.loads(chunk)
    except: 
        try: return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
        except: return None

def uk_now():
    return datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(pytz.timezone("Europe/London"))

def between_uk(start_h: int, end_h: int) -> bool:
    now = uk_now().time()
    return (time(start_h,0) <= now < time(end_h,0)) if start_h < end_h else (now >= time(start_h,0) or now < time(end_h,0))

def is_fri_after_17() -> bool:
    n = uk_now()
    return n.weekday() == 4 and n.time() >= time(17,0)

def call_gpt(prompt: str) -> str:
    def _once(model_id: str) -> str:
        chat = openai_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role":"system","content":"You are a disciplined scalper. Reply ONLY as a valid JSON object with the keys: action, reason, confidence, categories, new_sl, new_tp, lot, session_block, god_mode_used, force_close, policy."},
                {"role":"user","content":prompt}
            ],
            max_completion_tokens=1200,
            response_format={"type":"json_object"}
        )
        return chat.choices[0].message.content or ""
    try:
        return _once(MODEL_ID)
    except Exception as e:
        logging.warning(f"Primary '{MODEL_ID}' failed → fallback '{FALLBACK_ID}': {e}")
        return _once(FALLBACK_ID)

def sanitize_decision(d: Dict[str,Any]) -> DecisionOut:
    # defaults
    d.setdefault("action", "hold")
    d.setdefault("reason", "No reason given.")
    d.setdefault("confidence", 0)
    d.setdefault("categories", [])
    d.setdefault("session_block", False)
    d.setdefault("god_mode_used", False)
    d.setdefault("force_close", False)
    d.setdefault("policy", "none")
    try:
        return DecisionOut(**d)
    except ValidationError:
        # last-resort minimal
        return DecisionOut(action="hold", reason="Validation failed.", confidence=0)

# ---------- Core endpoint ----------
@app.post("/gpt/manage")
async def manage(wrapper: TradeWrapper):
    t = wrapper.data
    ind_main = t.indicators or IndicatorsMain()
    ind_tf2  = t.tf2_indicators or IndicatorsTF2()
    pos      = t.position

    logging.info(f"🔹 Payload (trunc): {wrapper.json()[:1200]}")

    # Guard: news override
    if t.news_override:
        return JSONResponse(content=DecisionOut(
            action="hold", reason="News conflict — override active",
            confidence=0, policy="news_conflict", session_block=True
        ).model_dump())

    # Session blocks: no entries 19:00–07:00 UK; Friday 17:00+
    if between_uk(19,7):
        # allow CLOSE if already profitable; otherwise HOLD
        if pos and (pos.pnl or 0) > 0:
            return JSONResponse(content=DecisionOut(
                action="close", reason="Session policy 19:00 UK — locking profit",
                confidence=10, force_close=True, session_block=True, policy="session_exit"
            ).model_dump())
        return JSONResponse(content=DecisionOut(
            action="hold", reason="Session block 19:00–07:00 UK",
            confidence=0, session_block=True, policy="session_exit"
        ).model_dump())

    if is_fri_after_17():
        if pos and (pos.pnl or 0) > 0:
            return JSONResponse(content=DecisionOut(
                action="close", reason="Friday ≥17:00 UK — close profit before weekend",
                confidence=10, force_close=True, session_block=True, policy="friday_exit"
            ).model_dump())
        return JSONResponse(content=DecisionOut(
            action="hold", reason="Friday ≥17:00 UK — no new trades",
            confidence=0, session_block=True, policy="friday_exit"
        ).model_dump())

    # --- Build compact prompt (TF2 = trend authority) ---
    tf_names = t.tf_names or {}
    main_tf = tf_names.get("main","main")
    tf2_tf  = tf_names.get("tf2","tf2")

    prompt = f"""
You are a prop-firm scalper. Return STRICT JSON with keys:
action, reason, confidence, categories, new_sl, new_tp, lot, session_block, god_mode_used, force_close, policy.

RULES:
- Use TF2 EMA trend as the trend authority:
  * BUY only if tf2.ema_trend=="up" AND tf2.price_vs_ema=="above".
  * SELL only if tf2.ema_trend=="down" AND tf2.price_vs_ema=="below".
  * Otherwise HOLD unless god_mode_used=true (only if not strongly against).
- Include a 'Confluences:' line in reason listing up to these cats:
  ["trend","momentum","volatility","volume","structure","adx"].
- If require_sr_tp=true and you choose BUY/SELL, include "structure" in categories and set TP to next S/R (numeric).
- Always provide numeric new_sl and new_tp for BUY/SELL (≥2R or S/R-based).
- Respect spread_to_sl:
  if spread_to_sl>0.20, prefer SR-based TP or RR≥2.0; else HOLD.
- Confidence 0–10.

DATA:
symbol={t.symbol} tf_main={main_tf} tf2={tf2_tf}
update={t.update_type} signal={t.cross_signal} meaning={t.cross_meaning}
tf2_indicators={ind_tf2.dict()}
ind_main={ind_main.dict()}
spread_pips={t.spread_pips} spread_to_sl={t.spread_to_sl} require_sr_tp={bool(t.require_sr_tp)}
recent_candles_main={(t.timeframes.main or [])[-5:] if t.timeframes and t.timeframes.main else []}
recent_candles_tf2={(t.timeframes.tf2 or [])[-5:] if t.timeframes and t.timeframes.tf2 else []}
position={pos.dict() if pos else None}
"""

    # --- Call GPT ---
    raw = call_gpt(prompt)
    logging.info(f"📩 GPT raw (trunc): {raw[:800]}")
    out = extract_json_object(raw) or {"action":"hold","reason":"Decode failed","confidence":0}
    dec = sanitize_decision(out)

    # --- Lightweight server-side checks (keep short) ---
    # Enforce tf2 trend alignment for entries
    if dec.action in ("buy","sell"):
        up   = ind_tf2.ema_trend == "up"   and ind_tf2.price_vs_ema == "above"
        down = ind_tf2.ema_trend == "down" and ind_tf2.price_vs_ema == "below"
        if (dec.action == "buy" and not up) or (dec.action == "sell" and not down):
            dec.action = "hold"
            dec.reason += " | TF2 EMA trend/price misaligned."

        # Require SR-based TP when EA flagged spread risk
        if t.require_sr_tp and "structure" not in [c.lower() for c in (dec.categories or [])]:
            dec.action = "hold"
            dec.reason += " | High spread: need Structure-based TP."

        # Must include numeric SL/TP
        if dec.new_sl is None or dec.new_tp is None:
            dec.action = "hold"
            dec.reason += " | Missing new_sl/new_tp."

    return JSONResponse(content=dec.model_dump())

@app.get("/")
async def root():
    return {"message":"SmartGPT EA Server (Lite): TF2 trend, OC-candles, strict JSON, UK/news guards, spread-aware."}
