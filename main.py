# main.py — GPT decision server (actions only; EA manages SL/TP)
from __future__ import annotations
from typing import Any, Dict, Optional, Literal, Tuple
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, time, logging

# ---------- OpenAI ----------
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # gpt-4o / gpt-4o-mini
OAI = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Server knobs (ENV overrides) ----------
SYMBOL_ALLOW      = set(filter(None, (os.getenv("SYMBOL_ALLOW", "XAUUSD,XAUUSDECN,XAUUSD.M,XAUUSD_I").split(","))))
HARD_STOP_R       = float(os.getenv("HARD_STOP_R_MULT", "1.30"))   # emergency close if r_now <= -1.30
MIN_ADX_SHORT     = float(os.getenv("MIN_ADX_SHORT", "24.0"))      # shorts need stronger ADX
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO").upper()

# Debug toggles (surface payload + raw GPT JSON in telemetry)
ECHO_PAYLOAD      = os.getenv("ECHO_PAYLOAD", "1").lower() in ("1","true","yes","y","on")
ECHO_GPT_RAW      = os.getenv("ECHO_GPT_RAW", "1").lower() in ("1","true","yes","y","on")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gpt-trade-server")

# ---------- FastAPI ----------
app = FastAPI(title="GM GPT Trade Server (Actions Only)", version="5.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

# ---------- Schemas ----------
class DecisionOut(BaseModel):
    action: Literal["open_long","open_short","close","hold"]
    reason: str
    align: Optional[bool] = None
    confidence: Optional[float] = None
    # Rich introspection so you see EXACTLY why it held and what’s needed to approve:
    veto_reasons: Optional[list[str]] = None   # concrete blockers hit
    requirements: Optional[dict] = None        # what would satisfy an OPEN right now
    telemetry: Optional[dict] = None           # snapshot of key inputs (debugging)

# ---------- Small helpers ----------
def getf(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    try: return float(d.get(k, default))
    except Exception: return default

def geti(d: Dict[str, Any], k: str, default: int = 0) -> int:
    try: return int(d.get(k, default))
    except Exception: return default

def getb(d: Dict[str, Any], k: str, default: bool = False) -> bool:
    v = d.get(k, default)
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return v != 0
    if isinstance(v, str): return v.strip().lower() in ("true","1","yes","y","t","on")
    return bool(v)

def position_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = payload.get("position") or {}
    return {
        "has": getb(p, "has", False),
        "dir": (p.get("dir") or "").lower() if isinstance(p.get("dir"), str) else None,
        "volume": getf(p, "volume", 0.0),
        "entry": getf(p, "entry", 0.0),
        "r_now": getf(p, "r_now", 0.0),
        "unrealized": getf(p, "unrealized", 0.0),
        "partial_done": getb(p, "partial_done", False),
    }

def guards_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    g = payload.get("guards") or {}
    if not g:
        # Back-compat: accept flat keys if EA sent older payload
        g = {
            "spread_ok": getb(payload, "max_spread_pips_ok", True),
            "dd_locked": getb(payload, "daily_dd_locked", False),
            "active_window": True,
            "news_block": False,
        }
    return {
        "spread_ok": getb(g, "spread_ok", True),
        "dd_locked": getb(g, "dd_locked", False),
        "active_window": getb(g, "active_window", True),
        "news_block": getb(g, "news_block", False),
    }

def constraints_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    c = payload.get("constraints") or {}
    return {
        "require_align": getb(c, "require_align", True),
        "min_confidence": float(c.get("min_confidence", 0.60)),  # slightly friendlier default
    }

def readiness_from(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = payload.get("readiness") or {}
    # Also accept common synonyms from flat payloads
    flat = payload
    def any_true(d, *keys):
        for k in keys:
            v = d.get(k)
            if isinstance(v, bool) and v: return True
            if isinstance(v, (int,float)) and v != 0: return True
        return False
    return {
        "pullback_long":  any_true(r,"pullback_long","pullback_long_ready")  or any_true(flat,"pullback_long_ready","pullbackReadyLong","long_pullback_ready"),
        "breakout_long":  any_true(r,"breakout_long","breakout_long_ready")  or any_true(flat,"breakout_long_ready","breakoutReadyLong","long_breakout_ready"),
        "orb_long":       any_true(r,"orb_long","orb_long_ready")            or any_true(flat,"orb_long_ready","orbReadyLong","long_orb_ready"),
        "avwap_long":     any_true(r,"avwap_long","avwap_long_ready")        or any_true(flat,"avwap_long_ready","avwapReadyLong","long_avwap_ready"),
        "pullback_short": any_true(r,"pullback_short","pullback_short_ready")or any_true(flat,"pullback_short_ready","pullbackReadyShort","short_pullback_ready"),
        "breakout_short": any_true(r,"breakout_short","breakout_short_ready")or any_true(flat,"breakout_short_ready","breakoutReadyShort","short_breakout_ready"),
        "orb_short":      any_true(r,"orb_short","orb_short_ready")          or any_true(flat,"orb_short_ready","orbReadyShort","short_orb_ready"),
        "avwap_short":    any_true(r,"avwap_short","avwap_short_ready")      or any_true(flat,"avwap_short_ready","avwapReadyShort","short_avwap_ready"),
    }

def safe_parse_json(txt: str) -> Optional[dict]:
    try:
        return json.loads(txt)
    except Exception:
        try:
            return json.loads(txt.replace(",}", "}").replace(",]", "]"))
        except Exception:
            return None

# ---------- Event side + flags ----------
def trigger_side(payload: Dict[str, Any]) -> Optional[str]:
    t = (payload.get("trigger") or "").lower()
    if t.startswith("long_"):  return "long"
    if t.startswith("short_"): return "short"
    return None

def align_flags_from(payload: Dict[str, Any]) -> Tuple[bool, bool]:
    af = payload.get("align_flags") or {}
    return bool(af.get("long_ok", False)), bool(af.get("short_ok", False))

# ---------- Requirements builder (what would have approved the trade) ----------
def side_from_action(a: str) -> Optional[str]:
    if a == "open_long": return "long"
    if a == "open_short": return "short"
    return None

def build_requirements(payload: Dict[str, Any], proposed: str, conf_now: float) -> dict:
    side = side_from_action(proposed) or trigger_side(payload) or "long"
    g = guards_from(payload)
    cons = constraints_from(payload)
    r = readiness_from(payload)
    allow_longs  = getb(payload, "allow_longs", True)
    allow_shorts = getb(payload, "allow_shorts", False)
    long_ok_flag, short_ok_flag = align_flags_from(payload)
    adx = getf(payload, "adx", 0.0)
    bbw = getf(payload, "bb_width", 0.0)
    ema = getf(payload, "ema_sep_pct", 0.0)
    regime = (payload.get("regime") or "").lower()

    need = []
    targets = {}

    # environment
    if not g["spread_ok"]:     need.append("spread_ok");           targets["spread_ok"] = True
    if g["dd_locked"]:         need.append("dd_unlocked");         targets["dd_locked"] = False
    if not g["active_window"]: need.append("active_window");       targets["active_window"] = True
    if g["news_block"]:        need.append("news_window_clear");   targets["news_block"] = False

    # permission
    if side == "long" and not allow_longs:
        need.append("allow_longs"); targets["allow_longs"] = True
    if side == "short" and not allow_shorts:
        need.append("allow_shorts"); targets["allow_shorts"] = True

    # alignment (either align_flag or readiness)
    if side == "long":
        if not (long_ok_flag or any([getb(r,"pullback_long"), getb(r,"breakout_long"), getb(r,"orb_long"), getb(r,"avwap_long")])):
            need.append("long_setup_ready")
    else:
        if not (short_ok_flag or any([getb(r,"pullback_short"), getb(r,"breakout_short"), getb(r,"orb_short"), getb(r,"avwap_short")])):
            need.append("short_setup_ready")

    # confidence
    if cons.get("min_confidence", 0.60) > conf_now:
        need.append("confidence")
        targets["min_confidence_required"] = cons["min_confidence"]
        targets["confidence_now"] = round(conf_now, 3)
        targets["confidence_drivers"] = {
            "adx_now": adx, "bbw_now": bbw, "ema_sep_pct_now": ema,
            "regime_trend_or_breakout": regime in ("trend","breakout")
        }

    # short-specific strength
    if side == "short" and adx < MIN_ADX_SHORT:
        need.append("adx_for_short")
        targets["adx_min_short"] = MIN_ADX_SHORT
        targets["adx_now"] = adx

    return {"side": side, "need": need, "targets": targets}

# ---------- Alignment + confidence ----------
def compute_alignment_and_confidence(payload: Dict[str, Any], proposed: str) -> Tuple[bool, float, str, list[str]]:
    """Return (align_ok, confidence_0_1, note, veto_reasons[])"""
    veto = []
    r = readiness_from(payload)
    g = guards_from(payload)
    allow_longs  = getb(payload, "allow_longs", True)
    allow_shorts = getb(payload, "allow_shorts", False)

    adx = getf(payload, "adx", 0.0)
    bbw = getf(payload, "bb_width", 0.0)
    ema_sep = getf(payload, "ema_sep_pct", 0.0)
    regime = (payload.get("regime") or "").lower()

    long_ok_flag, short_ok_flag = align_flags_from(payload)
    trig_side = trigger_side(payload)

    # base confidence
    adx_score = min(1.0, max(0.0, (adx - 10.0) / 20.0))     # ~0 at 10, ~1 at 30
    bbw_score = min(1.0, max(0.0, (bbw - 0.02) / 0.06))     # expansion helps
    ema_score = min(1.0, max(0.0, abs(ema_sep) / 0.25))     # separation helps
    base_conf = 0.25*adx_score + 0.25*bbw_score + 0.25*ema_score + 0.25*(1.0 if regime in ("trend","breakout") else 0.0)

    # environment
    if not g["spread_ok"]:     veto.append("spread_ok=false")
    if g["dd_locked"]:         veto.append("dd_locked=true")
    if not g["active_window"]: veto.append("active_window=false")
    if g["news_block"]:        veto.append("news_block=true")
    if veto:
        return (False, 0.0, "Environment block", veto)

    # permissions
    if proposed == "open_long" and not allow_longs:
        veto.append("allow_longs=false")
    if proposed == "open_short" and not allow_shorts:
        veto.append("allow_shorts=false")
    if veto:
        return (False, base_conf*0.5, "Side permission blocked", veto)

    # keep GPT on the EA event side
    if trig_side == "long" and proposed == "open_short":
        veto.append("opposite_to_event_long")
        return (False, base_conf*0.6, "Opposite to event", veto)
    if trig_side == "short" and proposed == "open_long":
        veto.append("opposite_to_event_short")
        return (False, base_conf*0.6, "Opposite to event", veto)

    # alignment by flags/votes
    long_vote  = long_ok_flag  or any([getb(r,"pullback_long"), getb(r,"breakout_long"), getb(r,"orb_long"), getb(r,"avwap_long")])
    short_vote = short_ok_flag or any([getb(r,"pullback_short"), getb(r,"breakout_short"), getb(r,"orb_short"), getb(r,"avwap_short")])
    if proposed == "open_long" and not long_vote:
        veto.append("no_long_setup_ready")
        return (False, base_conf*0.6, "No long setup", veto)
    if proposed == "open_short" and not short_vote:
        veto.append("no_short_setup_ready")
        return (False, base_conf*0.6, "No short setup", veto)

    # shorts need extra juice
    if proposed == "open_short":
        base_conf *= 0.9
        if adx < MIN_ADX_SHORT:
            veto.append(f"adx_short<{MIN_ADX_SHORT}")
            return (False, base_conf*0.7, "Weak ADX for short", veto)

    # close/hold are always aligned
    if proposed in ("close","hold"):
        return (True, 0.8 if proposed=="close" else 0.6, "Non-open action", veto)

    return (True, max(0.0, min(1.0, base_conf)), "Aligned", veto)

# ---------- Emergency guard ----------
def hard_guards(payload: Dict[str, Any]) -> Optional[DecisionOut]:
    sym = (payload.get("symbol") or "").upper()
    if SYMBOL_ALLOW and all(s.upper() != sym for s in SYMBOL_ALLOW):
        return DecisionOut(action="hold", reason=f"Symbol not allowed: {sym}", align=False, confidence=0.0,
                           veto_reasons=["symbol_not_allowed"], requirements=None)
    g = guards_from(payload)
    if not g["spread_ok"]:
        return DecisionOut(action="hold", reason="Spread filter (EA)", align=False, confidence=0.0,
                           veto_reasons=["spread"], requirements=None)
    if g["dd_locked"]:
        pos = position_from(payload)
        if pos["has"]:
            return DecisionOut(action="close", reason="Daily DD lock (EA) — closing", align=True, confidence=1.0)
        return DecisionOut(action="hold", reason="Daily DD lock (EA) — flat", align=True, confidence=1.0)
    if g["news_block"]:
        pos = position_from(payload)
        if not pos["has"]:
            return DecisionOut(action="hold", reason="News window — no new trades", align=False, confidence=0.0,
                               veto_reasons=["news_window"], requirements=None)
    return None

def last_resort_management(payload: Dict[str, Any]) -> Optional[DecisionOut]:
    pos = position_from(payload)
    if pos["has"] and pos["r_now"] <= -HARD_STOP_R:
        return DecisionOut(action="close", reason=f"Hard stop: r_now={pos['r_now']:.2f}", align=True, confidence=1.0)
    return None

# ---------- Prompt building ----------
SYSTEM_PROMPT = (
    "You are a disciplined, risk-aware trading decision engine for XAUUSD.\n"
    "Return STRICT JSON ONLY. Keys:\n"
    '  action: \"open_long\" | \"open_short\" | \"close\" | \"hold\"\n'
    "  reason: concise evidence-based explanation\n"
    "  (optional) requirements: if not opening, list what would need to change "
    "(e.g., long_setup_ready=true, confidence>=0.65, ADX>=24 for short).\n"
    "- Respect EA guards and the EA's event side (trigger).\n"
    "- Shorts need stronger ADX.\n"
    "- Do NOT suggest SL/TP; EA manages them.\n"
)

def build_user_payload_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Keep payload compact (include only relevant bits)
    candles_txt = "null"
    if isinstance(payload.get("candles"), list) and payload["candles"]:
        try:
            last = payload["candles"][-8:]
            slim = [{"t": c.get("t"), "o": c.get("o"), "h": c.get("h"), "l": c.get("l"), "c": c.get("c")} for c in last]
            candles_txt = json.dumps(slim, separators=(",", ":"))
        except Exception:
            candles_txt = "null"

    return {
        "symbol": payload.get("symbol"),
        "server_time": payload.get("server_time"),
        "trigger": payload.get("trigger"),
        "regime": payload.get("regime"),
        "bias_dir": payload.get("bias_dir"),
        "adx": payload.get("adx"),
        "bb_width": payload.get("bb_width"),
        "ema_sep_pct": payload.get("ema_sep_pct"),
        "lrc_angle_deg": payload.get("lrc_angle_deg"),
        "readiness": readiness_from(payload),
        "market": payload.get("market") or {"bid": None, "ask": None, "atr": None},
        "position": payload.get("position"),
        "guards": guards_from(payload),
        "constraints": constraints_from(payload),
        "allow_longs": getb(payload, "allow_longs", True),
        "allow_shorts": getb(payload, "allow_shorts", False),
        "align_flags": payload.get("align_flags") or {},
        "candles_tail": candles_txt,
    }

def build_user_prompt(payload: Dict[str, Any]) -> str:
    return json.dumps(build_user_payload_dict(payload), separators=(",", ":"))

# ---------- GPT call ----------
def call_gpt(payload: Dict[str, Any]) -> tuple[Optional[dict], Optional[str]]:
    try:
        user_content = build_user_prompt(payload)
        rsp = OAI.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.15,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        txt = (rsp.choices[0].message.content or "").strip()
        log.debug(f"GPT raw: {txt[:1200]}")
        return safe_parse_json(txt), txt
    except Exception as e:
        log.error(f"OpenAI error: {e}")
        return None, None

# ---------- Endpoints ----------
@app.post("/gpt/manage", response_model=DecisionOut)
async def gpt_manage(req: Request):
    try:
        payload: Dict[str, Any] = await req.json()
    except Exception:
        raw = await req.body()
        log.warning(f"Non-JSON body received: {raw[:200]!r}")
        return DecisionOut(action="hold", reason="Bad request body", align=False, confidence=0.0,
                           veto_reasons=["bad_body"], requirements=None)

    log.debug(f"payload: {json.dumps(payload)[:1500]}")

    # 1) Hard guards (spread/DD/news/symbol)
    hg = hard_guards(payload)
    if hg is not None:
        return hg

    # 2) Emergency risk guard if already in a trade (does not touch SL/TP)
    lr = last_resort_management(payload)
    if lr is not None:
        return lr

    # 3) GPT proposes
    g, gpt_raw = call_gpt(payload)
    user_payload_dict = build_user_payload_dict(payload)  # for echo

    if not isinstance(g, dict):
        return DecisionOut(
            action="hold", reason="Model returned no/invalid JSON.",
            align=False, confidence=0.0,
            veto_reasons=["model_invalid"],
            requirements=build_requirements(payload, "open_long", 0.0),
            telemetry={
                "payload_keys": list(payload.keys())[:20],
                **({"payload_echo": user_payload_dict} if ECHO_PAYLOAD else {}),
                **({"gpt_raw": gpt_raw} if (ECHO_GPT_RAW and gpt_raw) else {})
            }
        )

    proposed = str(g.get("action", "hold")).strip().lower()
    if proposed not in ("open_long","open_short","close","hold"):
        proposed = "hold"

    # 4) Alignment + confidence
    align_ok, conf, note, veto = compute_alignment_and_confidence(payload, proposed)
    cons = constraints_from(payload)

    # Alignment required but failed -> HOLD with requirements
    if cons["require_align"] and not align_ok:
        return DecisionOut(
            action="hold", reason=f"Alignment veto: {note}",
            align=False, confidence=conf,
            veto_reasons=veto or ["alignment_false"],
            requirements=build_requirements(payload, proposed, conf),
            telemetry={
                "proposed": proposed,
                "adx": getf(payload,"adx",0.0),
                "bb_width": getf(payload,"bb_width",0.0),
                "ema_sep_pct": getf(payload,"ema_sep_pct",0.0),
                "regime": payload.get("regime"),
                "allow_longs": getb(payload,"allow_longs",True),
                "allow_shorts": getb(payload,"allow_shorts",False),
                "align_flags": payload.get("align_flags"),
                **({"payload_echo": user_payload_dict} if ECHO_PAYLOAD else {}),
                **({"gpt_raw": gpt_raw} if (ECHO_GPT_RAW and gpt_raw) else {})
            }
        )

    # Confidence gate
    if proposed in ("open_long","open_short") and conf < cons["min_confidence"]:
        return DecisionOut(
            action="hold",
            reason=f"Confidence {conf:.2f} < min {cons['min_confidence']:.2f}",
            align=align_ok, confidence=conf,
            veto_reasons=["confidence_below_min"],
            requirements=build_requirements(payload, proposed, conf),
            telemetry={
                "proposed": proposed,
                "min_conf": cons["min_confidence"],
                "conf_now": conf,
                **({"payload_echo": user_payload_dict} if ECHO_PAYLOAD else {}),
                **({"gpt_raw": gpt_raw} if (ECHO_GPT_RAW and gpt_raw) else {})
            }
        )

    # Shorts: additional ADX requirement
    if proposed == "open_short":
        adx = getf(payload, "adx", 0.0)
        if adx < MIN_ADX_SHORT:
            return DecisionOut(
                action="hold",
                reason=f"Veto weak short (ADX {adx:.1f}<{MIN_ADX_SHORT})",
                align=False, confidence=conf,
                veto_reasons=[f"adx_short<{MIN_ADX_SHORT}"],
                requirements=build_requirements(payload, proposed, conf),
                telemetry={
                    "adx": adx,
                    **({"payload_echo": user_payload_dict} if ECHO_PAYLOAD else {}),
                    **({"gpt_raw": gpt_raw} if (ECHO_GPT_RAW and gpt_raw) else {})
                }
            )

    # 5) Pass-through approval (EA executes)
    reason = str(g.get("reason", "")).strip() or note
    return DecisionOut(
        action=proposed, reason=reason,
        align=align_ok, confidence=conf,
        veto_reasons=[],
        requirements=None,
        telemetry={
            "approved": True,
            **({"payload_echo": user_payload_dict} if ECHO_PAYLOAD else {}),
            **({"gpt_raw": gpt_raw} if (ECHO_GPT_RAW and gpt_raw) else {})
        }
    )

# Quick diagnostics to see what the server thinks of your payload
@app.post("/diag")
async def diag(req: Request):
    payload = await req.json()
    proposed = "open_short" if getb(payload,"test_short",False) else "open_long"
    align_ok, conf, note, veto = compute_alignment_and_confidence(payload, proposed)
    return {
        "symbol": payload.get("symbol"),
        "trigger": payload.get("trigger"),
        "guards": guards_from(payload),
        "constraints": constraints_from(payload),
        "readiness": readiness_from(payload),
        "allow_longs": getb(payload,"allow_longs",True),
        "allow_shorts": getb(payload,"allow_shorts",False),
        "align_flags": payload.get("align_flags"),
        "proposed_probe": proposed,
        "align_ok": align_ok, "confidence": conf, "note": note, "veto": veto,
        "requirements": build_requirements(payload, proposed, conf)
    }

@app.get("/")
async def root():
    return {"ok": True, "msg": "GM GPT Trade Server — Actions only (EA owns SL/TP)."}
