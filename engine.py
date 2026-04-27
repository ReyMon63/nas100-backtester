"""
NAS100 — Canal Apertura NY · HAL&Reymon v1.1 — Motor de Backtesting
Traducción del Pine Script v1.1 activo en TradingView (sin VPC).

CSV format : DD/MM/YYYY;HH:MM;Open;High;Low;Close;Volume
Timezone   : Chicago time (CDT=UTC-5 verano / CST=UTC-6 invierno)
NY time    : siempre Chicago + 1 hora

Instrumento: ThinkMarkets US100 CFD  →  pv = $1.00 por punto por lote
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN  —  espejo de todos los inputs del Pine v1.1 (sin VPC)
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # ── Canal de Apertura ─────────────────────────────────────────────
    rr:          float = 2.0    # Ratio Riesgo:Beneficio
    swing_on:    bool  = True   # SL Swing — activar ampliación
    swing_bars:  int   = 44     # SL Swing — lookback (44 velas × 5min = 6:00 AM NY)
    sl_buffer:   float = 0.0    # SL Buffer — margen adicional (%)

    # ── Mi Cuenta ─────────────────────────────────────────────────────
    initial_capital: float = 100_000.0  # Capital inicial en USD
    leverage:        float = 20.0       # Apalancamiento (×)
    risk_pct:        float = 1.0        # % del equity a arriesgar por operación

    # ── Indicadores base  (VPC eliminado) ─────────────────────────────
    use_ema:     bool  = True
    ema_len:     int   = 288

    use_vidya:   bool  = True
    vidya_len:   int   = 5
    vidya_mom:   int   = 10

    use_tp:      bool  = True   # Two-Pole Oscillator
    tp_double:   bool  = False  # Two-Pole peso doble (cuenta como 2 votos)
    tp_len:      int   = 30
    tp_favor:    bool  = True   # True = "A favor" → neg=LONG, pos=SHORT

    # ── TRIADA mandatoria ─────────────────────────────────────────────
    triada_on:   bool  = False

    # ── Gestión de Riesgo ─────────────────────────────────────────────
    manage_risk: bool  = False  # Breakeven en 1:1 (default OFF igual que Pine v1.1)

    # ── Cierre de Sesión (hora NY) ────────────────────────────────────
    eod_on:      bool  = True
    eod_hour:    int   = 16
    eod_min:     int   = 55

    # ── Contrato (constantes del instrumento) ─────────────────────────
    # ThinkMarkets US100 CFD: $1 por punto por lote
    point_value:     float = 1.0
    commission_pct:  float = 0.01   # % por lado
    slippage_ticks:  int   = 2      # ticks adversos en entry/stop
    tick_size:       float = 0.25   # US100 CFD: mínimo 0.25 pts


# ══════════════════════════════════════════════════════════════════════
#  CARGA Y PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath, sep=";", header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    df["dt_chicago"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M"
    )
    df["dt_chicago"] = df["dt_chicago"].dt.tz_localize(
        "America/Chicago", ambiguous="infer", nonexistent="shift_forward"
    )
    df["dt_ny"]   = df["dt_chicago"].dt.tz_convert("America/New_York")
    df["h_ny"]    = df["dt_ny"].dt.hour
    df["m_ny"]    = df["dt_ny"].dt.minute
    df["date_ny"] = df["dt_ny"].dt.date
    return df.sort_values("dt_chicago").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
#  INDICADORES
# ══════════════════════════════════════════════════════════════════════
def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha  = 1.0 / length
    vals   = series.values
    result = np.full(len(vals), np.nan)
    start  = next((k for k, v in enumerate(vals) if not np.isnan(v)), None)
    if start is None:
        return pd.Series(result, index=series.index)
    result[start] = vals[start]
    for k in range(start + 1, len(vals)):
        result[k] = (1.0 - alpha) * result[k-1] + alpha * (vals[k] if not np.isnan(vals[k]) else result[k-1])
    return pd.Series(result, index=series.index)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"]  - df["close"].shift(1)).abs()
    return _rma(pd.concat([hl, hpc, lpc], axis=1).max(axis=1), length)


def calc_ema(close: pd.Series, length: int) -> np.ndarray:
    return close.ewm(span=length, adjust=False).mean().values


def calc_vidya(close: pd.Series, vlen: int, vmom: int) -> np.ndarray:
    """VIDYA [BigBeluga] — ta.sma(vid_raw, 15)"""
    mom   = close.diff().values
    src   = close.values
    n     = len(src)
    alpha = 2.0 / (vlen + 1)

    sum_pos = np.zeros(n)
    sum_neg = np.zeros(n)
    for i in range(n):
        chunk       = mom[max(0, i - vmom + 1) : i + 1]
        sum_pos[i]  = np.nansum(np.where(chunk >= 0, chunk, 0.0))
        sum_neg[i]  = np.nansum(np.where(chunk <  0, -chunk, 0.0))

    denom   = sum_pos + sum_neg
    abs_cmo = np.where(denom != 0.0, np.abs(100.0 * (sum_pos - sum_neg) / denom), 0.0)

    vid    = np.zeros(n)
    vid[0] = src[0] if not np.isnan(src[0]) else 0.0
    for i in range(1, n):
        k      = alpha * abs_cmo[i] / 100.0
        vid[i] = k * src[i] + (1.0 - k) * vid[i-1]

    return pd.Series(vid, index=close.index).rolling(15).mean().values


def calc_vidya_plot(df: pd.DataFrame, vidya_line: np.ndarray,
                    atr_200: np.ndarray) -> np.ndarray:
    """Devuelve vidya_plot: vidya_lob si up, vidya_hib si down."""
    hib  = vidya_line + atr_200 * 2.0
    lob  = vidya_line - atr_200 * 2.0
    cl   = df["close"].values
    n    = len(cl)
    plot = np.full(n, np.nan)
    state = False
    for i in range(n):
        if np.isnan(hib[i]) or np.isnan(lob[i]):
            continue
        if i > 0 and not np.isnan(hib[i-1]) and cl[i-1] <= hib[i-1] and cl[i] > hib[i]:
            state = True
        if i > 0 and not np.isnan(lob[i-1]) and cl[i-1] >= lob[i-1] and cl[i] < lob[i]:
            state = False
        plot[i] = lob[i] if state else hib[i]
    return plot


def calc_two_pole(close: pd.Series, length: int) -> np.ndarray:
    """Two-Pole Oscillator [BigBeluga]"""
    sma25   = close.rolling(25).mean()
    det     = close - sma25
    std25   = det.rolling(25).std(ddof=1)
    sma_det = det.rolling(25).mean()
    norm    = np.where(std25 != 0.0, (det - sma_det) / std25, 0.0)

    a  = 2.0 / (length + 1)
    n  = len(norm)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    for i in range(n):
        v = norm[i]
        if np.isnan(v):
            continue
        if i == 0 or np.isnan(s1[i-1]):
            s1[i] = v;  s2[i] = v
        else:
            s1[i] = (1.0 - a) * s1[i-1] + a * v
            s2[i] = (1.0 - a) * s2[i-1] + a * s1[i]
    return s2


# ══════════════════════════════════════════════════════════════════════
#  ESTRUCTURA DE TRADE
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    entry_bar:   int
    entry_dt:    object
    direction:   str            # 'LONG' | 'SHORT'
    entry_price: float          # fill price (con slippage)
    entry_close: float          # close sin slippage (pos_entry de Pine — BE trigger)
    sl_orig:     float          # SL original en la entrada
    sl:          float          # SL actual (puede moverse a BE)
    tp:          float
    qty:         int
    tag:         str
    # Desglose del canal y SL para auditoría
    ch_hi:       float = 0.0
    ch_lo:       float = 0.0
    ch_mid:      float = 0.0
    risk_pts:    float = 0.0    # distancia SL original en puntos
    # Indicadores al momento de la entrada
    ind_ema:     float = float("nan")
    ind_vidya:   float = float("nan")
    ind_twop:    float = float("nan")
    # Resultado
    exit_bar:    Optional[int]    = None
    exit_dt:     Optional[object] = None
    exit_price:  float = 0.0
    exit_reason: str   = ""     # TP | SL | BE | EOD | END
    pnl_pts:     float = 0.0
    pnl_usd:     float = 0.0
    be_active:   bool  = False
    equity_post: float = 0.0    # equity del cuenta tras cerrar el trade


# ══════════════════════════════════════════════════════════════════════
#  TAMAÑO DE POSICIÓN  (fórmula exacta del Pine v1.1)
# ══════════════════════════════════════════════════════════════════════
def calc_qty(cfg: Config, equity: float, entry_price: float, risk_pts: float) -> int:
    """
    Pine v1.1:
        raw_qty = floor(equity × risk% / (risk_pts × pv))
        max_qty = floor(equity × leverage / (close × pv))
        qty     = min(max(1, raw_qty), max(1, max_qty))
    """
    if risk_pts <= 0:
        return 1
    pv      = cfg.point_value
    raw_qty = int(equity * cfg.risk_pct / 100.0 / (risk_pts * pv))
    max_qty = int(equity * cfg.leverage / (entry_price * pv))
    return max(1, min(raw_qty, max_qty))


# ══════════════════════════════════════════════════════════════════════
#  MOTOR DE BACKTESTING
# ══════════════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame, cfg: Config) -> list[Trade]:
    bar_940 = (df["h_ny"] == 9)  & (df["m_ny"] == 40)
    bar_eod = (df["h_ny"] == cfg.eod_hour) & (df["m_ny"] == cfg.eod_min)

    cl = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    op = df["open"].values

    # ── Indicadores ────────────────────────────────────────────────────
    ema_v   = calc_ema(df["close"], cfg.ema_len)           if cfg.use_ema   else None
    atr_200 = _atr(df, 200).values
    if cfg.use_vidya:
        vl      = calc_vidya(df["close"], cfg.vidya_len, cfg.vidya_mom)
        vidya_v = calc_vidya_plot(df, vl, atr_200)
    else:
        vidya_v = None
    twop_v  = calc_two_pole(df["close"], cfg.tp_len)       if cfg.use_tp    else None

    slippage = cfg.slippage_ticks * cfg.tick_size
    pv       = cfg.point_value

    trades:   list[Trade] = []
    pos:      Optional[Trade] = None
    equity:   float = cfg.initial_capital
    day_done: dict  = {}
    n = len(df)

    for i in range(3, n):
        date_ny = df["date_ny"].iloc[i]
        dt_ny   = df["dt_ny"].iloc[i]

        # ── 1. Actualizar breakeven (usando entry_close = pos_entry de Pine) ──
        if pos and cfg.manage_risk and not pos.be_active:
            orig_risk  = abs(pos.entry_close - pos.sl_orig)
            be_trigger = (hi[i] >= pos.entry_close + orig_risk) if pos.direction == "LONG" \
                    else (lo[i] <= pos.entry_close - orig_risk)
            if be_trigger:
                pos.be_active = True
                pos.sl        = pos.entry_close   # Pine: stop = pos_entry (close puro)

        # ── 2. Verificar SL / TP ──────────────────────────────────────────
        if pos:
            hit_tp = (hi[i] >= pos.tp) if pos.direction == "LONG" else (lo[i] <= pos.tp)
            hit_sl = (lo[i] <= pos.sl) if pos.direction == "LONG" else (hi[i] >= pos.sl)

            if hit_tp or hit_sl:
                if hit_sl:
                    reason = "BE" if pos.be_active else "SL"
                    exit_p = pos.sl
                    exit_p = exit_p - slippage if pos.direction == "LONG" else exit_p + slippage
                else:
                    reason = "TP"
                    exit_p = pos.tp

                pts  = (exit_p - pos.entry_price) if pos.direction == "LONG" \
                       else (pos.entry_price - exit_p)
                comm = (pos.entry_price + exit_p) * pv * cfg.commission_pct / 100.0 * pos.qty
                pnl  = pts * pos.qty * pv - comm

                pos.exit_bar    = i
                pos.exit_dt     = dt_ny
                pos.exit_price  = exit_p
                pos.exit_reason = reason
                pos.pnl_pts     = pts
                pos.pnl_usd     = pnl
                equity         += pnl
                pos.equity_post = equity
                trades.append(pos)
                pos = None
                continue

        # ── 3. EOD — cierre forzado ───────────────────────────────────────
        if pos and cfg.eod_on and bar_eod.iloc[i]:
            exit_p = cl[i]
            pts    = (exit_p - pos.entry_price) if pos.direction == "LONG" \
                     else (pos.entry_price - exit_p)
            comm   = (pos.entry_price + exit_p) * pv * cfg.commission_pct / 100.0 * pos.qty
            pnl    = pts * pos.qty * pv - comm

            pos.exit_bar    = i
            pos.exit_dt     = dt_ny
            pos.exit_price  = exit_p
            pos.exit_reason = "EOD"
            pos.pnl_pts     = pts
            pos.pnl_usd     = pnl
            equity         += pnl
            pos.equity_post = equity
            trades.append(pos)
            pos             = None
            day_done[date_ny] = False
            continue

        # ── 4. Evaluar barra 9:40 NY ──────────────────────────────────────
        if not bar_940.iloc[i] or day_done.get(date_ny, False) or pos:
            continue

        # ── 5. Canal (barras 9:30, 9:35, 9:40 NY) ────────────────────────
        ch_hi  = max(hi[i-2], hi[i-1], hi[i])
        ch_lo  = min(lo[i-2], lo[i-1], lo[i])
        ch_rng = ch_hi - ch_lo
        ch_mid = (ch_hi + ch_lo) / 2.0
        if ch_rng <= 0.0:
            continue

        price_now = cl[i]

        # ── 6. Condiciones de entrada ────────────────────────────────────
        tp_w   = 2 if (cfg.use_tp and cfg.tp_double) else 1
        active = (1 if cfg.use_ema else 0) + (1 if cfg.use_vidya else 0) + \
                 (tp_w if cfg.use_tp else 0)
        if active == 0:
            continue
        threshold = max(1, active - 1)

        c_l = {"ema": False, "vidya": False, "tp": False}
        c_s = {"ema": False, "vidya": False, "tp": False}
        ind_ema_v = ind_vidya_v = ind_twop_v = float("nan")

        if cfg.use_ema and ema_v is not None and not np.isnan(ema_v[i]):
            ind_ema_v  = ema_v[i]
            c_l["ema"] = ema_v[i] < price_now
            c_s["ema"] = ema_v[i] > price_now

        if cfg.use_vidya and vidya_v is not None and not np.isnan(vidya_v[i]):
            ind_vidya_v  = vidya_v[i]
            c_l["vidya"] = vidya_v[i] < price_now
            c_s["vidya"] = vidya_v[i] > price_now

        if cfg.use_tp and twop_v is not None and not np.isnan(twop_v[i]):
            ind_twop_v = twop_v[i]
            if cfg.tp_favor:
                c_l["tp"] = twop_v[i] < 0.0
                c_s["tp"] = twop_v[i] > 0.0
            else:
                c_l["tp"] = twop_v[i] > 0.0
                c_s["tp"] = twop_v[i] < 0.0

        long_score  = c_l["ema"] + c_l["vidya"] + (tp_w if c_l["tp"] else 0)
        short_score = c_s["ema"] + c_s["vidya"] + (tp_w if c_s["tp"] else 0)

        # ── 7. TRIADA ─────────────────────────────────────────────────────
        triada_long  = cl[i-2]>op[i-2] and cl[i-1]>op[i-1] and cl[i]>op[i]
        triada_short = cl[i-2]<op[i-2] and cl[i-1]<op[i-1] and cl[i]<op[i]
        hay_triada   = triada_long or triada_short

        if cfg.triada_on and hay_triada:
            enter_long  = triada_long
            enter_short = triada_short
        else:
            enter_long  = long_score  >= threshold
            enter_short = short_score >= threshold

        if not enter_long and not enter_short:
            continue

        direction = "LONG" if enter_long else "SHORT"
        score     = long_score if enter_long else short_score

        # ── 8. SL  (Pine v1.1: base = rng/2, swing solo si más protector) ─
        sl   = price_now - ch_rng/2.0 if direction == "LONG" else price_now + ch_rng/2.0
        risk = ch_rng / 2.0

        if cfg.swing_on:
            start_sw  = max(0, i - cfg.swing_bars + 1)
            swing_sl  = lo[start_sw:i+1].min() if direction == "LONG" \
                        else hi[start_sw:i+1].max()
            # Solo sobreescribe si el swing es MÁS protector que rng/2
            if (direction == "LONG"  and swing_sl < sl) or \
               (direction == "SHORT" and swing_sl > sl):
                sl   = swing_sl
                risk = abs(price_now - sl)

        if cfg.sl_buffer > 0.0:
            risk = risk * (1.0 + cfg.sl_buffer / 100.0)
            sl   = price_now - risk if direction == "LONG" else price_now + risk

        tp      = price_now + cfg.rr * risk if direction == "LONG" \
                  else price_now - cfg.rr * risk
        entry_p = price_now + slippage if direction == "LONG" else price_now - slippage

        # ── 9. Tamaño de posición (equity dinámico) ───────────────────────
        trade_qty = calc_qty(cfg, equity, price_now, risk)

        tag = f"{direction} TRIADA" if (cfg.triada_on and hay_triada) \
              else f"{direction} {score}/{active}"

        pos = Trade(
            entry_bar   = i,
            entry_dt    = dt_ny,
            direction   = direction,
            entry_price = entry_p,
            entry_close = price_now,
            sl_orig     = sl,
            sl          = sl,
            tp          = tp,
            qty         = trade_qty,
            tag         = tag,
            ch_hi       = ch_hi,
            ch_lo       = ch_lo,
            ch_mid      = ch_mid,
            risk_pts    = risk,
            ind_ema     = ind_ema_v,
            ind_vidya   = ind_vidya_v,
            ind_twop    = ind_twop_v,
        )
        day_done[date_ny] = True

    # Posición abierta al final del dataset
    if pos:
        exit_p = cl[-1]
        pts    = (exit_p - pos.entry_price) if pos.direction == "LONG" \
                 else (pos.entry_price - exit_p)
        comm   = (pos.entry_price + exit_p) * pv * cfg.commission_pct / 100.0 * pos.qty
        pnl    = pts * pos.qty * pv - comm
        pos.exit_bar    = n - 1
        pos.exit_dt     = df["dt_ny"].iloc[-1]
        pos.exit_price  = exit_p
        pos.exit_reason = "END"
        pos.pnl_pts     = pts
        pos.pnl_usd     = pnl
        equity         += pnl
        pos.equity_post = equity
        trades.append(pos)

    return trades


# ══════════════════════════════════════════════════════════════════════
#  REPORTE
# ══════════════════════════════════════════════════════════════════════
SEP  = "═" * 82
SEP2 = "─" * 82
SEP3 = "─" * 62

def _fmt(val: float) -> str:
    return f"+${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"

def print_report(trades: list[Trade], cfg: Config) -> None:
    print(f"\n{SEP}")
    print("  NAS100 — Canal Apertura NY · HAL&Reymon v1.1 · Backtester Python")
    print("  (VPC eliminado — ThinkMarkets US100 CFD  pv=$1/pt)")
    print(SEP)
    ind_str = []
    if cfg.use_ema:   ind_str.append(f"EMA({cfg.ema_len})")
    if cfg.use_vidya: ind_str.append(f"VIDYA({cfg.vidya_len}/{cfg.vidya_mom})")
    if cfg.use_tp:    ind_str.append(f"2-Pole({cfg.tp_len})" + (" ×2" if cfg.tp_double else ""))
    print(f"  Indicadores : {' | '.join(ind_str)}")
    print(f"  Config      : RR={cfg.rr}  Swing={cfg.swing_on}/{cfg.swing_bars}v  "
          f"Buffer={cfg.sl_buffer}%  TRIADA={cfg.triada_on}  BE={cfg.manage_risk}")
    print(f"  Cuenta      : ${cfg.initial_capital:,.0f}  ×{cfg.leverage}  "
          f"Riesgo={cfg.risk_pct}%/op  pv=${cfg.point_value}/pt")

    if not trades:
        print("\n  Sin operaciones en el período.")
        print(SEP + "\n")
        return

    print(f"\n{SEP2}")
    print("  DETALLE POR OPERACIÓN")
    print(SEP2)

    for k, t in enumerate(trades, 1):
        ed      = t.entry_dt.strftime("%Y-%m-%d %H:%M NY") if t.entry_dt else "—"
        xd      = t.exit_dt.strftime("%Y-%m-%d %H:%M NY") if t.exit_dt  else "—"
        be_note = "  → BE en " + f"{t.entry_close:.2f}" if t.be_active else ""
        be_tag  = " [BE]" if t.be_active else ""

        risk_usd   = t.risk_pts   * t.qty * cfg.point_value
        reward_usd = abs(t.tp - t.entry_close) * t.qty * cfg.point_value

        print(f"\n  [{k}]  {ed}  ·  {t.tag}")
        print(f"       Canal  :  High={t.ch_hi:.2f}  Low={t.ch_lo:.2f}  "
              f"Mid={t.ch_mid:.2f}  Rng={t.ch_hi - t.ch_lo:.2f}")

        if not math.isnan(t.ind_ema):
            lbl = "bajo close" if t.ind_ema < t.entry_close else "sobre close"
            print(f"       EMA({cfg.ema_len})   :  {t.ind_ema:.2f}  [{lbl}]")
        if not math.isnan(t.ind_vidya):
            lbl = "bajo close" if t.ind_vidya < t.entry_close else "sobre close"
            print(f"       VIDYA-plot :  {t.ind_vidya:.2f}  [{lbl}]")
        if not math.isnan(t.ind_twop):
            lbl = "→ LONG" if t.ind_twop < 0 else "→ SHORT"
            print(f"       Two-Pole   :  {t.ind_twop:.6f}  [{lbl}]")

        print(f"       {SEP3}")
        print(f"       ENTRADA  :  {t.entry_price:.2f}  (fill con slippage)")
        print(f"       SL orig  :  {t.sl_orig:.2f}  "
              f"({t.risk_pts:.2f} pts · ${t.risk_pts*cfg.point_value:.2f}/lote){be_note}")
        print(f"       TP       :  {t.tp:.2f}  "
              f"({abs(t.tp - t.entry_close):.2f} pts · ${abs(t.tp-t.entry_close)*cfg.point_value:.2f}/lote)")
        print(f"       Lotes    :  {t.qty}  "
              f"[equity ${t.equity_post - t.pnl_usd:,.0f}  ·  "
              f"riesgo ${risk_usd:,.0f} ({risk_usd/(t.equity_post-t.pnl_usd)*100:.2f}%)  ·  "
              f"objetivo ${reward_usd:,.0f}]")
        print(f"       {SEP3}")
        print(f"       CIERRE   :  {xd}  @{t.exit_price:.2f}  "
              f"({t.exit_reason}{be_tag})  "
              f"P&L = {_fmt(t.pnl_usd)}  ({t.pnl_pts:+.2f} pts)  "
              f"→  Equity ${t.equity_post:,.2f}")

    # ── Resumen ────────────────────────────────────────────────────────
    total_usd  = sum(t.pnl_usd for t in trades)
    total_pts  = sum(t.pnl_pts for t in trades)
    wins       = [t for t in trades if t.pnl_usd > 0]
    losses     = [t for t in trades if t.pnl_usd <= 0]
    n          = len(trades)
    wr         = len(wins) / n * 100 if n else 0.0
    avg_win    = sum(t.pnl_usd for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss   = sum(t.pnl_usd for t in losses) / len(losses) if losses else 0.0

    print(f"\n{SEP2}")
    print(f"  Operaciones  :  {n}  "
          f"(Ganadoras {len(wins)}  |  Perdedoras {len(losses)}  |  WR {wr:.1f}%)")
    print(f"  Prom. ganada :  {_fmt(avg_win)}")
    print(f"  Prom. perdida:  {_fmt(avg_loss)}")
    print(f"  P&L total    :  {_fmt(total_usd)}  ({total_pts:+.2f} pts)")
    print(f"  Capital ini. :  ${cfg.initial_capital:>10,.2f}")
    print(f"  Capital fin. :  ${cfg.initial_capital + total_usd:>10,.2f}")
    print(f"  Retorno      :  {total_usd / cfg.initial_capital * 100:+.2f}%")
    print(SEP + "\n")


# ══════════════════════════════════════════════════════════════════════
#  MAIN  —  modifica esta sección para cambiar parámetros
