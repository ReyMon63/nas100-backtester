"""
NAS100 — Canal Apertura NY · HAL&Reymon v1.1
Backtester App — Streamlit
"""

import io
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from engine import Config, load_data, run_backtest

# ── Página ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAS100 Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Botón "Ejecutar" fijo en la parte superior del sidebar ── */
section[data-testid="stSidebar"]
  [data-testid="stSidebarUserContent"]
  .stVerticalBlock > div:first-child {
    position: sticky;
    top: 0;
    z-index: 200;
    background-color: #0e1117;
    padding: 10px 0 12px;
    border-bottom: 1px solid #2d3548;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 NAS100 — Canal Apertura NY · HAL&Reymon v1.1")
st.caption("Backtester · ThinkMarkets US100 CFD · $1/pt por lote · Sin VPC")

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — botón primero (sticky), luego parámetros (scroll)
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── BOTÓN — primer elemento → CSS lo hace sticky ──────────────────
    run_btn = st.button("▶ Ejecutar Backtest", type="primary", use_container_width=True)

    st.header("⚙️ Parámetros")

    # ── Datos ──────────────────────────────────────────────────────────
    st.subheader("📂 Datos")
    uploaded = st.file_uploader(
        "Subir CSV (DD/MM/YYYY;HH:MM;O;H;L;C;V — hora Chicago)",
        type=["csv", "txt"],
    )

    # ── Mi Cuenta ──────────────────────────────────────────────────────
    st.subheader("💰 Mi Cuenta")
    initial_capital = st.number_input(
        "Capital inicial (USD)", min_value=1_000, max_value=10_000_000,
        value=100_000, step=1_000,
    )
    leverage = st.number_input(
        "Apalancamiento (×)", min_value=2, max_value=200, value=20,
        help="ThinkMarkets US100 CFD → 20×",
    )
    risk_pct = st.slider(
        "% Capital a arriesgar / op", min_value=0.1, max_value=10.0,
        value=1.0, step=0.1,
        help="1% de $100,000 = $1,000 de riesgo máximo por operación",
    )

    # ── Canal ──────────────────────────────────────────────────────────
    st.subheader("📐 Canal de Apertura")
    rr = st.select_slider(
        "Ratio Riesgo : Beneficio",
        options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        value=2.0,
    )
    swing_on = st.checkbox("SL Swing — ampliar al swing previo", value=True)
    swing_bars = st.number_input(
        "SL Swing — lookback (velas × 5min)",
        min_value=5, max_value=300, value=44, step=5,
        disabled=not swing_on,
        help="44 velas × 5min = 6:00 AM NY",
    )
    sl_buffer = st.slider(
        "SL Buffer — margen adicional (%)",
        min_value=0.0, max_value=50.0, value=0.0, step=0.5,
    )

    # ── Indicadores ────────────────────────────────────────────────────
    st.subheader("📊 Indicadores Base")

    col1, col2 = st.columns([1, 2])
    with col1:
        use_ema = st.checkbox("EMA", value=True)
    with col2:
        ema_len = st.number_input(
            "Periodo", min_value=1, max_value=500, value=288,
            key="ema_len", disabled=not use_ema, label_visibility="collapsed",
        )

    col1, col2, col3 = st.columns([1, 1.5, 1.5])
    with col1:
        use_vidya = st.checkbox("VIDYA", value=True)
    with col2:
        vidya_len = st.number_input(
            "Long.", min_value=1, max_value=50, value=5,
            key="vidya_len", disabled=not use_vidya, label_visibility="collapsed",
        )
    with col3:
        vidya_mom = st.number_input(
            "Mom.", min_value=1, max_value=50, value=10,
            key="vidya_mom", disabled=not use_vidya, label_visibility="collapsed",
        )

    col1, col2 = st.columns([1, 2])
    with col1:
        use_tp = st.checkbox("Two-Pole", value=True)
    with col2:
        tp_len = st.number_input(
            "Long.", min_value=1, max_value=100, value=30,
            key="tp_len", disabled=not use_tp, label_visibility="collapsed",
        )

    tp_double = st.checkbox(
        "Two-Pole peso doble (2 votos)", value=False, disabled=not use_tp,
        help="Útil cuando VPC y VIDYA están correlacionados",
    )
    tp_favor = st.radio(
        "Two-Pole dirección",
        ["A favor  (neg=LONG · pos=SHORT)", "En contra (pos=LONG · neg=SHORT)"],
        index=0, disabled=not use_tp,
    )

    # ── Filtros ────────────────────────────────────────────────────────
    st.subheader("🔀 Filtros")
    triada_on = st.checkbox(
        "TRIADA mandatoria",
        value=False,
        help="Las 3 velas del canal del mismo color sobreescriben el umbral N-1/N",
    )
    manage_risk = st.checkbox(
        "Gestión de riesgo — Breakeven 1:1",
        value=False,
        help="Mueve el SL a la entrada cuando el precio avanza 1R",
    )

    # ── Cierre de sesión ───────────────────────────────────────────────
    st.subheader("⏰ Cierre de Sesión")
    eod_on = st.checkbox("Cierre automático al EOD", value=True)
    col1, col2 = st.columns(2)
    with col1:
        eod_hour = st.number_input(
            "Hora NY", min_value=11, max_value=23, value=16, disabled=not eod_on,
        )
    with col2:
        eod_min = st.select_slider(
            "Min NY", options=list(range(0, 60, 5)), value=55, disabled=not eod_on,
        )

# ══════════════════════════════════════════════════════════════════════
#  EJECUTAR
# ══════════════════════════════════════════════════════════════════════
if run_btn:
    if uploaded is None:
        st.warning("Sube un archivo CSV primero.")
        st.stop()

    with st.spinner("Cargando datos y ejecutando backtest…"):
        try:
            df = load_data(uploaded)
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")
            st.stop()

        cfg = Config(
            rr              = float(rr),
            swing_on        = swing_on,
            swing_bars      = int(swing_bars),
            sl_buffer       = float(sl_buffer),
            initial_capital = float(initial_capital),
            leverage        = float(leverage),
            risk_pct        = float(risk_pct),
            use_ema         = use_ema,
            ema_len         = int(ema_len),
            use_vidya       = use_vidya,
            vidya_len       = int(vidya_len),
            vidya_mom       = int(vidya_mom),
            use_tp          = use_tp,
            tp_double       = tp_double,
            tp_len          = int(tp_len),
            tp_favor        = (tp_favor.startswith("A favor")),
            triada_on       = triada_on,
            manage_risk     = manage_risk,
            eod_on          = eod_on,
            eod_hour        = int(eod_hour),
            eod_min         = int(eod_min),
        )

        trades = run_backtest(df, cfg)
        st.session_state["trades"]  = trades
        st.session_state["cfg"]     = cfg
        st.session_state["tbl_page"] = 0   # reinicia paginación

    if not trades:
        st.info("No se encontraron operaciones en el período cargado.")
        st.stop()

# Recuperar trades previos o salir con pantalla inicial
if "trades" not in st.session_state or not st.session_state["trades"]:
    st.info(
        "👈 **Sube tu CSV** en el panel lateral y ajusta los parámetros, "
        "luego presiona **▶ Ejecutar Backtest**.\n\n"
        "**Formato del CSV:**  `DD/MM/YYYY;HH:MM;Open;High;Low;Close;Volume`  "
        "— hora en tiempo de Chicago (CDT/CST)."
    )
    st.image(
        "https://img.shields.io/badge/NAS100-HAL%26Reymon%20v1.1-00d4aa?style=for-the-badge",
        width=300,
    )
    st.stop()

trades = st.session_state["trades"]
cfg    = st.session_state["cfg"]

# ══════════════════════════════════════════════════════════════════════
#  MÉTRICAS RESUMEN
# ══════════════════════════════════════════════════════════════════════
total_usd = sum(t.pnl_usd for t in trades)
total_pts = sum(t.pnl_pts for t in trades)
n         = len(trades)

# ── Clasificación ─────────────────────────────────────────────────────
# BE  → exit_reason == "BE"  (independiente del monto)
# win → P&L > 0 y no BE      (incluye TP ganador y EOD ganador)
# loss→ P&L < 0 y no BE      (incluye SL perdedor y EOD perdedor)
# EOD → contador aparte; cada EOD ya está dentro de win/loss según P&L
bes    = [t for t in trades if t.exit_reason == "BE"]
wins   = [t for t in trades if t.exit_reason != "BE" and t.pnl_usd > 0]
losses = [t for t in trades if t.exit_reason != "BE" and t.pnl_usd < 0]
eods   = [t for t in trades if t.exit_reason == "EOD"]
eod_w  = [t for t in eods   if t.pnl_usd > 0]
eod_l  = [t for t in eods   if t.pnl_usd < 0]

wr       = len(wins) / n * 100 if n else 0.0
no_loss  = (len(wins) + len(bes)) / n * 100 if n else 0.0
avg_win  = sum(t.pnl_usd for t in wins)   / len(wins)   if wins   else 0.0
avg_loss = sum(t.pnl_usd for t in losses) / len(losses) if losses else 0.0
pf_denom = sum(t.pnl_usd for t in losses)
pf       = abs(sum(t.pnl_usd for t in wins) / pf_denom) \
           if losses and pf_denom != 0 else float("inf")

st.subheader("📋 Resumen")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Operaciones",   n)
c2.metric("Win Rate",      f"{wr:.1f}%",
          help="Ganadoras (TP + EOD+) sobre el total")
c3.metric("P&L Total",     f"${total_usd:+,.2f}")
c4.metric("Retorno",       f"{total_usd/initial_capital*100:+.2f}%")
c5.metric("Profit Factor", f"{pf:.2f}" if pf != float("inf") else "∞",
          help="Ganancias brutas / Pérdidas brutas (excluye BE)")
c6.metric("Sin pérdida",   f"{no_loss:.1f}%",
          help="(Ganadoras + BE) ÷ Total")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Ganadoras",    len(wins),
          help=f"TP: {len(wins)-len(eod_w)}  |  EOD+: {len(eod_w)}")
c2.metric("BE",           len(bes))
c3.metric("Perdedoras",   len(losses),
          help=f"SL: {len(losses)-len(eod_l)}  |  EOD−: {len(eod_l)}")
c4.metric("EOD",          len(eods),
          help=f"Cerradas por hora: {len(eod_w)} ganadoras, {len(eod_l)} perdedoras")
c5.metric("Capital ini.", f"${initial_capital:,.0f}")
c6.metric("Capital fin.", f"${initial_capital + total_usd:,.2f}")

# ══════════════════════════════════════════════════════════════════════
#  CURVA DE EQUITY
# ══════════════════════════════════════════════════════════════════════
st.subheader("📈 Curva de Equity")
eq_x_full = ["Inicio"] + [t.entry_dt.strftime("%Y-%m-%d") for t in trades]
eq_y      = [initial_capital] + [
    initial_capital + sum(t2.pnl_usd for t2 in trades[: k + 1])
    for k in range(len(trades))
]
colors_eq = ["#00d4aa" if y >= initial_capital else "#ff4b4b" for y in eq_y[1:]]

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=eq_x_full, y=eq_y,
    mode="lines+markers",
    line=dict(color="#00d4aa", width=2),
    marker=dict(size=10, color=["#00d4aa"] + colors_eq),
    hovertemplate="<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>",
))
fig_eq.add_hline(y=initial_capital, line_dash="dash",
                 line_color="gray", annotation_text="Capital inicial")
fig_eq.update_layout(
    height=300, margin=dict(l=0, r=0, t=10, b=0),
    yaxis_tickprefix="$", yaxis_tickformat=",.0f",
    plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
    font_color="#fafafa",
    xaxis=dict(gridcolor="#2d3548"),
    yaxis=dict(gridcolor="#2d3548"),
)
st.plotly_chart(fig_eq, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  GRÁFICOS — distribución y P&L por operación (ANTES de la tabla)
# ══════════════════════════════════════════════════════════════════════
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("🥧 Distribución")
    # Sub-gajos por tipo de cierre; EOD en tono claro dentro de la misma familia
    wins_tp  = [t for t in wins   if t.exit_reason != "EOD"]
    loss_sl  = [t for t in losses if t.exit_reason != "EOD"]
    pie_items = [
        ("Ganadoras TP",    len(wins_tp),  "#00d4aa"),   # verde intenso
        ("Ganadoras EOD+",  len(eod_w),    "#7fe8cf"),   # verde claro
        ("BE",              len(bes),      "#f0a500"),   # ámbar
        ("Perdedoras SL",   len(loss_sl),  "#ff4b4b"),   # rojo intenso
        ("Perdedoras EOD−", len(eod_l),    "#ff9999"),   # rojo claro
    ]
    pie_data = [(l, v, c) for l, v, c in pie_items if v > 0]
    fig_pie = go.Figure(go.Pie(
        labels=[d[0] for d in pie_data],
        values=[d[1] for d in pie_data],
        marker_colors=[d[2] for d in pie_data],
        hole=0.4,
        textinfo="label+percent+value",
        sort=False,   # mantener el orden: TP → EOD+ → BE → SL → EOD−
    ))
    fig_pie.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
        font_color="#fafafa", showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_r:
    st.subheader("📊 P&L por operación")
    bar_colors = ["#00d4aa" if t.pnl_usd > 0 else "#ff4b4b" for t in trades]
    bar_labels = [t.entry_dt.strftime("%d-%b") if t.entry_dt else str(k)
                  for k, t in enumerate(trades, 1)]
    fig_bar = go.Figure(go.Bar(
        x=bar_labels, y=[t.pnl_usd for t in trades],
        marker_color=bar_colors,
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:+,.2f}<extra></extra>",
    ))
    fig_bar.add_hline(y=0, line_color="gray", line_width=1)
    fig_bar.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickprefix="$", yaxis_tickformat="+,.0f",
        plot_bgcolor="#1a1f2e", paper_bgcolor="#1a1f2e",
        font_color="#fafafa",
        xaxis=dict(gridcolor="#2d3548"),
        yaxis=dict(gridcolor="#2d3548"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  TABLA DE OPERACIONES — con paginación
# ══════════════════════════════════════════════════════════════════════

# ── Encabezado: título + selector de filas por página ────────────────
col_title, col_rpp = st.columns([4, 1])
with col_title:
    st.subheader("📑 Operaciones")
with col_rpp:
    rows_per_page = st.selectbox(
        "Filas/pág.",
        options=[5, 10, 20, 30, 50, 100],
        index=1,
        key="rows_per_page",
    )
    # Resetear página si cambió el tamaño
    if st.session_state.get("_prev_rpp") != rows_per_page:
        st.session_state["tbl_page"] = 0
        st.session_state["_prev_rpp"] = rows_per_page

# ── Construir DataFrame completo ─────────────────────────────────────
rows = []
for k, t in enumerate(trades, 1):
    eq_before = t.equity_post - t.pnl_usd
    risk_usd  = t.risk_pts * t.qty * cfg.point_value
    rows.append({
        "#":        k,
        "Fecha":    t.entry_dt.strftime("%Y-%m-%d %H:%M") if t.entry_dt else "—",
        "Dir":      t.direction,
        "Tag":      t.tag,
        "Lotes":    t.qty,
        "Entrada":  round(t.entry_price, 2),
        "SL orig":  round(t.sl_orig, 2),
        "TP":       round(t.tp, 2),
        "SL pts":   round(t.risk_pts, 2),
        "Riesgo $": round(risk_usd, 2),
        "Riesgo %": round(risk_usd / eq_before * 100, 2),
        "Cierre":   t.exit_dt.strftime("%Y-%m-%d %H:%M") if t.exit_dt else "—",
        "Exit px":  round(t.exit_price, 2),
        "Razón":    t.exit_reason,
        "P&L pts":  round(t.pnl_pts, 2),
        "P&L $":    round(t.pnl_usd, 2),
        "Equity":   round(t.equity_post, 2),
    })

df_trades = pd.DataFrame(rows)

# ── Paginación ────────────────────────────────────────────────────────
total_rows  = len(df_trades)
total_pages = max(1, math.ceil(total_rows / rows_per_page))
page        = int(st.session_state.get("tbl_page", 0))
page        = max(0, min(page, total_pages - 1))   # clamp

start_idx = page * rows_per_page
end_idx   = min(start_idx + rows_per_page, total_rows)
df_page   = df_trades.iloc[start_idx:end_idx]

# ── Estilos ───────────────────────────────────────────────────────────
def _color_row(df):
    """Colorea P&L $ y P&L pts según exit_reason: TP=verde, BE=ámbar, SL/otro=rojo."""
    out = pd.DataFrame("", index=df.index, columns=df.columns)
    for idx, row in df.iterrows():
        reason = row.get("Razón", "")
        if reason == "TP" or (reason not in ("BE", "SL") and row["P&L $"] > 0):
            c = "color: #00d4aa; font-weight: bold"
        elif reason == "BE":
            c = "color: #f0a500; font-weight: bold"
        else:
            c = "color: #ff4b4b; font-weight: bold"
        out.loc[idx, "P&L $"]  = c
        out.loc[idx, "P&L pts"] = c
    return out

def color_dir(val):
    return "color: #00d4aa" if val == "LONG" else "color: #ff4b4b"

def color_reason(val):
    if val == "TP":              return "color: #00d4aa; font-weight: bold"
    if val == "BE":              return "color: #f0a500; font-weight: bold"
    if val == "SL":              return "color: #ff4b4b; font-weight: bold"
    if val in ("EOD", "END"):   return "color: #7cacf8; font-weight: bold"
    return ""

fmt = {
    "Entrada":  "{:.2f}", "SL orig": "{:.2f}", "TP":       "{:.2f}",
    "Exit px":  "{:.2f}", "SL pts":  "{:.2f}",
    "Riesgo $": "${:,.2f}", "Riesgo %": "{:.2f}%",
    "P&L pts":  "{:+.2f}", "P&L $":  "${:+,.2f}",
    "Equity":   "${:,.2f}",
}

_s = df_page.style.apply(_color_row, axis=None)
try:
    styled = (
        _s.map(color_dir,    subset=["Dir"])
          .map(color_reason, subset=["Razón"])
          .format(fmt)
    )
except AttributeError:
    styled = (
        _s.applymap(color_dir,    subset=["Dir"])
          .applymap(color_reason, subset=["Razón"])
          .format(fmt)
    )

st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Controles de paginación ───────────────────────────────────────────
st.markdown(
    f"<p style='text-align:center; color:#aaa; margin:4px 0;'>"
    f"Mostrando {start_idx+1}–{end_idx} de {total_rows} operaciones &nbsp;|&nbsp; "
    f"Página {page+1} de {total_pages}"
    f"</p>",
    unsafe_allow_html=True,
)

p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns([1, 1, 3, 1, 1])

with p_col1:
    if st.button("⏮ Primera", use_container_width=True, disabled=(page == 0)):
        st.session_state["tbl_page"] = 0
        st.rerun()

with p_col2:
    if st.button("◀ Anterior", use_container_width=True, disabled=(page == 0)):
        st.session_state["tbl_page"] = page - 1
        st.rerun()

with p_col4:
    if st.button("Siguiente ▶", use_container_width=True, disabled=(page >= total_pages - 1)):
        st.session_state["tbl_page"] = page + 1
        st.rerun()

with p_col5:
    if st.button("Última ⏭", use_container_width=True, disabled=(page >= total_pages - 1)):
        st.session_state["tbl_page"] = total_pages - 1
        st.rerun()

# ── Exportar ─────────────────────────────────────────────────────────
st.subheader("💾 Exportar")
csv_out = df_trades.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Descargar todos los resultados (CSV)",
    data=csv_out,
    file_name="nas100_backtest_results.csv",
    mime="text/csv",
)
