import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
import os
from dotenv import load_dotenv
import numpy as np
import traceback

load_dotenv()

TICKERS = ["AAPL", "CVS", "MSFT"]
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# ==================== FED DATES 2026 ====================
FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16"
]

def is_fed_meeting_today():
    today = datetime.now().strftime("%Y-%m-%d")
    return today in FOMC_DATES_2026

def is_fed_meeting_this_week():
    today = datetime.now().date()
    week_end = today + timedelta(days=(6 - today.weekday()))
    for d in FOMC_DATES_2026:
        d_date = datetime.strptime(d, "%Y-%m-%d").date()
        if today <= d_date <= week_end:
            return d_date
    return None

def get_fed_status():
    if is_fed_meeting_today():
        return "🔴 Fed Meeting HOY"
    week_meeting = is_fed_meeting_this_week()
    if week_meeting:
        return f"🟡 Fed Meeting esta semana ({week_meeting.strftime('%b %d')})"
    return "Fed: No programado esta semana"

# ==================== EARNINGS ====================
def get_earnings_info(ticker):
    """
    Retorna (texto, dias_hasta_earnings). dias_hasta_earnings es None si no hay dato.
    NOTA: yfinance reciente devuelve stock.calendar como dict (no DataFrame) -- hay que
    soportar ambos formatos o cae siempre en el except y muestra 'No disponible' aunque
    sí exista una fecha estimada de Yahoo.
    """
    try:
        stock = yf.Ticker(ticker)
        e_date = None

        cal = stock.calendar
        if isinstance(cal, dict):
            dates = cal.get('Earnings Date')
            if dates:
                e_date = pd.to_datetime(dates[0])
        elif cal is not None and hasattr(cal, 'empty') and not cal.empty and 'Earnings Date' in cal.iloc[0]:
            e_date = pd.to_datetime(cal.iloc[0]['Earnings Date'])

        if e_date is None:
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                upcoming = earnings[earnings.index >= datetime.now()].head(1)
                if not upcoming.empty:
                    e_date = upcoming.index[0]

        if e_date is not None:
            days_out = (e_date.date() - datetime.now().date()).days
            tag = "🔴" if 0 <= days_out <= 5 else "⚪"
            return f"{tag} Earnings (estimado): {e_date.strftime('%b %d')} ({days_out}d)", days_out
        return "Earnings: Sin fecha estimada disponible", None
    except Exception as e:
        print(f"[debug] earnings error {ticker}: {e}")
        return "Earnings: No disponible", None

# ==================== MACRO CONTEXT (SPY/QQQ) — Bloque B ====================
def get_macro_context():
    """
    Contexto macro rápido (Pring/Bloque B). No reemplaza el análisis de ciclo completo,
    es un semáforo de referencia: ¿el mercado amplio está a favor o en contra del trade?
    """
    out = {}
    for sym in ["SPY", "QQQ"]:
        try:
            t = yf.Ticker(sym)
            info = t.info
            hist = t.history(period="2d", interval="15m", prepost=True)
            if hist.empty:
                out[sym] = f"{sym}: No disponible"
                continue
            prev_close = info.get('regularMarketPreviousClose')
            current = hist['Close'].iloc[-1]
            gap_pct = ((current - prev_close) / prev_close * 100) if prev_close else 0
            direction = "🟢 Alcista" if gap_pct > 0.3 else "🔴 Bajista" if gap_pct < -0.3 else "⚪ Plano"
            out[sym] = f"{sym}: {direction} ({gap_pct:+.2f}%)"
        except Exception:
            out[sym] = f"{sym}: No disponible"
    return out

# ==================== BLOQUE A — FILTROS FUNDAMENTALES ====================
def get_bloque_a_filters(ticker):
    """
    Semáforo Bloque A: Recomendación ≤2.5, Target Price > precio actual, sin earnings en 5 días.
    Informativo (no bloquea el bot). Market Cap y 'opcionable' ya están validados manualmente
    al fijar la watchlist AAPL/CVS/MSFT.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        recommendation = info.get('recommendationMean')
        target_price = info.get('targetMeanPrice')
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')

        flags = []
        ok = True

        if recommendation is not None:
            if recommendation <= 2.5:
                flags.append(f"✅ Recom. {recommendation}")
            else:
                flags.append(f"⚠️ Recom. {recommendation} (>2.5)")
                ok = False
        else:
            flags.append("⚠️ Recom. N/D")

        if target_price and current_price:
            if target_price > current_price:
                flags.append(f"✅ Target ${target_price:.0f} > precio")
            else:
                flags.append(f"⚠️ Target ${target_price:.0f} ≤ precio")
                ok = False
        else:
            flags.append("⚠️ Target N/D")

        _, days_out = get_earnings_info(ticker)
        if days_out is not None and 0 <= days_out <= 5:
            flags.append(f"🔴 Earnings en {days_out}d")
            ok = False

        return {"pass": ok, "flags": flags}
    except Exception:
        return {"pass": None, "flags": ["Bloque A: No disponible"]}

# ==================== PRECIO ANCLA (al momento exacto de cada ventana de break) ====================
def get_snapshot_price(hist):
    """
    Busca el precio más cercano al momento ACTUAL (hora real en que corre el job:
    9:30, 11:30 o 2:00 ET), en vez de 'el último dato disponible' o un ancla fija a las 9AM.
    Este bot ya no es un scan premarket -- es tu sustituto de ojos en TOS durante
    cada ventana de break (no puedes ver desktop antes de las 9:30 ET).
    """
    try:
        today = datetime.now(ZoneInfo("America/New_York")).date()
        today_data = hist[hist.index.date == today]
        if today_data.empty:
            # SOLO PRUEBA: fuera de horario de mercado (fin de semana, feriado) no hay
            # barras de "hoy". TEST_ALLOW_STALE permite ver la última barra extendida
            # disponible (ej. after-hours del viernes) para validar que prepost=True
            # sí trae post-market real. NO se activa en producción -- requiere la
            # variable de entorno explícita, que no está en el .yml del cron.
            if os.environ.get("TEST_ALLOW_STALE") == "1" and not hist.empty:
                return round(float(hist['Close'].iloc[-1]), 2)
            return None
        idx = today_data.index.get_indexer([today_data.index[-1]], method='nearest')[0]
        return round(float(today_data['Close'].iloc[idx]), 2)
    except Exception:
        return None

# ==================== CONTEXTO DÍA (Bloque C, temporalidad DÍA) ====================
def get_daily_context(ticker):
    """
    MA20D/MA40D como CONTEXTO macro (Bloque C, temporalidad Día). En la arquitectura de
    3 temporalidades de Felipe, el Día es contexto, no bloqueo -- la decisión vive en Hora.
    NOTA: 'primera visita a la MA' es un criterio de las Estrategias 3/4 (Rebote punto medio),
    NO de E1/E2 (Cambio de tendencia), donde el cruce de MA20 es justo la señal que se busca.
    Por eso esa validación fue removida de aquí y de analyze_yoel_e1_e2.
    """
    try:
        daily = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if daily.empty or len(daily) < 45:
            return {"bias_dia": "DATA LIMITADA"}

        ma20d = daily['Close'].rolling(20).mean().iloc[-1]
        ma40d = daily['Close'].rolling(40).mean().iloc[-1]
        price = daily['Close'].iloc[-1]

        if ma20d > ma40d and price > ma20d:
            bias_dia = "🟢 ALCISTA (Día)"
        elif ma20d < ma40d and price < ma20d:
            bias_dia = "🔴 BAJISTA (Día)"
        else:
            bias_dia = "⚪ LATERAL (Día)"

        return {"bias_dia": bias_dia, "ma20d": round(ma20d, 2), "ma40d": round(ma40d, 2)}
    except Exception as e:
        print(f"[debug] daily context error {ticker}: {e}")
        return {"bias_dia": "No disponible"}

# ==================== YOEL E1 E2 ====================
def analyze_yoel_e1_e2(df):
    if len(df) < 80:
        return {"bias": "DATA LIMITADA", "strength": "", "setup": "INSUFICIENTE"}

    df = df.copy()
    df['hour'] = df.index.floor('h')
    hourly = df.groupby('hour').agg({
        'High': 'max', 'Low': 'min', 'Close': 'last', 'Open': 'first'
    }).dropna()

    if len(hourly) < 10:
        return {"bias": "DATA LIMITADA", "strength": "", "setup": "INSUFICIENTE"}

    h_high = hourly['High']
    h_low = hourly['Low']
    h_close = hourly['Close']

    swing_high = (h_high.rolling(7, center=True).max() == h_high) & (h_high.shift(1) < h_high) & (h_high.shift(-1) < h_high)
    swing_low = (h_low.rolling(7, center=True).min() == h_low) & (h_low.shift(1) > h_low) & (h_low.shift(-1) > h_low)

    recent_highs = h_high[swing_high].dropna().tail(3)
    recent_lows = h_low[swing_low].dropna().tail(3)

    lower_highs_ok = len(recent_highs) >= 2 and recent_highs.iloc[-1] < recent_highs.iloc[-2]
    higher_lows_ok = len(recent_lows) >= 2 and recent_lows.iloc[-1] > recent_lows.iloc[-2]

    ma20_h_series = h_close.rolling(20).mean()
    ma20_h = ma20_h_series.iloc[-1]

    # BB15 real (Bloque F — timing 15min). df ya es 15min, temporalidad correcta del checklist.
    # Opción A: ancho actual > promedio móvil 50 barras * factor_abierta (1.0).
    # Mismo cálculo que Yoel_E1_E2_CambioDeTendencia.ts en TOS.
    bb15_width_series = df['Close'].rolling(20).std() * 4   # (upper−lower) = 4σ
    bb15_current      = bb15_width_series.iloc[-1]
    bb15_avg          = bb15_width_series.rolling(50).mean().iloc[-1]
    no_lateral = bool(not pd.isna(bb15_avg) and bb15_avg > 0 and bb15_current > bb15_avg)

    price = df['Close'].iloc[-1]

    # ── Volumen relativo (Pring) ──────────────────────────────────────────────
    # Promedio de las últimas 4 barras de 15min (≈1h) vs media histórica de la ventana.
    # Confirma que la ruptura tiene convicción real detrás.
    try:
        avg_vol    = df['Volume'].mean()
        recent_vol = df['Volume'].tail(4).mean()
        vol_relativo = round(recent_vol / avg_vol, 2) if avg_vol > 0 else None
        vol_confirma = vol_relativo is not None and vol_relativo >= 1.2
    except Exception:
        vol_relativo, vol_confirma = None, False

    # ── Soporte y resistencia horizontales (Murphy) ───────────────────────────
    # Swing highs/lows del horario ya calculados arriba como niveles de referencia.
    # El más cercano por encima = resistencia; el más cercano por debajo = soporte.
    try:
        swing_high_prices = h_high[swing_high].dropna()
        swing_low_prices  = h_low[swing_low].dropna()
        resistencias_arr  = swing_high_prices[swing_high_prices > price].sort_values()
        soportes_arr      = swing_low_prices[swing_low_prices < price].sort_values(ascending=False)
        nivel_resistencia = float(resistencias_arr.iloc[0]) if len(resistencias_arr) > 0 else None
        nivel_soporte     = float(soportes_arr.iloc[0])     if len(soportes_arr) > 0     else None
    except Exception:
        nivel_resistencia, nivel_soporte = None, None

    # ── Distancia a MA20H en % (contexto de timing) ───────────────────────────
    # Un precio ya alejado 2%+ de la MA20H entra tarde -- mayor riesgo de perseguir.
    dist_ma20h_pct = round((price - ma20_h) / ma20_h * 100, 2) if ma20_h else None

    # ── Conteo DeMark simplificado (agotamiento) ──────────────────────────────
    # TD Setup: 9 cierres consecutivos en 15min, cada uno < (o >) al cierre de 4 barras antes.
    # Bearish setup (9 cierres < cierre[−4]) = agotamiento bajista → señal de posible reversión alcista.
    # Bullish setup (9 cierres > cierre[−4]) = agotamiento alcista → señal de posible reversión bajista.
    try:
        cl = hourly['Close']  # Hora — horizonte correcto para TD Setup (9h ≈ 9 sesiones reales)
        bear_cnt = bull_cnt = 0
        for i in range(len(cl) - 1, 3, -1):
            if cl.iloc[i] < cl.iloc[i - 4]:
                if bull_cnt > 0: break
                bear_cnt += 1
            elif cl.iloc[i] > cl.iloc[i - 4]:
                if bear_cnt > 0: break
                bull_cnt += 1
            else:
                break
        if bear_cnt >= 9:
            demark_signal = f"⚠️ Agotamiento bajista {bear_cnt}/9+ → vigilar reversión alcista"
        elif bull_cnt >= 9:
            demark_signal = f"⚠️ Agotamiento alcista {bull_cnt}/9+ → vigilar reversión bajista"
        elif bear_cnt >= 6:
            demark_signal = f"🟡 Conteo bajista {bear_cnt}/9"
        elif bull_cnt >= 6:
            demark_signal = f"🟡 Conteo alcista {bull_cnt}/9"
        else:
            demark_signal = f"{max(bear_cnt, bull_cnt)}/9"
    except Exception:
        demark_signal = "N/D"

    # BB15 upper/lower para el cálculo de strength
    bb15_mid   = df['Close'].rolling(20).mean().iloc[-1]
    bb15_std   = df['Close'].rolling(20).std().iloc[-1]
    bb15_upper = bb15_mid + 2 * bb15_std
    bb15_lower = bb15_mid - 2 * bb15_std

    broken_e1 = lower_highs_ok and price > ma20_h
    broken_e2 = higher_lows_ok and price < ma20_h

    if broken_e1 and no_lateral:
        bias = "🟢 E1 ALCISTA (CALL)"
        strength = "FUERTE" if price > bb15_upper else "MODERADA"
        setup = "E1 COMPLETO"
    elif broken_e2 and no_lateral:
        bias = "🔴 E2 BAJISTA (PUT)"
        strength = "FUERTE" if price < bb15_lower else "MODERADA"
        setup = "E2 COMPLETO"
    elif broken_e1:
        bias = "🟡 E1 WATCH"
        strength = "ESPERAR MA20 + BB"
        setup = "E1 WATCH"
    elif broken_e2:
        bias = "🟡 E2 WATCH"
        strength = "ESPERAR MA20 + BB"
        setup = "E2 WATCH"
    else:
        bias = "⚪ SIN SETUP CLARO"
        strength = "LATERAL / SIN RUPTURA"
        setup = "SIN SETUP"

    return {
        "bias": bias, "strength": strength, "setup": setup,
        "ma20_h": round(ma20_h, 2), "price": round(price, 2),
        "lower_highs": lower_highs_ok, "higher_lows": higher_lows_ok,
        "no_lateral": bool(no_lateral),
        "dist_ma20h_pct":   dist_ma20h_pct,
        "nivel_soporte":    round(nivel_soporte, 2)    if nivel_soporte    is not None else None,
        "nivel_resistencia": round(nivel_resistencia, 2) if nivel_resistencia is not None else None,
        "vol_relativo":     vol_relativo,
        "vol_confirma":     bool(vol_confirma),
        "demark_signal":    demark_signal,
    }

# ==================== TREND LINE (MECHAS, UNA SOLA, SEGÚN BIAS) ====================
def _build_trendline(df_index, x1, y1, x2, y2):
    """
    Construye una línea de tendencia conectando dos pivotes (mecha a mecha) y la
    EXTIENDE con la misma pendiente hasta la última vela del gráfico.

    IMPORTANTE: la pendiente se calcula por POSICIÓN de barra (entero), no por
    tiempo real transcurrido. mplfinance grafica con show_nontrading=False, lo que
    espacia cada vela por posición fija sin importar el gap real de tiempo (noche,
    fin de semana). Si la pendiente se calcula con nanosegundos reales, los gaps
    de tiempo real generan saltos verticales enormes en un solo paso de posición,
    produciendo el efecto "escalonado". Usando posición entera, la línea queda
    recta y alineada exactamente con cómo se ven las velas en el gráfico.
    """
    if x1 == x2:
        return None
    mask = df_index >= x1
    if mask.sum() < 2:
        return None
    pos0 = df_index.get_loc(x1)
    pos1 = df_index.get_loc(x2)
    if pos1 == pos0:
        return None
    slope = (y2 - y1) / (pos1 - pos0)
    line = pd.Series(index=df_index, dtype=float)
    for i, ts in enumerate(df_index):
        if mask[i]:
            line.loc[ts] = y1 + slope * (i - pos0)
    return line

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _upper_hull(points):
    """Borde superior del casco convexo. Ningún punto queda por encima de los
    segmentos resultantes -- garantiza que una línea de resistencia trazada sobre
    dos vértices consecutivos del hull nunca sea perforada por una vela intermedia."""
    pts = sorted(points)
    hull = []
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull

def _lower_hull(points):
    """Borde inferior del casco convexo -- equivalente para líneas de soporte."""
    pts = sorted(points)
    hull = []
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull

def _anchor_from_extreme(hull, extreme_idx, min_span=3):
    """
    Filtra el casco convexo a solo los vértices EN O DESPUÉS del índice del
    extremo global (el high más alto para resistencia, el low más bajo para
    soporte) y devuelve el primer tramo significativo desde ahí.

    Por qué: anclar simplemente en el primer tramo del hull (sin este filtro)
    funciona si la tendencia es monótona en toda la ventana (ej. MSFT cayendo
    desde el día 1), pero falla si el ticker tiene forma de 'joroba' (ej. CVS:
    sube hasta un pico y luego baja) -- el primer tramo del hull agarraría la
    SUBIDA previa al pico, que ya no es relevante porque el precio ya la rompió.
    Anclar en el extremo global más reciente (el pico/valle que define la fase
    ACTUAL del precio) y avanzar desde ahí da la línea correcta en los tres
    casos: tendencia monótona, reversión alcista, y reversión bajista.
    """
    sub = [p for p in hull if p[0] >= extreme_idx]
    if len(sub) < 2:
        return None
    for i in range(len(sub) - 1):
        x1, _ = sub[i]
        x2, _ = sub[i + 1]
        if x2 - x1 >= min_span:
            return (sub[i], sub[i + 1])
    return (sub[0], sub[-1])

def _recent_direction(window, recent_bars=30):
    """
    Pendiente de regresión lineal (mínimos cuadrados) sobre las últimas
    `recent_bars` velas, NO una simple resta entre el primer y último close.
    La resta simple es muy sensible a dónde cae el corte de la ventana: si hay
    un pico o valle intermedio (ej. un rally que ya hizo top y viene
    retrocediendo dentro de esa ventana), la resta puede dar negativo aunque la
    tendencia de fondo siga siendo claramente alcista. La regresión usa TODOS
    los puntos de la ventana, así que un solo pico/valle intermedio pesa mucho
    menos que la dirección dominante del conjunto.
    """
    recent = window.tail(recent_bars)
    if len(recent) < 5:
        return 0
    y = recent['Close'].values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return slope

def _get_relevant_trendline(df_h, yoel_setup, lower_highs_ok=False, higher_lows_ok=False, lookback=None):
    """
    Dibuja SOLO la línea relevante, nunca ambas a la vez.
    Prioridad: setup COMPLETO (E1/E2 en yoel_setup) manda primero.
    Si no hay setup completo (WATCH o SIN SETUP), usa el bias de estructura
    (lower_highs_ok = bajista -> línea de highs; higher_lows_ok = alcista -> línea de lows).
    Si ambos flags están activos a la vez (o ninguno), es indecisión -> no se dibuja nada.

    La línea se ancla en el PRIMER tramo del casco convexo: el pivote más
    EXTREMO real disponible en los datos hasta el siguiente toque que confirma
    la pendiente -- igual que se traza a mano en TOS (desde el high/low más
    significativo, no desde los últimos pivotes ni desde el tramo de mayor
    longitud horizontal). lookback=None usa TODO el histórico disponible para
    no perder el pivote extremo real por un recorte de ventana arbitrario.
    """
    window = df_h if lookback is None else df_h.tail(lookback)
    s = yoel_setup or ""
    # Setup "completo" = la señal final confirmada (ej. 'E1', 'E2'), NO un estado
    # intermedio como 'E1 WATCH' / 'E2 WATCH' ni 'SIN SETUP'. Antes "E2" in s
    # también hacía match con "E2 WATCH", forzando la línea bajista aunque el
    # bias diario fuera Alcista y la estructura mostrara higher-lows.
    is_e1_setup = "E1" in s and "WATCH" not in s and "SIN SETUP" not in s
    is_e2_setup = "E2" in s and "WATCH" not in s and "SIN SETUP" not in s

    if is_e1_setup:
        draw_e1, draw_e2 = True, False
    elif is_e2_setup:
        draw_e1, draw_e2 = False, True
    elif _recent_direction(window) != 0:
        # Para la línea VISUAL (sin setup completo), priorizamos la estructura
        # RECIENTE sobre los flags oficiales lower_highs_ok/higher_lows_ok.
        # Esos flags son correctos como señal (miran el histórico multi-día
        # completo), pero por eso mismo pueden seguir marcando 'lower_highs_ok'
        # mientras el precio sigue por debajo de un high de hace +1 semana,
        # aunque la estructura de los últimos días ya sea claramente alcista.
        # Esta rama no toca la señal oficial E1/E2 ni Bloques C-F, solo decide
        # qué línea pintar como referencia visual.
        draw_e1, draw_e2 = (True, False) if _recent_direction(window) > 0 else (False, True)
    elif higher_lows_ok and not lower_highs_ok:
        draw_e1, draw_e2 = True, False
    elif lower_highs_ok and not higher_lows_ok:
        draw_e1, draw_e2 = False, True
    else:
        return None, None

    if draw_e1:
        lows_arr = window['Low'].values
        idx_min = int(lows_arr.argmin())
        pts = list(enumerate(lows_arr))
        hull = _lower_hull(pts)
        edge = _anchor_from_extreme(hull, idx_min)
        if edge:
            (i1, y1), (i2, y2) = edge
            x1, x2 = window.index[i1], window.index[i2]
            trend = _build_trendline(df_h.index, x1, y1, x2, y2)
            return trend, '#4fc3f7'  # cyan -- alcista

    if draw_e2:
        highs_arr = window['High'].values
        idx_max = int(highs_arr.argmax())
        pts = list(enumerate(highs_arr))
        hull = _upper_hull(pts)
        edge = _anchor_from_extreme(hull, idx_max)
        if edge:
            (i1, y1), (i2, y2) = edge
            x1, x2 = window.index[i1], window.index[i2]
            trend = _build_trendline(df_h.index, x1, y1, x2, y2)
            return trend, '#ff4fa3'  # rosado -- bajista

    return None, None

# ==================== CHART ====================
def create_premarket_chart(ticker, df, yoel_data):
    """Gráfico mplfinance: 1h, MA20 amarilla + MA40 roja, UNA línea de tendencia según bias."""
    try:
        import mplfinance as mpf

        df_h = df.resample('h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()

        if len(df_h) < 25:
            return None

        ma20 = df_h['Close'].rolling(20).mean()
        ma40 = df_h['Close'].rolling(40).mean()

        apds = [
            mpf.make_addplot(ma20, color='#ffeb3b', width=2),
            mpf.make_addplot(ma40, color='#ff5252', width=2),
        ]

        trend, color = _get_relevant_trendline(
            df_h, yoel_data.get('yoel_setup', ''),
            lower_highs_ok=bool(yoel_data.get('lower_highs')),
            higher_lows_ok=bool(yoel_data.get('higher_lows')),
        )
        if trend is not None:
            apds.append(mpf.make_addplot(trend, color=color, width=1.8))

        mc = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='inherit', wick='inherit'
        )
        s = mpf.make_mpf_style(
            marketcolors=mc,
            base_mpl_style='dark_background',
            gridstyle='--',
            facecolor='#131722'
        )

        filepath = os.path.join(CHARTS_DIR, f"{ticker}_premarket_{datetime.now().strftime('%Y%m%d_%H%M')}.png")

        mpf.plot(
            df_h,
            type='candle',
            style=s,
            addplot=apds,
            title=f"{ticker}  •  1h  •  {yoel_data.get('var_dir','')}  •  {yoel_data.get('yoel_setup','')}",
            ylabel='Price',
            figsize=(13, 6.5),
            show_nontrading=False,
            datetime_format='%m/%d %Hh',
            xrotation=15,
            tight_layout=True,
            savefig=dict(fname=filepath, dpi=160, bbox_inches='tight', facecolor='#131722')
        )
        return filepath
    except Exception as e:
        print(f"Chart error {ticker}: {e}")
        traceback.print_exc()
        return None

# ==================== FULL ====================
def get_full_premarket_analysis(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="10d", interval="15m")  # regular hours -- gráfico y lógica Yoel E1/E2 sin tocar

    if hist.empty:
        return None

    # Serie aparte SOLO para el precio actual en pre-market/extended hours.
    # No se usa para el gráfico ni para analyze_yoel_e1_e2 -- esas dos siguen
    # leyendo `hist` (regular hours) exactamente como antes.
    try:
        hist_extended = stock.history(period="2d", interval="15m", prepost=True)
    except Exception:
        hist_extended = hist

    prev_close = info.get('regularMarketPreviousClose')
    snapshot_price = get_snapshot_price(hist_extended)
    current = snapshot_price if snapshot_price is not None else float(hist['Close'].iloc[-1])

    var_pct = ((current - prev_close) / prev_close * 100) if prev_close else 0
    var_dir = "🟢 Alcista" if var_pct > 0.3 else "🔴 Bajista" if var_pct < -0.3 else "⚪ Plano"

    earnings_str, _ = get_earnings_info(ticker)
    fed_str = get_fed_status()
    bloque_a = get_bloque_a_filters(ticker)
    yoel = analyze_yoel_e1_e2(hist)
    daily_ctx = get_daily_context(ticker)
    chart_meta = {
        "yoel_setup": yoel.get("setup", ""), "var_dir": var_dir,
        "lower_highs": yoel.get("lower_highs"), "higher_lows": yoel.get("higher_lows"),
    }
    chart_path = create_premarket_chart(ticker, hist, chart_meta)

    return {
        "ticker": ticker,
        "prev_close": round(prev_close, 2) if prev_close else None,
        "snapshot_price": round(current, 2) if current is not None else None,
        "var_pct": round(var_pct, 2), "var_dir": var_dir,
        "earnings": earnings_str, "fed": fed_str,
        "bloque_a_pass": bloque_a.get("pass"), "bloque_a_flags": bloque_a.get("flags", []),
        "yoel_bias": yoel.get("bias", ""), "yoel_setup": yoel.get("setup", ""),
        "yoel_lower_highs": yoel.get("lower_highs"),
        "yoel_higher_lows": yoel.get("higher_lows"),
        "yoel_no_lateral":  yoel.get("no_lateral"),
        "ma20_h":           yoel.get("ma20_h"),
        "yoel_dist_ma20h_pct":    yoel.get("dist_ma20h_pct"),
        "yoel_nivel_soporte":     yoel.get("nivel_soporte"),
        "yoel_nivel_resistencia": yoel.get("nivel_resistencia"),
        "yoel_vol_relativo":      yoel.get("vol_relativo"),
        "yoel_vol_confirma":      yoel.get("vol_confirma"),
        "yoel_demark_signal":     yoel.get("demark_signal"),
        "bias_dia": daily_ctx.get("bias_dia"),
        "chart_path": chart_path, "df": hist
    }

def generate_report():
    report = f"🕒 **SNAPSHOT Yoel_E1_E2 | {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M ET')}**\n\n"

    macro = get_macro_context()
    report += f"📊 Macro (Bloque B): {macro.get('SPY','')} | {macro.get('QQQ','')}\n\n"

    for t in TICKERS:
        data = get_full_premarket_analysis(t)
        if not data:
            report += f"**{t}**: Error\n\n"
            continue

        report += f"**{t}**  |  Cierre ant.: ${data['prev_close']}  |  Actual: ${data['snapshot_price']}  |  Var: {data['var_pct']}% {data['var_dir']}\n"
        report += f"{data['earnings']}  |  {data['fed']}\n"
        bloque_a_tag = "✅" if data['bloque_a_pass'] else ("⚠️" if data['bloque_a_pass'] is False else "—")
        report += f"Bloque A {bloque_a_tag}: {' | '.join(data['bloque_a_flags'])}\n"
        report += f"Yoel: {data['yoel_bias']} | Setup: {data['yoel_setup']}\n"
        report += f"LH: {data['yoel_lower_highs']} | HL: {data['yoel_higher_lows']} | NoLateral: {data['yoel_no_lateral']} | Contexto Día: {data['bias_dia']}\n"
        report += f"MA20H: ${data['ma20_h']}\n"
        if data.get('chart_path'):
            report += f"📊 Chart saved: {data['chart_path']}\n"
        report += "\n"

    report += "Automatizado • Var vs cierre + Earnings + Fed + Bloque A + Macro SPY/QQQ + Yoel_E1_E2 + Chart con línea de tendencia única"
    return report

def send_to_discord(message, image_paths=None):
    if not DISCORD_WEBHOOK:
        print("Webhook no configurado")
        return

    try:
        if image_paths:
            files = {}
            for i, path in enumerate(image_paths):
                if os.path.exists(path):
                    files[f"file{i}"] = open(path, "rb")

            payload = {"content": message}
            requests.post(DISCORD_WEBHOOK, data=payload, files=files)

            for f in files.values():
                f.close()
            print("✅ Reporte + gráficos enviados a Discord")
        else:
            requests.post(DISCORD_WEBHOOK, json={"content": message})
            print("✅ Reporte enviado a Discord")
    except Exception as e:
        print(f"Error enviando a Discord: {e}")

def job():
    print("Generando reporte de break (snapshot TOS)...")

    macro = get_macro_context()
    header = (
        f"🕒 **SNAPSHOT Yoel_E1_E2 | {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M ET')}**\n"
        f"📊 Macro (Bloque B): {macro.get('SPY','')} | {macro.get('QQQ','')}"
    )
    send_to_discord(header)

    for t in TICKERS:
        data = get_full_premarket_analysis(t)
        if not data:
            send_to_discord(f"**{t}**: Error obteniendo datos")
            continue

        # Debug interno (consola, no va a Discord)
        print(f"[debug] {t} LH={data['yoel_lower_highs']} HL={data['yoel_higher_lows']}")

        setup      = data['yoel_setup']
        ma20h      = data['ma20_h']
        soporte    = data.get('yoel_nivel_soporte')
        resist     = data.get('yoel_nivel_resistencia')
        dist       = data.get('yoel_dist_ma20h_pct')
        vol_rel    = data.get('yoel_vol_relativo')
        vol_ok     = data.get('yoel_vol_confirma')
        demark     = data.get('yoel_demark_signal', 'N/D')
        bb15_str   = "Abierto ✅" if data['yoel_no_lateral'] else "Lateral ⚠️"

        # Tesis condicional SI/ENTONCES — formato Mark Douglas:
        # el análisis pre-market prepara la mente, no predice el resultado.
        s_str = f"${soporte}"  if soporte is not None else "N/D"
        r_str = f"${resist}"   if resist  is not None else "N/D"
        if 'E1' in setup and 'WATCH' not in setup:
            tesis = (f"✅ SI precio mantiene sobre MA20H (${ma20h}) CON BB15 {bb15_str} → entrada válida\n"
                     f"❌ INVALIDA si precio cae bajo soporte {s_str}")
        elif 'E2' in setup and 'WATCH' not in setup:
            tesis = (f"✅ SI precio mantiene bajo MA20H (${ma20h}) CON BB15 {bb15_str} → entrada válida\n"
                     f"❌ INVALIDA si precio sube sobre resistencia {r_str}")
        elif 'E1 WATCH' in setup:
            tesis = (f"👁 SI precio cierra sobre MA20H (${ma20h}) CON BB15 abierto → E1 confirmado\n"
                     f"❌ INVALIDA si precio no logra superar resistencia {r_str}")
        elif 'E2 WATCH' in setup:
            tesis = (f"👁 SI precio cierra bajo MA20H (${ma20h}) CON BB15 abierto → E2 confirmado\n"
                     f"❌ INVALIDA si precio rebota sobre resistencia {r_str}")
        else:
            tesis = f"⏸ Sin setup activo. Esperar ruptura de MA20H (${ma20h})\n   R: {r_str}  |  S: {s_str}"

        dist_str  = f"{dist:+.1f}%" if dist is not None else "N/D"
        vol_str   = (f"{vol_rel:.1f}x {'✅' if vol_ok else '⚠️'}"
                     if vol_rel is not None else "N/D")
        bloque_a_tag = "✅" if data['bloque_a_pass'] else ("⚠️" if data['bloque_a_pass'] is False else "—")

        msg = (
            f"**{t}**  |  ${data['snapshot_price']}  ({data['var_pct']:+.2f}% {data['var_dir']})  |  Cierre ant.: ${data['prev_close']}\n"
            f"{data['earnings']}  |  {data['fed']}\n"
            f"Bloque A {bloque_a_tag}: {' | '.join(data['bloque_a_flags'])}\n"
            f"─────────────────────────────\n"
            f"Setup: {data['yoel_bias']}  |  Ctx Día: {data['bias_dia']}\n"
            f"{tesis}\n"
            f"📏 MA20H: ${ma20h} ({dist_str})  |  S: {s_str}  |  R: {r_str}\n"
            f"📊 Vol: {vol_str}  |  DeMark: {demark}  |  BB15: {bb15_str}"
        )
        images = [data['chart_path']] if data.get('chart_path') else None
        send_to_discord(msg, image_paths=images)

    print("✅ Reportes individuales + gráficos enviados a Discord")

TARGET_TIMES_ET = [(9, 0), (11, 0), (13, 30)]
TOLERANCE_MIN = 30  # margen ampliado — GitHub Actions puede arrancar hasta 30min tarde


def _is_target_window():
    """
    True si la hora actual en America/New_York cae dentro de +-TOLERANCE_MIN
    de alguna de las 3 ventanas. Se calcula en ET (no UTC) para que el mismo
    workflow de GitHub Actions siga funcionando correctamente al cambiar
    horario de verano/invierno, sin tener que editar el cron dos veces al año.

    FORCE_RUN=1 salta la guarda -- útil para correr test_premarket.py manualmente
    o para un dispatch manual del workflow sin esperar la ventana real.
    """
    if os.environ.get("FORCE_RUN") == "1":
        return True
    now = datetime.now(ZoneInfo("America/New_York"))
    now_min = now.hour * 60 + now.minute
    for h, m in TARGET_TIMES_ET:
        if abs(now_min - (h * 60 + m)) <= TOLERANCE_MIN:
            return True
    return False


if __name__ == "__main__":
    print("🚀 Agente iniciado (ejecución única, pensada para cron externo / GitHub Actions)...")
    if _is_target_window():
        job()
    else:
        now_et = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M ET")
        print(f"⏭️  {now_et} no cae en ninguna ventana objetivo ({TARGET_TIMES_ET}) -- no se ejecuta job().")
