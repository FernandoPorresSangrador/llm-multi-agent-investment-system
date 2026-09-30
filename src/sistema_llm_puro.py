"""

Modulo principal del sistema multiagente de inversion.
Contiene los 5 agentes especializados y el orquestador.

Cambios v2:
- herramienta_sentimiento_historico(): usa dataset FNSPID para backtesting
- cargar_noticias(): carga el parquet FNSPID una sola vez en memoria
- agente_sentimiento() acepta parametro opcional fecha para backtesting
- orquestador() acepta parametro opcional fecha para backtesting
- generar_señales_mensuales() usa sentimiento historico automaticamente
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import timedelta
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from langchain_ollama import ChatOllama

# Configuracion 

MODELO = "mistral:7b-instruct-v0.3-q4_0"

FNSPID_CUTOFF = pd.Timestamp("2024-01-09")   # ultimo dia con cobertura FNSPID
TICKERS_SIN_FNSPID = {"META", "HON"}         # sin cobertura en el dataset

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "ORCL", "INTC", "QCOM",
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "MDT", "BMY",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", "USB", "PNC",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "F", "GM",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "CAT", "BA", "HON", "GE", "LMT",
]

NOMBRES = {
    "AAPL": "Apple",             "MSFT": "Microsoft",        "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",         "META": "Meta",              "AVGO": "Broadcom",
    "AMD": "AMD",                "ORCL": "Oracle",            "INTC": "Intel",
    "QCOM": "Qualcomm",          "JNJ": "Johnson & Johnson",  "UNH": "UnitedHealth",
    "LLY": "Eli Lilly",          "ABBV": "AbbVie",            "MRK": "Merck",
    "TMO": "Thermo Fisher",      "ABT": "Abbott",             "PFE": "Pfizer",
    "MDT": "Medtronic",          "BMY": "Bristol-Myers",      "JPM": "JPMorgan",
    "BAC": "Bank of America",    "WFC": "Wells Fargo",        "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",      "BLK": "BlackRock",          "AXP": "American Express",
    "C": "Citigroup",            "USB": "U.S. Bancorp",       "PNC": "PNC Financial",
    "AMZN": "Amazon",            "TSLA": "Tesla",             "HD": "Home Depot",
    "MCD": "McDonald's",         "NKE": "Nike",               "SBUX": "Starbucks",
    "TGT": "Target",             "LOW": "Lowe's",             "F": "Ford",
    "GM": "General Motors",      "XOM": "ExxonMobil",         "CVX": "Chevron",
    "COP": "ConocoPhillips",     "SLB": "Schlumberger",       "EOG": "EOG Resources",
    "CAT": "Caterpillar",        "BA": "Boeing",              "HON": "Honeywell",
    "GE": "General Electric",    "LMT": "Lockheed Martin",
}

PERFILES = {
    "conservador": {
        "descripcion": "Inversor conservador. Prioriza la preservacion del capital sobre la rentabilidad. Solo recomienda BUY si el riesgo es claramente bajo. Prefiere HOLD ante la duda.",
        "confianza_minima": 7,
        "max_posiciones": 5,
    },
    "moderado": {
        "descripcion": "Inversor moderado. Equilibra rentabilidad y riesgo. Acepta volatilidad moderada. Recomienda BUY si los indicadores son mayoritariamente positivos.",
        "confianza_minima": 5,
        "max_posiciones": 10,
    },
    "agresivo": {
        "descripcion": "Inversor agresivo. Maximiza la rentabilidad aceptando alto riesgo. Recomienda BUY ante cualquier señal positiva.",
        "confianza_minima": 3,
        "max_posiciones": 15,
    },
}

# Modelos Pydantic

class SeñalInversion(BaseModel):
    ticker: str = Field(description="Simbolo de la accion en mayusculas")
    decision: Literal["BUY", "HOLD", "SELL"] = Field(
        description="Decision de inversion. SOLO puede ser BUY, HOLD o SELL en mayusculas"
    )
    confianza: int = Field(ge=1, le=10, description="Nivel de confianza del 1 al 10")
    justificacion: str = Field(
        description="Justificacion breve en maximo 15 palabras en español"
    )

class SeñalBatch(BaseModel):
    señales: List[SeñalInversion] = Field(
        description="Lista de señales una por cada ticker analizado"
    )

class DecisionFinal(BaseModel):
    ticker: str = Field(description="Simbolo de la accion")
    decision: Literal["BUY", "HOLD", "SELL"] = Field(
        description="Decision final de inversion"
    )
    confianza: int = Field(ge=1, le=10, description="Confianza del 1 al 10")
    resumen_agentes: str = Field(
        description="Resumen de las señales de cada agente en una linea"
    )
    justificacion: str = Field(
        description="Justificacion final en maximo 20 palabras en español"
    )

# LLMs

def get_llm():
    return ChatOllama(
        model=MODELO,
        temperature=0,
        base_url="http://localhost:11434"
    )

# Carga de datos

def cargar_datos(carpeta: str = "../datos") -> dict:
    datos = {}
    for ticker in TICKERS:
        ruta = f"{carpeta}/{ticker}.parquet"
        if os.path.exists(ruta):
            datos[ticker] = pd.read_parquet(ruta)
    print(f"Datos cargados: {len(datos)}/{len(TICKERS)} tickers")
    return datos

# Carga de noticias FNSPID 

# Variable global para cachear el DataFrame de noticias en memoria.
# Se carga una sola vez con cargar_noticias() y se reutiliza en todas
# las llamadas a herramienta_sentimiento_historico().
_df_noticias: Optional[pd.DataFrame] = None

def cargar_noticias(carpeta: str = "../datos") -> pd.DataFrame:
    """
    Carga el dataset FNSPID desde parquet y lo cachea en memoria.
    Llama a esta funcion al inicio del notebook de backtesting para
    evitar releer el archivo en cada llamada al agente de sentimiento.

    Uso:
        from sistema_llm_puro import cargar_noticias
        cargar_noticias()   # carga y cachea; imprime resumen
    """
    global _df_noticias
    if _df_noticias is not None:
        print(f"Noticias ya cargadas en memoria: {len(_df_noticias):,} registros")
        return _df_noticias

    ruta = os.path.join(carpeta, "fnspid_50tickers.parquet")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontro el dataset FNSPID en '{ruta}'.\n"
            "Ejecuta primero el notebook 01_datos_y_exploracion para descargarlo."
        )

    df = pd.read_parquet(ruta)
    df["Date"] = pd.to_datetime(df["Date"])

    # Normalizar nombre de columna de ticker (puede ser 'Ticker' o 'ticker')
    if "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    elif "ticker" not in df.columns:
        raise ValueError("El parquet FNSPID no tiene columna 'Ticker' o 'ticker'")

    # Normalizar nombre de columna de titular
    if "Article_title" in df.columns:
        df = df.rename(columns={"Article_title": "titulo"})
    elif "title" in df.columns:
        df = df.rename(columns={"title": "titulo"})
    elif "titulo" not in df.columns:
        # Tomar la primera columna de texto que no sea Date ni ticker
        cols_texto = [c for c in df.columns if c not in ("Date", "ticker", "date")]
        if cols_texto:
            df = df.rename(columns={cols_texto[0]: "titulo"})
        else:
            raise ValueError("No se encontro columna de titulares en el parquet FNSPID")

    _df_noticias = df[["Date", "ticker", "titulo"]].copy()
    rango_inicio = _df_noticias["Date"].min().date()
    rango_fin    = _df_noticias["Date"].max().date()
    tickers_cub  = _df_noticias["ticker"].nunique()
    print(
        f"FNSPID cargado: {len(_df_noticias):,} noticias | "
        f"{rango_inicio} → {rango_fin} | "
        f"{tickers_cub} tickers"
    )
    return _df_noticias

# Herramientas

def herramienta_tecnica(ticker: str, datos: dict) -> str:
    df = datos[ticker].copy()

    delta    = df["Close"].diff()
    ganancia = delta.clip(lower=0).rolling(14).mean()
    perdida  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi      = 100 - (100 / (1 + ganancia / perdida))
    rsi_val  = round(float(rsi.iloc[-1]), 1)

    ema12       = df["Close"].ewm(span=12, adjust=False).mean()
    ema26       = df["Close"].ewm(span=26, adjust=False).mean()
    macd        = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_val    = round(float(macd.iloc[-1]), 4)
    signal_val  = round(float(macd_signal.iloc[-1]), 4)

    sma20    = df["Close"].rolling(20).mean()
    std20    = df["Close"].rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pct   = (df["Close"] - bb_lower) / (bb_upper - bb_lower)
    bb_val   = round(float(bb_pct.iloc[-1]), 2)

    precio = round(float(df["Close"].iloc[-1]), 2)
    return f"Precio:${precio} RSI:{rsi_val} MACD:{macd_val} Señal:{signal_val} BB:{bb_val}"


def herramienta_tecnica_historica(ticker: str, datos: dict, fecha: pd.Timestamp) -> str:
    """
    Igual que herramienta_tecnica pero usando datos hasta 'fecha' inclusive.
    Usada en backtesting para evitar look-ahead bias.
    """
    df = datos[ticker].copy()
    df = df[df.index <= fecha]
    if len(df) < 30:
        return f"Datos insuficientes para {ticker} en {fecha.date()}"

    delta    = df["Close"].diff()
    ganancia = delta.clip(lower=0).rolling(14).mean()
    perdida  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi      = 100 - (100 / (1 + ganancia / perdida))
    rsi_val  = round(float(rsi.iloc[-1]), 1)

    ema12       = df["Close"].ewm(span=12, adjust=False).mean()
    ema26       = df["Close"].ewm(span=26, adjust=False).mean()
    macd        = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_val    = round(float(macd.iloc[-1]), 4)
    signal_val  = round(float(macd_signal.iloc[-1]), 4)

    sma20    = df["Close"].rolling(20).mean()
    std20    = df["Close"].rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pct   = (df["Close"] - bb_lower) / (bb_upper - bb_lower)
    bb_val   = round(float(bb_pct.iloc[-1]), 2)

    precio = round(float(df["Close"].iloc[-1]), 2)
    return f"Precio:${precio} RSI:{rsi_val} MACD:{macd_val} Señal:{signal_val} BB:{bb_val}"


def herramienta_fundamental(ticker: str) -> str:
    try:
        info        = yf.Ticker(ticker).info
        pe          = round(info.get("trailingPE", 0), 2)
        roe         = round(info.get("returnOnEquity", 0) * 100, 1)
        crecimiento = round(info.get("revenueGrowth", 0) * 100, 1)
        deuda       = round(info.get("debtToEquity", 0), 2)
        margen      = round(info.get("profitMargins", 0) * 100, 1)
        return f"P/E:{pe} ROE:{roe}% Crec:{crecimiento}% Deuda/Cap:{deuda} Margen:{margen}%"
    except:
        return "Datos fundamentales no disponibles"


def herramienta_riesgo(ticker: str, datos: dict) -> str:
    retornos = datos[ticker]["Close"].pct_change().dropna()
    vol      = round(float(retornos.std() * np.sqrt(252)) * 100, 2)
    sharpe   = round(float((retornos.mean() - 0.05/252) / retornos.std() * np.sqrt(252)), 3)
    var95    = round(float(-np.percentile(retornos, 5)) * 100, 2)
    acumulado= (1 + retornos).cumprod()
    max_roll = acumulado.cummax()
    drawdown = (acumulado - max_roll) / max_roll
    max_dd   = round(float(drawdown.min()) * 100, 2)

    try:
        spy      = yf.download("SPY", start="2019-01-01", end="2024-12-31",
                               progress=False, auto_adjust=True)["Close"].squeeze()
        ret_spy  = spy.pct_change().dropna()
        alineado = pd.concat([retornos, ret_spy], axis=1).dropna()
        beta     = round(float(alineado.cov().iloc[0,1] / alineado.iloc[:,1].var()), 3)
        alpha    = round(float(retornos.mean()*252 - (0.05 + beta*(ret_spy.mean()*252 - 0.05))), 4)
    except:
        beta, alpha = 0, 0

    return f"Vol:{vol}% Sharpe:{sharpe} VaR95:{var95}% MaxDD:{max_dd}% Beta:{beta} Alpha:{alpha}"


def _proxy_sentimiento_precio(ticker: str, datos: dict, fecha: pd.Timestamp) -> str:
    """
    Proxy de sentimiento basado en momentum de precio.
    Usado cuando no hay noticias FNSPID disponibles:
      - Tickers sin cobertura: META, HON
      - Fechas posteriores a 2024-01-09
    Calcula retorno de los ultimos 5 y 20 dias de negociacion hasta 'fecha'.
    """
    df = datos.get(ticker)
    if df is None:
        return f"[PROXY PRECIO] Sin datos de precio para {ticker}"

    df_hist = df[df.index <= fecha]["Close"]
    if len(df_hist) < 21:
        return f"[PROXY PRECIO] Datos insuficientes para {ticker} en {fecha.date()}"

    ret_5d  = round(float((df_hist.iloc[-1] / df_hist.iloc[-6]  - 1) * 100), 2)
    ret_20d = round(float((df_hist.iloc[-1] / df_hist.iloc[-21] - 1) * 100), 2)

    tendencia = (
        "alcista" if ret_5d > 1 and ret_20d > 2
        else "bajista" if ret_5d < -1 and ret_20d < -2
        else "lateral"
    )
    return (
        f"[PROXY PRECIO — sin noticias FNSPID] "
        f"Retorno 5d: {ret_5d:+.2f}% | Retorno 20d: {ret_20d:+.2f}% | "
        f"Tendencia de precio: {tendencia}"
    )


def herramienta_sentimiento_historico(
    ticker: str,
    fecha: pd.Timestamp,
    datos: dict,
    ventana_dias: int = 7,
    max_noticias: int = 5,
    carpeta_datos: str = "../datos",
) -> str:
    """
    Herramienta de sentimiento para backtesting historico.
    Usa el dataset FNSPID en lugar de yfinance (tiempo real).

    Logica de fallback:
      1. Ticker sin cobertura FNSPID (META, HON)  → proxy de precio
      2. Fecha posterior a 2024-01-09              → proxy de precio
      3. Noticias encontradas en ventana 7 dias    → devuelve titulares
      4. Ventana ampliada a 30 dias si ventana=7   → intenta de nuevo
      5. Sin resultados tras ampliacion            → proxy de precio

    Parametros
    ----------
    ticker        : simbolo de la accion (e.g. "AAPL")
    fecha         : fecha de la señal (primer dia del mes en backtesting)
    datos         : dict {ticker: DataFrame} con precios historicos
    ventana_dias  : dias hacia atras para buscar noticias (por defecto 7)
    max_noticias  : numero maximo de titulares a devolver (por defecto 5)
    carpeta_datos : ruta al directorio de datos (para cargar FNSPID si no esta cacheado)

    Retorna
    -------
    str con titulares separados por ' | ' o descripcion del proxy de precio
    """
    fecha = pd.Timestamp(fecha)

    # Caso 1: ticker sin cobertura FNSPID 
    if ticker in TICKERS_SIN_FNSPID:
        return _proxy_sentimiento_precio(ticker, datos, fecha)

    # Caso 2: fecha fuera del rango FNSPID 
    if fecha > FNSPID_CUTOFF:
        return _proxy_sentimiento_precio(ticker, datos, fecha)

    # Cargar noticias si no estan en memoria 
    global _df_noticias
    if _df_noticias is None:
        try:
            cargar_noticias(carpeta_datos)
        except FileNotFoundError as e:
            return f"[ERROR FNSPID] {e}"

    #  Caso 3: busqueda en ventana inicial 
    fecha_inicio = fecha - timedelta(days=ventana_dias)
    mask = (
        (_df_noticias["ticker"] == ticker) &
        (_df_noticias["Date"] >= fecha_inicio) &
        (_df_noticias["Date"] <= fecha)
    )
    noticias = _df_noticias[mask].sort_values("Date", ascending=False)

    # Caso 4: ampliar ventana a 30 dias si no hay resultados 
    if noticias.empty and ventana_dias < 30:
        fecha_inicio_ampliada = fecha - timedelta(days=30)
        mask_ampliada = (
            (_df_noticias["ticker"] == ticker) &
            (_df_noticias["Date"] >= fecha_inicio_ampliada) &
            (_df_noticias["Date"] <= fecha)
        )
        noticias = _df_noticias[mask_ampliada].sort_values("Date", ascending=False)

    # Caso 5: sin resultados tras ampliacion → proxy de precio
    if noticias.empty:
        return _proxy_sentimiento_precio(ticker, datos, fecha)

    # Retorno normal: titulares encontrados 
    titulares = noticias["titulo"].dropna().head(max_noticias).tolist()
    titulares = [str(t).strip() for t in titulares if str(t).strip()]

    n_encontradas = len(noticias)
    prefijo = f"[FNSPID {n_encontradas} noticias]"
    return prefijo + " " + " | ".join(titulares)


def herramienta_sentimiento(ticker: str) -> str:
    """Sentimiento en TIEMPO REAL via yfinance. Solo para analisis del momento actual."""
    try:
        noticias = yf.Ticker(ticker).news
        if not noticias:
            return "Sin noticias recientes"
        titulares = [n.get("content", {}).get("title", "") for n in noticias[:5]]
        return " | ".join([t for t in titulares if t])
    except:
        return "Sin noticias disponibles"


def herramienta_montecarlo(ticker: str, datos: dict, dias: int = 20, simulaciones: int = 1000) -> str:
    retornos    = datos[ticker]["Close"].pct_change().dropna()
    mu          = float(retornos.mean())
    sigma       = float(retornos.std())
    precio_hoy  = float(datos[ticker]["Close"].iloc[-1])

    dt = 1/252
    trayectorias = np.zeros((dias, simulaciones))
    trayectorias[0] = precio_hoy

    for t in range(1, dias):
        ruido = np.random.normal(0, 1, simulaciones)
        trayectorias[t] = trayectorias[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * ruido
        )

    precios_finales = trayectorias[-1]
    prob_subida     = round(float((precios_finales > precio_hoy).mean()) * 100, 1)
    precio_esperado = round(float(precios_finales.mean()), 2)
    precio_p5       = round(float(np.percentile(precios_finales, 5)), 2)
    precio_p95      = round(float(np.percentile(precios_finales, 95)), 2)

    return (
        f"MonteCarlo({simulaciones} sim, {dias} dias): "
        f"Prob.subida:{prob_subida}% "
        f"Precio esperado:${precio_esperado} "
        f"Rango 90%:[${precio_p5} - ${precio_p95}]"
    )


def herramienta_montecarlo_historica(
    ticker: str, datos: dict, fecha: pd.Timestamp,
    dias: int = 20, simulaciones: int = 1000
) -> str:
    """
    Monte Carlo usando solo datos hasta 'fecha' inclusive.
    Evita look-ahead bias en backtesting.
    """
    df_hist = datos[ticker][datos[ticker].index <= fecha]["Close"]
    if len(df_hist) < 30:
        return f"Datos insuficientes para Monte Carlo de {ticker} en {fecha.date()}"

    retornos    = df_hist.pct_change().dropna()
    mu          = float(retornos.mean())
    sigma       = float(retornos.std())
    precio_hoy  = float(df_hist.iloc[-1])

    dt = 1/252
    trayectorias = np.zeros((dias, simulaciones))
    trayectorias[0] = precio_hoy

    for t in range(1, dias):
        ruido = np.random.normal(0, 1, simulaciones)
        trayectorias[t] = trayectorias[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * ruido
        )

    precios_finales = trayectorias[-1]
    prob_subida     = round(float((precios_finales > precio_hoy).mean()) * 100, 1)
    precio_esperado = round(float(precios_finales.mean()), 2)
    precio_p5       = round(float(np.percentile(precios_finales, 5)), 2)
    precio_p95      = round(float(np.percentile(precios_finales, 95)), 2)

    return (
        f"MonteCarlo({simulaciones} sim, {dias} dias): "
        f"Prob.subida:{prob_subida}% "
        f"Precio esperado:${precio_esperado} "
        f"Rango 90%:[${precio_p5} - ${precio_p95}]"
    )

# Agentes 

def agente_tecnico(ticker: str, datos: dict, llm, fecha: Optional[pd.Timestamp] = None) -> SeñalInversion:
    llm_ind = llm.with_structured_output(SeñalInversion)
    if fecha is not None:
        datos_tech = herramienta_tecnica_historica(ticker, datos, fecha)
    else:
        datos_tech = herramienta_tecnica(ticker, datos)
    prompt = f"""Eres un analista tecnico experto. Analiza estos indicadores de {ticker} y da una señal objetiva:
{datos_tech}
RSI>70=sobrecompra(SELL), RSI<30=sobreventa(BUY), MACD>Señal=alcista, BB>0.8=sobrecompra, BB<0.2=sobreventa"""
    return llm_ind.invoke(prompt)


def agente_fundamental(ticker: str, llm) -> SeñalInversion:
    llm_ind = llm.with_structured_output(SeñalInversion)
    datos_fund = herramienta_fundamental(ticker)
    prompt = f"""Eres un analista fundamental experto. Analiza estos ratios de {ticker} y da una señal objetiva:
{datos_fund}
P/E<20=barato(BUY), P/E>40=caro(SELL), ROE>15%=bueno, Crec>10%=bueno, Deuda/Cap<1=saludable"""
    return llm_ind.invoke(prompt)


def agente_riesgo(ticker: str, datos: dict, llm, fecha: Optional[pd.Timestamp] = None) -> SeñalInversion:
    llm_ind = llm.with_structured_output(SeñalInversion)
    # Riesgo siempre usa datos completos disponibles (no look-ahead en metricas de riesgo)
    datos_risk = herramienta_riesgo(ticker, datos)
    prompt = f"""Eres un gestor de riesgos experto. Analiza estas metricas de {ticker} y da una señal objetiva:
{datos_risk}
Sharpe>1=bueno(BUY), Sharpe<0=malo(SELL), Beta>1.5=muy volatil, Alpha>0=genera valor, MaxDD>-40%=alerta"""
    return llm_ind.invoke(prompt)


def agente_sentimiento(
    ticker: str,
    llm,
    fecha: Optional[pd.Timestamp] = None,
    datos: Optional[dict] = None,
    carpeta_datos: str = "../datos",
) -> SeñalInversion:
    """
    Agente de sentimiento.
    - Si fecha es None → usa yfinance (tiempo real, para analisis actual)
    - Si fecha es Timestamp → usa FNSPID historico (para backtesting)

    En modo historico, 'datos' es obligatorio para el proxy de precio.
    """
    llm_ind = llm.with_structured_output(SeñalInversion)

    if fecha is not None:
        if datos is None:
            raise ValueError("agente_sentimiento: 'datos' es obligatorio cuando se pasa 'fecha'")
        datos_sent = herramienta_sentimiento_historico(
            ticker, fecha, datos, carpeta_datos=carpeta_datos
        )
    else:
        datos_sent = herramienta_sentimiento(ticker)

    prompt = f"""Eres un analista de sentimiento experto. Analiza estas noticias de {ticker} y da una señal objetiva:
{datos_sent}
Evalua si el sentimiento es positivo(BUY), negativo(SELL) o neutro(HOLD)"""
    return llm_ind.invoke(prompt)


def agente_montecarlo(
    ticker: str, datos: dict, llm, fecha: Optional[pd.Timestamp] = None
) -> SeñalInversion:
    llm_ind = llm.with_structured_output(SeñalInversion)
    if fecha is not None:
        datos_mc = herramienta_montecarlo_historica(ticker, datos, fecha)
    else:
        datos_mc = herramienta_montecarlo(ticker, datos)
    prompt = f"""Eres un analista cuantitativo experto. Analiza esta simulacion Monte Carlo de {ticker} y da una señal objetiva:
{datos_mc}
Prob.subida>60%=BUY, Prob.subida<40%=SELL, entre 40-60%=HOLD."""
    return llm_ind.invoke(prompt)


def orquestador(
    ticker: str,
    datos: dict,
    llm,
    perfil: str = "moderado",
    fecha: Optional[pd.Timestamp] = None,
    carpeta_datos: str = "../datos",
) -> DecisionFinal:
    """
    Orquestador principal.
    - fecha=None  → analisis en tiempo real (usa datos actuales y noticias yfinance)
    - fecha=Timestamp → backtesting historico (usa datos hasta 'fecha' y FNSPID)
    """
    llm_dec  = llm.with_structured_output(DecisionFinal)
    ticker   = ticker.upper().strip()
    contexto = PERFILES[perfil]["descripcion"]

    s_tech = agente_tecnico(ticker, datos, llm, fecha=fecha)
    s_fund = agente_fundamental(ticker, llm)           # fundamentales: siempre tiempo real
    s_risk = agente_riesgo(ticker, datos, llm)
    s_sent = agente_sentimiento(ticker, llm, fecha=fecha, datos=datos, carpeta_datos=carpeta_datos)
    s_mc   = agente_montecarlo(ticker, datos, llm, fecha=fecha)

    prompt = f"""Eres un gestor de cartera experto.
Perfil del inversor: {contexto}

Has recibido 5 analisis objetivos de {ticker}:
Agente Tecnico     : {s_tech.decision} (confianza {s_tech.confianza}/10) — {s_tech.justificacion}
Agente Fundamental : {s_fund.decision} (confianza {s_fund.confianza}/10) — {s_fund.justificacion}
Agente Riesgo      : {s_risk.decision} (confianza {s_risk.confianza}/10) — {s_risk.justificacion}
Agente Sentimiento : {s_sent.decision} (confianza {s_sent.confianza}/10) — {s_sent.justificacion}
Agente Monte Carlo : {s_mc.decision}  (confianza {s_mc.confianza}/10)  — {s_mc.justificacion}

Teniendo en cuenta el perfil del inversor pondera las señales y toma una decision final."""

    return llm_dec.invoke(prompt)

# Ranking batch (tiempo real) 

def calcular_datos_batch(ticker: str, datos: dict) -> str:
    try:
        tech = herramienta_tecnica(ticker, datos)
        mc   = herramienta_montecarlo(ticker, datos)
        return f"{ticker}: {tech} | {mc}"
    except:
        return f"{ticker}: sin datos"


def ranking_50(tickers: list, datos: dict, llm, perfil: str = "moderado", batch_size: int = 5) -> list:
    llm_batch     = llm.with_structured_output(SeñalBatch)
    todas_señales = []
    grupos        = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    contexto      = PERFILES[perfil]["descripcion"]
    inicio_total  = time.time()

    for idx, grupo in enumerate(grupos):
        inicio_grupo = time.time()
        print(f"Procesando grupo {idx+1}/{len(grupos)}: {grupo}", end=" ")

        datos_grupo = "\n".join([calcular_datos_batch(t, datos) for t in grupo])
        prompt = f"""Perfil del inversor: {contexto}
Analiza estas acciones y devuelve una señal para cada una acorde al perfil:
{datos_grupo}"""

        try:
            resultado = llm_batch.invoke(prompt)
            todas_señales.extend(resultado.señales)
        except Exception as e:
            print(f"Error: {e}")

        fin_grupo = time.time()
        print(f"→ {round(fin_grupo-inicio_grupo,1)}s (acumulado: {round(fin_grupo-inicio_total,1)}s)")
        time.sleep(10)

    return todas_señales

# Generacion de señales mensuales (backtesting LLM)

def generar_señales_mensuales(
    datos: dict,
    llm,
    perfiles: list = ["conservador", "moderado", "agresivo"],
    carpeta_resultados: str = "../results",
    carpeta_datos: str = "../datos",
    año_inicio: int = 2019,
    año_fin: int = 2024,
) -> dict:
    """
    Genera señales LLM mes a mes para todos los tickers y perfiles indicados.
    Usa herramienta_sentimiento_historico automaticamente (FNSPID).
    Guarda resultados en CSV con checkpoint por mes para poder reanudar.

    Reanudacion automatica:
        Si existe el CSV de un perfil, lee los meses ya procesados y
        continua desde donde se quedo.

    Retorna dict {perfil: DataFrame} con todas las señales generadas.
    """
    os.makedirs(carpeta_resultados, exist_ok=True)

    # Cargar noticias una sola vez al inicio
    try:
        cargar_noticias(carpeta_datos)
    except FileNotFoundError as e:
        print(f"AVISO: {e}\nEl agente de sentimiento usara proxy de precio.")

    resultados = {}

    for perfil in perfiles:
        ruta_csv = os.path.join(carpeta_resultados, f"señales_mensuales_{perfil}.csv")

        # Cargar progreso previo 
        señales_existentes = []
        meses_procesados   = set()
        if os.path.exists(ruta_csv):
            df_prev = pd.read_csv(ruta_csv)
            señales_existentes = df_prev.to_dict("records")
            meses_procesados = {
                (int(r["año"]), int(r["mes"]))
                for r in señales_existentes
            }
            print(f"[{perfil}] Reanudando: {len(meses_procesados)}/72 meses ya procesados")
        else:
            print(f"[{perfil}] Iniciando desde cero")

        todas_señales = list(señales_existentes)
        inicio_total  = time.time()

        for año in range(año_inicio, año_fin + 1):
            for mes in range(1, 13):
                if (año, mes) in meses_procesados:
                    continue

                # Primer dia de mercado del mes (aproximacion: dia 1)
                fecha = pd.Timestamp(year=año, month=mes, day=1)

                print(f"\n[{perfil}] {año}-{mes:02d} ", end="", flush=True)
                inicio_mes = time.time()

                for ticker in TICKERS:
                    if ticker not in datos:
                        continue
                    try:
                        decision = orquestador(
                            ticker, datos, llm,
                            perfil=perfil,
                            fecha=fecha,
                            carpeta_datos=carpeta_datos,
                        )
                        todas_señales.append({
                            "año":         año,
                            "mes":         mes,
                            "fecha":       fecha.strftime("%Y-%m-%d"),
                            "ticker":      ticker,
                            "decision":    decision.decision,
                            "confianza":   decision.confianza,
                            "perfil":      perfil,
                            "justificacion": decision.justificacion,
                        })
                    except Exception as e:
                        print(f"[ERR {ticker}:{e}] ", end="", flush=True)

                # Guardar checkpoint tras cada mes
                df_checkpoint = pd.DataFrame(todas_señales)
                df_checkpoint.to_csv(ruta_csv, index=False)

                elapsed_mes = round(time.time() - inicio_mes, 1)
                elapsed_tot = round(time.time() - inicio_total, 1)
                procesados  = len(meses_procesados) + 1
                meses_procesados.add((año, mes))
                print(f"✓ {elapsed_mes}s | total {elapsed_tot}s | {procesados}/72")

        resultados[perfil] = pd.DataFrame(todas_señales)
        print(f"\n[{perfil}] Completado: {len(resultados[perfil])} señales → {ruta_csv}")

    return resultados