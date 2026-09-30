"""

Arquitectura:
  - Agente Tecnico     → Random Forest especializado (features tecnicas)
  - Agente Fundamental → Random Forest especializado (features fundamentales)
  - Agente Riesgo      → Random Forest especializado (features de riesgo)
  - Agente Sentimiento → FinBERT (ProsusAI/finbert, CPU, 110M params)
  - Agente Monte Carlo → matematico puro (sin modelo)
  - Orquestador        → Mistral 7B (unico LLM; recibe 5 señales + perfil)

Ventajas frente a LLM puro:
  - Mas rapido: 4 de 5 agentes no llaman al LLM
  - Reproducible: RF y FinBERT son deterministas
  - Especializado: cada RF entrena solo con las features de su dominio
  - Mistral se reserva para la decision final de alto nivel

Uso basico:
    from sistema_hibrido import (
        AgentesV3, TICKERS, PERFILES,
        cargar_datos, cargar_noticias
    )

    # 1. Cargar datos de precio y noticias
    datos = cargar_datos()
    cargar_noticias()

    # 2. Inicializar y entrenar modelos
    sistema = AgentesV3()
    sistema.entrenar(datos)           # train 2010-2018, test 2019-2024

    # 3. Analizar una accion
    resultado = sistema.orquestador("AAPL", datos, perfil="moderado")

    # 4. Backtesting completo
    señales = sistema.generar_señales_mensuales(datos)
"""

import os
import time
import warnings
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Literal, List, Optional, Dict, Tuple

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

#  Configuracion 

MODELO_LLM   = "mistral:7b-instruct-v0.3-q4_0"
MODELO_NLP   = "ProsusAI/finbert"   # FinBERT para sentimiento
FNSPID_CUTOFF        = pd.Timestamp("2024-01-09")
TICKERS_SIN_FNSPID   = {"META", "HON"}

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "ORCL", "INTC", "QCOM",
    "JNJ",  "UNH",  "LLY",  "ABBV",  "MRK",  "TMO",  "ABT", "PFE",  "MDT",  "BMY",
    "JPM",  "BAC",  "WFC",  "GS",    "MS",   "BLK",  "AXP", "C",    "USB",  "PNC",
    "AMZN", "TSLA", "HD",   "MCD",   "NKE",  "SBUX", "TGT", "LOW",  "F",    "GM",
    "XOM",  "CVX",  "COP",  "SLB",   "EOG",
    "CAT",  "BA",   "HON",  "GE",    "LMT",
]

NOMBRES = {
    "AAPL": "Apple",            "MSFT": "Microsoft",       "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",        "META": "Meta",             "AVGO": "Broadcom",
    "AMD": "AMD",               "ORCL": "Oracle",           "INTC": "Intel",
    "QCOM": "Qualcomm",         "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth",
    "LLY": "Eli Lilly",         "ABBV": "AbbVie",           "MRK": "Merck",
    "TMO": "Thermo Fisher",     "ABT": "Abbott",            "PFE": "Pfizer",
    "MDT": "Medtronic",         "BMY": "Bristol-Myers",     "JPM": "JPMorgan",
    "BAC": "Bank of America",   "WFC": "Wells Fargo",       "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",     "BLK": "BlackRock",         "AXP": "American Express",
    "C": "Citigroup",           "USB": "U.S. Bancorp",      "PNC": "PNC Financial",
    "AMZN": "Amazon",           "TSLA": "Tesla",            "HD": "Home Depot",
    "MCD": "McDonald's",        "NKE": "Nike",              "SBUX": "Starbucks",
    "TGT": "Target",            "LOW": "Lowe's",            "F": "Ford",
    "GM": "General Motors",     "XOM": "ExxonMobil",        "CVX": "Chevron",
    "COP": "ConocoPhillips",    "SLB": "Schlumberger",      "EOG": "EOG Resources",
    "CAT": "Caterpillar",       "BA": "Boeing",             "HON": "Honeywell",
    "GE": "General Electric",   "LMT": "Lockheed Martin",
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

# Features por dominio
# Cada RF especializado usa solo sus features para evitar filtraciones entre dominios
FEATURES_TECNICO = [
    "rsi", "macd", "macd_signal", "macd_hist", "bb_pct",
    "sma20_ratio", "sma50_ratio", "sma200_ratio", "retorno_1d", "retorno_5d",
]
FEATURES_RIESGO = [
    "volatilidad_20d", "volatilidad_60d", "sharpe_60d",
    "max_drawdown_60d", "beta_60d", "var95_20d", "retorno_20d",
]
# Fundamental: yfinance no da series temporales gratuitas para fundamentales historicos,
# asi que el RF fundamental usa un subconjunto de features de precio que aproximan
# salud empresarial relativa: momentum largo plazo y comparativa con el mercado.
FEATURES_FUNDAMENTAL = [
    "retorno_60d", "retorno_120d", "retorno_252d",
    "ratio_sma50_sma200",   # golden/death cross
    "volatilidad_252d",
    "rsi",                  # sobrecompra/venta en contexto fundamental
]

#  Modelos Pydantic 

class SeñalAgente(BaseModel):
    ticker: str = Field(description="Simbolo de la accion en mayusculas")
    decision: Literal["BUY", "HOLD", "SELL"] = Field(
        description="Decision de inversion. SOLO puede ser BUY, HOLD o SELL en mayusculas"
    )
    confianza: int = Field(ge=1, le=10, description="Nivel de confianza del 1 al 10")
    justificacion: str = Field(
        description="Justificacion breve en maximo 15 palabras en español"
    )

class _DecisionLLM(BaseModel):
    """Schema interno que rellena Mistral. Sin ticker para evitar alucinaciones."""
    decision: Literal["BUY", "HOLD", "SELL"] = Field(
        description="SOLO una de estas tres palabras: BUY, HOLD o SELL"
    )
    confianza: int = Field(ge=1, le=10, description="Numero entero del 1 al 10")
    justificacion: str = Field(
        description="Maximo 10 palabras en español explicando la decision"
    )

class DecisionFinal(BaseModel):
    """Objeto de retorno publico. ticker lo asigna Python, no el LLM."""
    ticker: str
    decision: Literal["BUY", "HOLD", "SELL"]
    confianza: int
    justificacion: str

#  Carga de datos y noticias 

def cargar_datos(carpeta: str = "../datos") -> dict:
    datos = {}
    for ticker in TICKERS:
        ruta = f"{carpeta}/{ticker}.parquet"
        if os.path.exists(ruta):
            datos[ticker] = pd.read_parquet(ruta)
    print(f"Datos cargados: {len(datos)}/{len(TICKERS)} tickers")
    return datos

_df_noticias: Optional[pd.DataFrame] = None

def cargar_noticias(carpeta: str = "../datos") -> pd.DataFrame:
    """Carga el dataset FNSPID una vez y lo cachea en memoria."""
    global _df_noticias
    if _df_noticias is not None:
        print(f"Noticias ya en memoria: {len(_df_noticias):,} registros")
        return _df_noticias

    ruta = os.path.join(carpeta, "fnspid_50tickers.parquet")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Dataset FNSPID no encontrado en '{ruta}'")

    df = pd.read_parquet(ruta)
    df["Date"] = pd.to_datetime(df["Date"])

    if "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    if "Article_title" in df.columns:
        df = df.rename(columns={"Article_title": "titulo"})
    elif "title" in df.columns:
        df = df.rename(columns={"title": "titulo"})
    elif "titulo" not in df.columns:
        cols = [c for c in df.columns if c not in ("Date", "ticker", "date")]
        df = df.rename(columns={cols[0]: "titulo"})

    _df_noticias = df[["Date", "ticker", "titulo"]].copy()
    print(
        f"FNSPID cargado: {len(_df_noticias):,} noticias | "
        f"{_df_noticias['Date'].min().date()} → {_df_noticias['Date'].max().date()}"
    )
    return _df_noticias

#  Construccion de features 

def _construir_features_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todas las features para un ticker a partir de su DataFrame de precios.
    Devuelve un DataFrame con una fila por dia.
    Sin look-ahead: todos los calculos usan datos pasados.
    """
    out = pd.DataFrame(index=df.index)
    c = df["Close"]

    #  Features tecnicas 
    delta    = c.diff()
    ganancia = delta.clip(lower=0).rolling(14).mean()
    perdida  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi      = 100 - (100 / (1 + ganancia / perdida))
    out["rsi"] = rsi

    ema12       = c.ewm(span=12, adjust=False).mean()
    ema26       = c.ewm(span=26, adjust=False).mean()
    macd        = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"]        = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"]   = macd - macd_signal

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200= c.rolling(200).mean()
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    out["bb_pct"]       = (c - bb_lower) / (bb_upper - bb_lower)
    out["sma20_ratio"]  = c / sma20   - 1   # % sobre/bajo SMA20
    out["sma50_ratio"]  = c / sma50   - 1
    out["sma200_ratio"] = c / sma200  - 1
    out["ratio_sma50_sma200"] = sma50 / sma200 - 1   # golden/death cross

    ret = c.pct_change()
    out["retorno_1d"]  = ret
    out["retorno_5d"]  = c.pct_change(5)
    out["retorno_20d"] = c.pct_change(20)
    out["retorno_60d"] = c.pct_change(60)
    out["retorno_120d"]= c.pct_change(120)
    out["retorno_252d"]= c.pct_change(252)

    #  Features de riesgo 
    out["volatilidad_20d"]  = ret.rolling(20).std()  * np.sqrt(252)
    out["volatilidad_60d"]  = ret.rolling(60).std()  * np.sqrt(252)
    out["volatilidad_252d"] = ret.rolling(252).std() * np.sqrt(252)

    # Sharpe rolling 60d (rf diaria = 0.05/252)
    rf_diaria = 0.05 / 252
    out["sharpe_60d"] = (
        (ret.rolling(60).mean() - rf_diaria) /
        ret.rolling(60).std().replace(0, np.nan)
    ) * np.sqrt(252)

    # VaR 95% rolling 20d
    out["var95_20d"] = ret.rolling(20).quantile(0.05) * -1

    # Max drawdown rolling 60d
    def rolling_maxdd(s, w=60):
        result = pd.Series(np.nan, index=s.index)
        for i in range(w, len(s)):
            ventana = s.iloc[i-w:i]
            acum = (1 + ventana).cumprod()
            dd   = (acum / acum.cummax() - 1).min()
            result.iloc[i] = dd
        return result

    out["max_drawdown_60d"] = rolling_maxdd(ret)

    # Beta rolling 60d vs SPY (se calcula en AgentesV3.entrenar para evitar descargas repetidas)
    # Se rellena con 1.0 por defecto; entrenar() lo sobreescribe con datos reales
    out["beta_60d"] = 1.0

    return out


def construir_dataset(datos: dict, ret_spy: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Construye el dataset completo con features y etiqueta para todos los tickers.
    ret_spy: retornos diarios de SPY para calcular beta (opcional).
    Etiqueta: BUY(1) si retorno 20d > 5%, SELL(-1) si < -5%, HOLD(0) resto.
    """
    registros = []

    for ticker in TICKERS:
        if ticker not in datos:
            continue
        df = datos[ticker].copy()
        feats = _construir_features_ticker(df)

        # Beta rolling 60d real si tenemos SPY
        if ret_spy is not None:
            ret = df["Close"].pct_change()
            alin = pd.concat([ret, ret_spy], axis=1).dropna()
            alin.columns = ["ticker", "spy"]
            beta_series = alin["ticker"].rolling(60).cov(alin["spy"]) / alin["spy"].rolling(60).var()
            feats["beta_60d"] = beta_series.reindex(feats.index)

        # Etiqueta sin look-ahead
        retorno_futuro = df["Close"].shift(-20) / df["Close"] - 1
        etiqueta = pd.Series(0, index=df.index)
        etiqueta[retorno_futuro >  0.05] =  1
        etiqueta[retorno_futuro < -0.05] = -1

        feats["etiqueta"] = etiqueta
        feats["ticker"]   = ticker
        feats["fecha"]    = df.index

        # Eliminar primeras filas sin datos suficientes y las ultimas 20 (etiqueta futura)
        feats = feats.dropna(how="all")
        registros.append(feats)

    dataset = pd.concat(registros, ignore_index=True)
    dataset = dataset.dropna(subset=FEATURES_TECNICO + FEATURES_RIESGO + FEATURES_FUNDAMENTAL)
    return dataset

#  FinBERT  Agente de Sentimiento 

class AgenteSentimientoFinBERT:
    """
    Agente de sentimiento basado en FinBERT (ProsusAI/finbert).
    Clasifica titulares financieros en positive / negative / neutral.
    Corre en CPU; tarda ~0.1s por titular.
    """

    def __init__(self):
        self._pipeline = None   # lazy loading

    def _cargar(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            print("Cargando FinBERT (primera vez, puede tardar ~30s)...")
            self._pipeline = pipeline(
                "text-classification",
                model=MODELO_NLP,
                tokenizer=MODELO_NLP,
                device=-1,          # -1 = CPU
                truncation=True,
                max_length=512,
            )
            print("FinBERT cargado.")
        except ImportError:
            raise ImportError(
                "FinBERT requiere transformers y torch.\n"
                "Instala con: pip install transformers torch"
            )

    def _score_titulares(self, titulares: List[str]) -> Tuple[float, int]:
        """
        Clasifica una lista de titulares en batch (una sola llamada al pipeline)
        y devuelve (score_neto, n_titulares).
        score_neto: promedio ponderado entre -1 (negativo) y +1 (positivo).
        """
        if not titulares:
            return 0.0, 0

        self._cargar()

        # Truncar cada titular a 512 chars y procesar en batch
        titulares_limpios = [t[:512] for t in titulares if t.strip()]
        if not titulares_limpios:
            return 0.0, 0

        try:
            resultados = self._pipeline(titulares_limpios, batch_size=len(titulares_limpios))
        except Exception:
            return 0.0, 0

        score_total = 0.0
        for res in resultados:
            label = res["label"].lower()
            prob  = res["score"]
            if label == "positive":
                score_total += prob
            elif label == "negative":
                score_total -= prob

        return score_total / len(titulares_limpios), len(titulares_limpios)

    def _obtener_titulares_historicos(
        self, ticker: str, fecha: pd.Timestamp, datos: dict
    ) -> List[str]:
        """
        Obtiene titulares del dataset FNSPID para el ticker y fecha dados.
        Fallback a proxy de precio si no hay cobertura.
        """
        global _df_noticias

        # Sin cobertura FNSPID o fuera de rango → proxy precio
        if ticker in TICKERS_SIN_FNSPID or fecha > FNSPID_CUTOFF:
            return []   # señal neutra; el score sera 0

        if _df_noticias is None:
            return []

        # Ventana 7 dias, ampliar a 30 si vacia
        for ventana in [7, 30]:
            mask = (
                (_df_noticias["ticker"] == ticker) &
                (_df_noticias["Date"] >= fecha - timedelta(days=ventana)) &
                (_df_noticias["Date"] <= fecha)
            )
            titulares = _df_noticias[mask]["titulo"].dropna().tolist()
            if titulares:
                return [str(t) for t in titulares[:5]]

        return []

    def señal(
        self,
        ticker: str,
        fecha: Optional[pd.Timestamp] = None,
        datos: Optional[dict] = None,
    ) -> SeñalAgente:
        """
        Genera una señal de sentimiento.
        - fecha=None  → titulares de yfinance (tiempo real)
        - fecha=Timestamp → titulares de FNSPID (backtesting)
        """
        if fecha is not None and datos is not None:
            titulares = self._obtener_titulares_historicos(ticker, fecha, datos)
            fuente = "FNSPID"
        else:
            # Tiempo real via yfinance
            try:
                noticias  = yf.Ticker(ticker).news
                titulares = [
                    n.get("content", {}).get("title", "")
                    for n in (noticias or [])[:5]
                ]
                titulares = [t for t in titulares if t]
            except Exception:
                titulares = []
            fuente = "yfinance"

        if not titulares:
            # Sin noticias: señal neutra con baja confianza
            return SeñalAgente(
                ticker=ticker,
                decision="HOLD",
                confianza=3,
                justificacion=f"Sin noticias {fuente} disponibles para {ticker}",
            )

        score, n = self._score_titulares(titulares)

        # Convertir score a decision
        if score > 0.15:
            decision, confianza = "BUY", min(10, int(5 + score * 10))
        elif score < -0.15:
            decision, confianza = "SELL", min(10, int(5 + abs(score) * 10))
        else:
            decision, confianza = "HOLD", max(3, int(5 - abs(score) * 5))

        return SeñalAgente(
            ticker=ticker,
            decision=decision,
            confianza=confianza,
            justificacion=f"FinBERT score={score:+.2f} sobre {n} titulares ({fuente})",
        )

#  Random Forest Especializado 

class AgenteRF:
    """
    Random Forest especializado en un dominio concreto.
    Entrena con las features de su dominio y devuelve BUY/HOLD/SELL + probabilidad.
    """

    PARAM_GRID = {
        "n_estimators":     [100, 200],
        "max_depth":        [5, 8, None],
        "min_samples_leaf": [50, 100],
        "max_features":     [0.5, 0.7],
    }

    def __init__(self, nombre: str, features: List[str]):
        self.nombre   = nombre
        self.features = features
        self.modelo   = None
        self.entrenado = False

    def entrenar(
        self,
        dataset: pd.DataFrame,
        año_train_fin: int = 2018,
        verbose: bool = True,
    ) -> dict:
        """
        Entrena el RF con datos hasta año_train_fin (inclusive).
        Evalua en datos posteriores.
        """
        feats_disponibles = [f for f in self.features if f in dataset.columns]
        if not feats_disponibles:
            raise ValueError(f"[{self.nombre}] Ninguna feature disponible en el dataset")

        self.features = feats_disponibles   # actualizar con lo disponible

        # Split temporal
        mask_train = dataset["fecha"].dt.year <= año_train_fin
        mask_test  = dataset["fecha"].dt.year >  año_train_fin

        X_train = dataset.loc[mask_train, self.features].fillna(0)
        y_train = dataset.loc[mask_train, "etiqueta"]
        X_test  = dataset.loc[mask_test,  self.features].fillna(0)
        y_test  = dataset.loc[mask_test,  "etiqueta"]

        if verbose:
            print(f"[{self.nombre}] Train: {len(X_train):,} | Test: {len(X_test):,}")

        # GridSearchCV con TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        rf   = RandomForestClassifier(random_state=42, n_jobs=-1)
        gs   = GridSearchCV(rf, self.PARAM_GRID, cv=tscv,
                            scoring="f1_macro", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)

        self.modelo    = gs.best_estimator_
        self.entrenado = True

        y_pred    = self.modelo.predict(X_test)
        acc_test  = (y_pred == y_test).mean()
        acc_train = (self.modelo.predict(X_train) == y_train).mean()

        if verbose:
            print(f"[{self.nombre}] Mejores params: {gs.best_params_}")
            print(f"[{self.nombre}] Acc train={acc_train:.3f} test={acc_test:.3f}")

        return {
            "nombre":      self.nombre,
            "acc_train":   acc_train,
            "acc_test":    acc_test,
            "best_params": gs.best_params_,
            "report":      classification_report(
                y_test, y_pred,
                target_names=["SELL", "HOLD", "BUY"],
                output_dict=True,
            ),
        }

    def señal(
        self,
        ticker: str,
        features_row: Dict[str, float],
    ) -> SeñalAgente:
        """
        Genera una señal para un ticker dado un diccionario de features actuales.
        """
        if not self.entrenado or self.modelo is None:
            raise RuntimeError(f"[{self.nombre}] Modelo no entrenado. Llama a entrenar() primero.")

        X = pd.DataFrame([features_row])[self.features].fillna(0)
        pred   = int(self.modelo.predict(X)[0])
        probas = self.modelo.predict_proba(X)[0]

        clases = list(self.modelo.classes_)   # [-1, 0, 1]
        prob_max = float(probas.max())

        decision_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
        decision = decision_map.get(pred, "HOLD")
        confianza = max(1, min(10, int(prob_max * 10)))

        return SeñalAgente(
            ticker=ticker,
            decision=decision,
            confianza=confianza,
            justificacion=(
                f"RF {self.nombre}: {decision} "
                f"(prob={prob_max:.2f}, features={len(self.features)})"
            ),
        )

#  Monte Carlo puro 

def _señal_montecarlo(
    ticker: str,
    datos: dict,
    fecha: Optional[pd.Timestamp] = None,
    dias: int = 20,
    simulaciones: int = 1000,
) -> SeñalAgente:
    """Monte Carlo sin LLM. Umbral directo sobre prob_subida."""
    df = datos[ticker]
    if fecha is not None:
        df = df[df.index <= fecha]

    retornos   = df["Close"].pct_change().dropna()
    mu         = float(retornos.mean())
    sigma      = float(retornos.std())
    precio_hoy = float(df["Close"].iloc[-1])

    dt = 1 / 252
    rng = np.random.default_rng(seed=42)  # reproducible
    tray = np.zeros((dias, simulaciones))
    tray[0] = precio_hoy

    for t in range(1, dias):
        ruido  = rng.standard_normal(simulaciones)
        tray[t] = tray[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * ruido
        )

    precios_finales = tray[-1]
    prob_subida     = float((precios_finales > precio_hoy).mean())
    precio_p5       = float(np.percentile(precios_finales, 5))
    precio_p95      = float(np.percentile(precios_finales, 95))

    if prob_subida >= 0.60:
        decision, confianza = "BUY",  min(10, int(prob_subida * 10))
    elif prob_subida <= 0.40:
        decision, confianza = "SELL", min(10, int((1 - prob_subida) * 10))
    else:
        decision, confianza = "HOLD", max(3, int(abs(prob_subida - 0.5) * 20))

    return SeñalAgente(
        ticker=ticker,
        decision=decision,
        confianza=confianza,
        justificacion=(
            f"MonteCarlo prob_subida={prob_subida:.0%} "
            f"rango90=[{precio_p5:.0f},{precio_p95:.0f}]"
        ),
    )

#  Sistema hibrido completo 

class AgentesV3:
    """
    Sistema multiagente v3.
    Combina 3 RF especializados + FinBERT + Monte Carlo + Mistral (solo orquestador).
    """

    def __init__(self, carpeta_modelos: str = "../results/modelos_v3"):
        self.carpeta_modelos = carpeta_modelos
        os.makedirs(carpeta_modelos, exist_ok=True)

        # Tres RF especializados
        self.rf_tecnico     = AgenteRF("Tecnico",     FEATURES_TECNICO)
        self.rf_fundamental = AgenteRF("Fundamental", FEATURES_FUNDAMENTAL)
        self.rf_riesgo      = AgenteRF("Riesgo",      FEATURES_RIESGO)

        # FinBERT (lazy loading)
        self.finbert = AgenteSentimientoFinBERT()

        # LLM solo para el orquestador
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=MODELO_LLM,
                temperature=0,
                base_url="http://localhost:11434",
                num_predict=120,   # max tokens: suficiente para ticker+decision+confianza+justificacion
            )
        return self._llm

    #  Entrenamiento 
    def entrenar(
        self,
        datos: dict,
        año_train_fin: int = 2018,
        forzar: bool = False,
        carpeta_datos_train: str = "../datos",
    ) -> Dict[str, dict]:
        """
        Entrena los 3 RF especializados.
        Si ya existen los PKL guardados y forzar=False, los carga directamente.

        Parametros
        ----------
        datos            : dict de DataFrames con precios 2010-2024
        año_train_fin    : ultimo año de entrenamiento (2018 por defecto)
        forzar           : si True, reentrena aunque existan los PKL
        carpeta_datos_train : donde buscar datos de entrenamiento extendidos (2010-2018)
        """
        ruta_tecnico     = os.path.join(self.carpeta_modelos, "rf_tecnico.pkl")
        ruta_fundamental = os.path.join(self.carpeta_modelos, "rf_fundamental.pkl")
        ruta_riesgo      = os.path.join(self.carpeta_modelos, "rf_riesgo.pkl")
        ruta_dataset     = os.path.join(self.carpeta_modelos, "dataset_v3.parquet")

        # Cargar modelos ya entrenados si existen
        if not forzar and all(os.path.exists(r) for r in [ruta_tecnico, ruta_fundamental, ruta_riesgo]):
            print("Cargando modelos RF v3 desde disco...")
            self.rf_tecnico.modelo     = joblib.load(ruta_tecnico)
            self.rf_fundamental.modelo = joblib.load(ruta_fundamental)
            self.rf_riesgo.modelo      = joblib.load(ruta_riesgo)
            self.rf_tecnico.entrenado     = True
            self.rf_fundamental.entrenado = True
            self.rf_riesgo.entrenado      = True
            print("Modelos cargados.")
            return {}

        # Cargar datos de entrenamiento extendidos (2010-2018) si existen
        datos_train = {}
        for ticker in TICKERS:
            ruta_train = os.path.join(carpeta_datos_train, f"{ticker}_train.parquet")
            if os.path.exists(ruta_train):
                datos_train[ticker] = pd.read_parquet(ruta_train)
        if datos_train:
            print(f"Datos de entrenamiento 2010-2018 cargados: {len(datos_train)} tickers")
            # Concatenar con datos 2019-2024 para tener el periodo completo
            datos_completos = {}
            for ticker in TICKERS:
                partes = []
                if ticker in datos_train:
                    partes.append(datos_train[ticker])
                if ticker in datos:
                    partes.append(datos[ticker])
                if partes:
                    datos_completos[ticker] = pd.concat(partes).sort_index().drop_duplicates()
        else:
            print("AVISO: Sin datos 2010-2018. Entrenando solo con 2019-2024.")
            datos_completos = datos

        # Descargar SPY para calcular beta
        print("Descargando SPY para calcular beta...")
        try:
            spy = yf.download(
                "SPY",
                start="2010-01-01", end="2024-12-31",
                progress=False, auto_adjust=True,
            )["Close"].squeeze()
            ret_spy = spy.pct_change().dropna()
        except Exception:
            ret_spy = None
            print("AVISO: No se pudo descargar SPY. Beta se calculara sin referencia.")

        # Construir dataset
        print("Construyendo dataset de features...")
        if os.path.exists(ruta_dataset) and not forzar:
            dataset = pd.read_parquet(ruta_dataset)
            print(f"Dataset cargado desde disco: {dataset.shape}")
        else:
            dataset = construir_dataset(datos_completos, ret_spy=ret_spy)
            dataset.to_parquet(ruta_dataset, index=False)
            print(f"Dataset construido y guardado: {dataset.shape}")

        # Entrenar 3 RF
        resultados = {}
        print("\n── Entrenando RF Tecnico ──")
        resultados["tecnico"] = self.rf_tecnico.entrenar(dataset, año_train_fin)
        joblib.dump(self.rf_tecnico.modelo, ruta_tecnico)

        print("\n── Entrenando RF Fundamental ──")
        resultados["fundamental"] = self.rf_fundamental.entrenar(dataset, año_train_fin)
        joblib.dump(self.rf_fundamental.modelo, ruta_fundamental)

        print("\n── Entrenando RF Riesgo ──")
        resultados["riesgo"] = self.rf_riesgo.entrenar(dataset, año_train_fin)
        joblib.dump(self.rf_riesgo.modelo, ruta_riesgo)

        print("\n✓ Entrenamiento completado.")
        return resultados

    # Features actuales para un ticker 

    def _features_actuales(self, ticker: str, datos: dict, fecha: Optional[pd.Timestamp] = None) -> Dict[str, float]:
        df = datos[ticker]
        if fecha is not None:
            df = df[df.index <= fecha]
    
        feats = _construir_features_ticker(df)
        feats = feats.dropna(subset=self.rf_tecnico.features[:3])  # minimo RSI+MACD calculados
        if feats.empty:
            return {}
    
        row = feats.iloc[-1]
        return {col: float(row[col]) if pd.notna(row[col]) else 0.0
                for col in feats.columns
                if col not in ("fecha", "ticker", "etiqueta")}

    # Orquestador principal 

    def orquestador(
        self,
        ticker: str,
        datos: dict,
        perfil: str = "moderado",
        fecha: Optional[pd.Timestamp] = None,
        carpeta_datos: str = "../datos",
    ) -> DecisionFinal:
        """
        Genera una decision final para el ticker dado.
        - fecha=None  → analisis en tiempo real
        - fecha=Timestamp → backtesting historico
        """
        ticker = ticker.upper().strip()

        # Extraer features
        feats = self._features_actuales(ticker, datos, fecha)

        # Señales de los 3 RF
        s_tech  = self.rf_tecnico.señal(ticker, feats)
        s_fund  = self.rf_fundamental.señal(ticker, feats)
        s_riesgo= self.rf_riesgo.señal(ticker, feats)

        # FinBERT
        s_sent = self.finbert.señal(ticker, fecha=fecha, datos=datos)

        # Monte Carlo
        s_mc = _señal_montecarlo(ticker, datos, fecha=fecha)

        # Orquestador LLM — unica llamada a Mistral
        llm     = self._get_llm()
        llm_dec = llm.with_structured_output(_DecisionLLM)
        contexto = PERFILES[perfil]["descripcion"]

        # Votos resumidos para reducir tokens de entrada
        votos = (
            f"Tech:{s_tech.decision}({s_tech.confianza}) "
            f"Fund:{s_fund.decision}({s_fund.confianza}) "
            f"Risk:{s_riesgo.decision}({s_riesgo.confianza}) "
            f"Sent:{s_sent.decision}({s_sent.confianza}) "
            f"MC:{s_mc.decision}({s_mc.confianza})"
        )

        prompt = (
            f"Gestor de cartera. Perfil: {contexto}\n"
            f"Señales: {votos}\n"
            f"Decide BUY, HOLD o SELL segun el perfil. "
            f"Justifica en maximo 10 palabras en español."
        )

        # Mistral rellena decision+confianza+justificacion; ticker lo asigna Python
        llm_out = llm_dec.invoke(prompt)
        return DecisionFinal(
            ticker=ticker,
            decision=llm_out.decision,
            confianza=llm_out.confianza,
            justificacion=llm_out.justificacion,
        )

    # Generacion masiva de señales (backtesting) 

    def generar_señales_mensuales(
        self,
        datos: dict,
        perfiles: list = ["conservador", "moderado", "agresivo"],
        carpeta_resultados: str = "../results",
        carpeta_datos: str = "../datos",
        año_inicio: int = 2019,
        año_fin: int = 2024,
    ) -> dict:
        """
        Genera señales mes a mes para todos los tickers y perfiles.
        Guarda CSV con checkpoint por mes para poder reanudar.
        Prefijo de archivo: 'v3_señales_mensuales_{perfil}.csv'
        """
        os.makedirs(carpeta_resultados, exist_ok=True)

        try:
            cargar_noticias(carpeta_datos)
        except FileNotFoundError as e:
            print(f"AVISO: {e}\nFinBERT usara señal neutra para todos los tickers.")

        resultados = {}

        for perfil in perfiles:
            ruta_csv = os.path.join(
                carpeta_resultados, f"v3_señales_mensuales_{perfil}.csv"
            )

            # Reanudar si existe checkpoint
            señales_existentes = []
            meses_procesados   = set()
            if os.path.exists(ruta_csv):
                df_prev = pd.read_csv(ruta_csv)
                señales_existentes = df_prev.to_dict("records")
                meses_procesados   = {
                    (int(r["año"]), int(r["mes"])) for r in señales_existentes
                }
                print(f"[v3/{perfil}] Reanudando: {len(meses_procesados)}/72 meses")
            else:
                print(f"[v3/{perfil}] Iniciando desde cero")

            todas_señales = list(señales_existentes)
            inicio_total  = time.time()

            for año in range(año_inicio, año_fin + 1):
                for mes in range(1, 13):
                    if (año, mes) in meses_procesados:
                        continue

                    fecha = pd.Timestamp(year=año, month=mes, day=1)
                    print(f"\n[v3/{perfil}] {año}-{mes:02d} ", end="", flush=True)
                    inicio_mes = time.time()

                    for ticker in TICKERS:
                        if ticker not in datos:
                            continue
                        try:
                            decision = self.orquestador(
                                ticker, datos,
                                perfil=perfil,
                                fecha=fecha,
                                carpeta_datos=carpeta_datos,
                            )
                            todas_señales.append({
                                "año":          año,
                                "mes":          mes,
                                "fecha":        fecha.strftime("%Y-%m-%d"),
                                "ticker":       ticker,
                                "decision":     decision.decision,
                                "confianza":    decision.confianza,
                                "perfil":       perfil,
                                "justificacion": decision.justificacion,
                                "version":      "v3",
                            })
                        except Exception as e:
                            print(f"[ERR {ticker}:{e}] ", end="", flush=True)

                    # Checkpoint tras cada mes
                    pd.DataFrame(todas_señales).to_csv(ruta_csv, index=False)
                    meses_procesados.add((año, mes))
                    t_mes = round(time.time() - inicio_mes, 1)
                    t_tot = round(time.time() - inicio_total, 1)
                    print(f"✓ {t_mes}s | total {t_tot}s | {len(meses_procesados)}/72")

            resultados[perfil] = pd.DataFrame(todas_señales)
            print(f"\n[v3/{perfil}] Completado → {ruta_csv}")

        return resultados