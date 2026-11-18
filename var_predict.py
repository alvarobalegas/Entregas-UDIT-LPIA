"""
var_predict.py — VaR 95 % a 1 día del SPY con el modelo Quantile GBM entrenado en el notebook.

Uso:
    python var_predict.py [FECHA] [CAPITAL]

    python var_predict.py 2020-03-16            -> VaR al cierre del 16-mar-2020, cartera de 100.000 €
    python var_predict.py 2020-03-16 250000     -> misma fecha, cartera de 250.000 €
    python var_predict.py                       -> última sesión disponible en el CSV

Para la fecha indicada (o la sesión hábil anterior, si es festivo o fin de semana) se calculan
las features con la información disponible al cierre de ese día y se predice el cuantil del 5 %
del retorno del día siguiente.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "spy.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "var_gbm.pkl")
DEFAULT_CAPITAL = 100_000.0


# -----------------------------------------------------------------------------
# Features: MISMA función que en el notebook (sección 3.4)
# -----------------------------------------------------------------------------
def build_features(prices):
    df = pd.DataFrame(index=prices.index)
    close, volume = prices["Close"], prices["Volume"]
    r = np.log(close).diff()

    df["ret"] = r
    df["ret_1d"] = r
    df["ret_5d"] = r.rolling(5).sum()
    df["vol_5d"] = r.rolling(5).std()
    df["vol_20d"] = r.rolling(20).std()
    df["vol_60d"] = r.rolling(60).std()
    df["vol_ratio"] = df["vol_5d"] / df["vol_60d"]
    df["drawdown"] = close / close.rolling(252, min_periods=1).max() - 1
    logv = np.log(volume.replace(0, np.nan))
    df["volume_z"] = (logv - logv.rolling(60).mean()) / logv.rolling(60).std()

    df["target"] = r.shift(-1)
    return df


def fail(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    # --- 1. Leer argumentos de la línea de comandos ---
    if len(sys.argv) > 3 or (len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help")):
        print(__doc__)
        sys.exit(0)

    fecha = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        capital = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAPITAL
    except ValueError:
        fail(f"capital no válido '{sys.argv[2]}'. Debe ser un número, p. ej. 100000.")
    if capital <= 0:
        fail("el capital debe ser positivo.")

    # --- 2. Cargar modelo y datos ---
    for path in (CSV_PATH, MODEL_PATH):
        if not os.path.exists(path):
            fail(f"no se encuentra '{path}'. Ejecuta antes el notebook para generarlo.")

    bundle = joblib.load(MODEL_PATH)
    model, features, alpha = bundle["model"], bundle["features"], bundle["alpha"]

    prices = pd.read_csv(CSV_PATH, parse_dates=["Date"], index_col="Date")
    prices = prices[["Close", "Volume"]].astype(float).sort_index()
    data = build_features(prices)

    # Benchmarks clásicos para comparar (mismas fórmulas que el notebook, sección 5.1)
    z = stats.norm.ppf(alpha)
    data["q_hist"] = data["ret"].rolling(250).quantile(alpha)
    data["q_normal"] = data["ret"].rolling(250).mean() + z * data["ret"].rolling(250).std()
    data["q_ewma"] = z * np.sqrt((data["ret"] ** 2).ewm(alpha=0.06, adjust=False).mean())

    valid = data.dropna(subset=features)

    # --- 3. Resolver la fecha ---
    if fecha is None:
        t = valid.index[-1]
    else:
        try:
            requested = pd.Timestamp(fecha)
        except ValueError:
            fail(f"fecha no válida '{fecha}'. Formato esperado: AAAA-MM-DD.")
        candidates = valid.index[valid.index <= requested]
        if len(candidates) == 0:
            fail(f"no hay datos suficientes antes de {requested.date()} "
                 f"(primera fecha válida: {valid.index[0].date()}).")
        t = candidates[-1]
        if requested > valid.index[-1]:
            print(f"Aviso: {requested.date()} es posterior al último dato; se usa {t.date()}.")
        elif t != requested:
            print(f"Aviso: {requested.date()} no fue sesión hábil; se usa {t.date()}.")

    # --- 4. Predecir ---
    row = valid.loc[[t]]
    q = float(model.predict(row[features])[0])   # cuantil 5 % del log-retorno (negativo)
    var_pct = 1 - np.exp(q)                      # pérdida en % "normal" (positiva)
    var_eur = capital * var_pct

    # --- 5. Mostrar resultados ---
    nivel = int(round((1 - alpha) * 100))
    print("=" * 62)
    print(f" VaR {nivel} % a 1 día · {bundle['ticker']} · al cierre del {t.date()}")
    print("=" * 62)
    print(f" Cartera:                 {capital:>14,.0f} €")
    print(f" Vol. realizada 20d:      {row['vol_20d'].iloc[0] * np.sqrt(252):>14.1%} (anualizada)")
    print(f" Drawdown (252d):         {row['drawdown'].iloc[0]:>14.1%}")
    print("-" * 62)
    print(f" Quantile GBM (modelo):   {var_pct:>8.2%}  →  {var_eur:>12,.0f} €")

    for label, col in [("Normal 250d", "q_normal"), ("Histórico 250d", "q_hist"),
                       ("EWMA RiskMetrics", "q_ewma")]:
        qb = row[col].iloc[0]
        if pd.notna(qb):
            loss = 1 - np.exp(qb)
            print(f" {label + ':':<25}{loss:>8.2%}  →  {capital * loss:>12,.0f} €")

    print("-" * 62)
    print(f" Interpretación: con un {nivel} % de confianza, la pérdida de mañana no")
    print(f" debería superar {var_eur:,.0f} € (1 de cada {round(1 / alpha)} días sí la superará).")

    # Si ya conocemos el día siguiente, comprobamos qué pasó de verdad
    realized = row["target"].iloc[0]
    if pd.notna(realized):
        next_day = data.index[data.index.get_loc(t) + 1].date()
        real_pct = np.exp(realized) - 1
        status = "VIOLACIÓN del VaR" if realized < q else "dentro del VaR"
        print(f" Resultado real {next_day}: {real_pct:+.2%} ({capital * real_pct:+,.0f} €) → {status}")

    if t <= pd.Timestamp(bundle["train_end"]):
        print(f" Nota: la fecha está dentro del periodo de entrenamiento (≤ {bundle['train_end']}).")
    print("=" * 62)


if __name__ == "__main__":
    main()
