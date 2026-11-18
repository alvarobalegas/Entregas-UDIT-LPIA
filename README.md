# Clase 4 · EDA y Entreno — Tail Risk y VaR con Machine Learning

**Máster Universitario en Inteligencia Artificial (UDIT)** · Lenguajes de Programación para IA  
**Autor:** Álvaro Manuel Balegas López

## Descripción

Análisis del riesgo de cola del S&P 500 (ETF **SPY**, 2005–2025) y estimación del **Valor en Riesgo (VaR) al 95 %** a 1 día con un modelo de **regresión cuantílica**.

En lugar de suponer retornos normales, el modelo aprende directamente el cuantil del 5 % de la distribución condicional del retorno de mañana, minimizando la *pinball loss*.

## Contenido

| Archivo | Descripción |
|---|---|
| `EDA_VaR_TailRisk_SPY.ipynb` | EDA, simulación de crisis (Monte Carlo), entrenamiento y backtesting |
| `var_predict.py` | Carga el modelo entrenado y predice el VaR para una fecha pasada por línea de comandos |
| `data/spy.csv` | Precios diarios ajustados del SPY (descargados con `yfinance`) |
| `models/var_gbm.pkl` | Modelo entrenado (`joblib`) |
| `requirements.txt` | Dependencias |

## Metodología

1. **EDA:** colas gruesas (curtosis, QQ-plot, Jarque-Bera), volatility clustering (ACF de |r|), drawdowns y crisis (2008, 2011, COVID-19, 2022).
2. **Monte Carlo:** probabilidad de una caída superior al 20 % en un mes con escenarios Normal, Student-t y bootstrap histórico, más un *stress test* con días del COVID-19.
3. **Modelos:** `QuantileRegressor` como línea base y `GradientBoostingRegressor(loss="quantile")`, ajustados con `GridSearchCV` + `TimeSeriesSplit`. Train 2005–2017, test 2018–2025.
4. **Backtesting:** comparación con el VaR histórico, el normal y EWMA RiskMetrics mediante los tests de Kupiec (cobertura) y Christoffersen (independencia de violaciones).

## Uso

```bash
pip install -r requirements.txt

# 1. Ejecutar el notebook (genera data/ y models/ si no existen)
jupyter notebook EDA_VaR_TailRisk_SPY.ipynb

# 2. Predecir el VaR desde la terminal
python var_predict.py 2020-03-16            # cartera de 100.000 € por defecto
python var_predict.py 2020-03-16 250000     # con otro capital
python var_predict.py                       # última sesión disponible
```

> Los gráficos son interactivos (Plotly) y no se muestran en la vista previa de GitHub. Para verlos, abre el notebook en Jupyter o VS Code.
