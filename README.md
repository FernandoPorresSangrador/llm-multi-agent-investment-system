# Multi-Agent LLM Investment System

Bachelor's thesis (TFG) in Computer Engineering, Universidad de Cantabria.
Author: Fernando Porres Sangrador · Supervisor: Carlos Blanco Bueno

A research prototype that generates BUY / HOLD / SELL decisions for 50 S&P 500 stocks across 5 sectors by combining specialised agents: technical analysis, fundamental analysis, risk, news sentiment (FinBERT) and Monte Carlo simulation. Agent outputs are structured with Pydantic and combined by a local LLM (Mistral 7B via Ollama and LangChain).

## Systems compared

| System | Description |
|---|---|
| LLM pure | Five agents, each interpreted by Mistral 7B (6 LLM calls per stock) |
| Hybrid | Same agents, Monte Carlo thresholds applied in Python (1 LLM call per stock) |
| Random Forest | Monolithic ML baseline (GridSearchCV + TimeSeriesSplit) |
| Buy & Hold SPY | Benchmark |

Each strategy is evaluated under three risk profiles (conservative, moderate, aggressive).

## Hybrid system

The hybrid system keeps the same five-agent structure but gives each agent the tool best suited to its task. Only the final decision uses the LLM, which is the one step that has to reason over several sources of evidence and the investor's risk profile.

```
Technical agent    ->  Random Forest (technical indicators)      ─┐
Fundamental agent  ->  Random Forest (fundamental ratios)         │
Risk agent         ->  Random Forest (risk metrics)               ├─>  Orchestrator (Mistral 7B)  ->  BUY / HOLD / SELL
Sentiment agent    ->  FinBERT on financial news                  │    + risk profile
Monte Carlo agent  ->  simulated price paths, Python thresholds  ─┘
```

**Specialised agents**
- The technical, fundamental and risk agents are Random Forest classifiers, each trained only on its own feature set. Hyperparameters are tuned with GridSearchCV over a 5-fold TimeSeriesSplit, so validation always uses data later than training.
- The sentiment agent scores news headlines with FinBERT, a BERT model fine-tuned on financial text. When no news is available (FNSPID ends on 9 Jan 2024), it falls back to a price-momentum proxy.
- The Monte Carlo agent simulates future price paths and turns the probability of gain or loss into a signal using fixed thresholds in Python. In the LLM pure system, this interpretation is done by the LLM instead.

**Structured signals**
Every agent returns a Pydantic object with a signal (BUY / HOLD / SELL), a confidence score and a short justification. This gives the orchestrator a fixed, validated input format.

**Orchestrator**
Mistral 7B, running locally through Ollama and LangChain, receives the five signals and the risk profile and returns the final decision. The output schema is split in two (`_DecisionLLM` for what the model generates, `DecisionFinal` for the validated result with the ticker added in code), so the model cannot hallucinate or alter the ticker. Output length is capped (`num_predict=120`) to keep inference time low.

**Why hybrid**
Numerical tasks go to models built for them (tree ensembles, a finance-specific NLP model, simulation), and the LLM is used only where reasoning adds value. This cuts LLM calls from 6 to 1 per stock and makes each agent's output reproducible and easier to audit.

## Results (monthly backtest, 2019 to 2024, $100k initial capital)

| Strategy | Cumulative return | CAGR | Sharpe | Max drawdown |
|---|---|---|---|---|
| Hybrid, aggressive | +334.23% | +27.79% | 1.256 | -31.18% |
| Buy & Hold SPY | +158.40% | | | |

- Hybrid aggressive was the only configuration with a positive Information Ratio (+0.690) and had the highest alpha (+10.73%).
- The LLM pure system had the best risk-adjusted performance (Sharpe above 1.4, volatility under 9% in all profiles).
- The Random Forest baseline did not beat the benchmark in any profile.

## Methodology notes

- Look-ahead bias avoided: TimeSeriesSplit (5 splits) and point-in-time data only.
- HOLD liquidates positions, so results are not inflated by hidden buy-and-hold exposure.
- Sentiment data from the FNSPID dataset (coverage ends 9 Jan 2024). For later dates and uncovered tickers, 5-day and 20-day returns are used as a sentiment proxy.
- Risk-free rate of 0% for Sharpe and Sortino, 5% for alpha.

## Limitations

- The LLM pure backtest uses a simplified batch run (groups of 5 tickers, technical indicators only) rather than the full five-agent orchestrator, to keep local inference time manageable.
- Research prototype: no transaction costs model beyond the one described in the thesis, and not intended for live trading.

## Repository structure

```
src/         Python modules (main system: sistema_llm_puro.py)
notebooks/   Jupyter notebooks for data preparation, training and backtesting
results/     Backtest outputs, metrics and figures
docs/        Full thesis (PDF, Spanish)
```

Files and notebooks use an internal v1 / v2 / v3 naming convention, explained in Appendix A of the thesis.

## Reproducing

```bash
git clone https://github.com/FernandoPorresSangrador/llm-multi-agent-investment-system.git
cd llm-multi-agent-investment-system
pip install -r requirements.txt

# local LLM
ollama pull mistral
```

Then run the notebooks in `notebooks/` in numerical order.

## Tech stack

Python · LangChain · Ollama (Mistral 7B) · Pydantic · FinBERT (Hugging Face Transformers) · scikit-learn · pandas · NumPy
