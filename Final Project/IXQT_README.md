# IXQT AI Trading Bot

**Version:** `v1.20.1-data-service-policy`
**Status:** Production trading system

IXQT is a full-stack AI trading platform that handles the entire pipeline from data ingestion through model training to live order execution. It runs as a Streamlit web app for interactive research, a terminal REPL for hands-on trading, and a set of headless daemons for automated overnight workflows — all backed by the same SQLite database, ML engines, and risk layer.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [UI Pages](#ui-pages)
- [CLI Workflows](#cli-workflows)
- [Cloud Training (GCP)](#cloud-training-gcp)
- [Testing](#testing)
- [Performance Tuning Notes](#performance-tuning-notes)
- [Development Notes](#development-notes)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Overview

IXQT combines:
- **Data layer** — Market data ingestion via yfinance with validation, anomaly filtering, and incremental updates. Macro indicators from FRED (CPI, 10Y Treasury, unemployment) and news sentiment via VADER. All stored in a local SQLite warehouse.
- **Feature engineering** — 15+ technical indicators (SMA, RSI, MACD, Bollinger, Ichimoku, ATR, ADX, BMSB, MCDX+), macro change features, lag features, multi-timeframe features, market session encoding, and normalized features for sector/universe models.
- **ML models** — Seven model families: XGBoost, Random Forest, LightGBM, SimpleNN, DeepNeuralNet, LSTM, GRU. Supports both classification (directional buy/sell) and regression (mean reversion price bands). Three labeling modes: naive (shift-1), cost-aware (min return threshold), and triple-barrier (TP/SL/timeout).
- **Evaluation** — Purged/embargoed train/test splits at all boundaries. Walk-forward validation with configurable folds. Risk-adjusted metrics: Sharpe, Sortino, MaxDD, Profit Factor, Calmar.
- **Backtesting** — Transaction cost modeling (EDGE bid-ask spread estimator + fixed spread + commission), slippage (next-bar open + optional stochastic noise), and kill switches (daily loss, drawdown, vol spike cooldown). ATR-based position sizing with risk presets.
- **Tournament** — Grid search over model, strategy, confidence, SL/TP, momentum, risk preset, and more. Results ranked by vs Buy&Hold, trimmed to top 5K per (Model, Res) with subgroup guarantees. Cached via signature-based dedup.
- **Execution** — Paper trading engine and live trading engine (Schwab API). Interactive terminal REPL with async event display, runtime config changes, and status footer. Kill switches, GTC order resubmission, and daily reconciliation.
- **Cloud training** — Nightly tournament runs on ephemeral GCP Spot GPU VMs (T4). Parallel ticker execution, heartbeat monitoring, automatic billing protection, and signature-based result merging.

## Key Features

### Models & Training
- Seven ML model families: `XGBoost`, `RandomForest`, `LightGBM`, `NeuralNet`, `DeepNeuralNet`, `LSTM`, `GRU`
- Directional (classification) and Mean Reversion (regression) strategy modes
- Three label types: Naive, Cost-Aware (default, 0.01% threshold), Triple Barrier
- Purged cross-validation (`PurgedTimeSeriesSplit`) replaces bare `TimeSeriesSplit` everywhere
- Walk-forward validation with configurable folds, embargo bars, and test period
- Hyperparameter optimization via `RandomizedSearchCV` for tree models

### Backtesting & Evaluation
- Transaction costs: EDGE bid-ask spread estimator (Ardia-Guidotti-Kroencke 2024), fixed spread, commission
- Slippage: next-bar open fill, optional stochastic noise within half-spread
- Kill switches: max daily loss, max drawdown halt, vol spike cooldown
- ATR-based position sizing with Standard/Large/None risk presets
- Auto-replay of execution params without retraining on Strategy Tester
- Risk-adjusted metrics: Sharpe, Sortino, MaxDD, Profit Factor, Calmar

### Tournament
- Grid search across models, strategies, confidence thresholds, SL/TP, momentum, risk presets, order types
- Signature-based caching — only new parameter combinations are evaluated
- Result trimming: top 5,000 per (Model, Res) with guaranteed min 100 per subgroup
- Periodic saves every 10 new results; full save on completion
- Best tournament config auto-loadable on Strategy Tester page

### Execution
- Paper and live trading engines with shared base class
- Interactive REPL via `prompt_toolkit` with async event display (fills, orders, warnings)
- Multi-level verbosity (0/1/2) with independent scanner detail toggle
- Runtime config changes via `set PARAM VALUE` commands
- Status footer: scan state, countdown, balance, trades today, market session
- Data service daemon eliminates SQLite lock contention between concurrent traders
- Kill switches wired to live/paper path via `RiskManager.check_kill_switches()`

### Cloud Training (GCP)
- Ephemeral Spot GPU VMs (`n1-highmem-4`, T4, pd-ssd) with automatic self-deletion
- Parallel ticker execution with configurable tree/neural parallelism
- Three billing safety guards: max runtime (4h), heartbeat staleness, progress stall
- Inner heartbeat telemetry: job-level + in-flight task progress
- Signature-based result merge/upsert + canonical re-trim (local state preserved while weaker rows are displaced by stronger rows)
- Operator helper (`ixqt-gcp.sh`) for status, watch, pull, merge workflows

### UI & Notifications
- Streamlit multi-page app with 9 pages across 3 navigation groups
- AgGrid + st.dataframe with conditional row/cell highlighting
- Per-channel Discord webhooks (paper_trades, live_trades, training, system)
- Persistent widget state across page navigation (`p_`/`w_` pattern)
- Centralized `HELP_TEXT` dict for consistent widget tooltips

## Architecture

```
Data Ingestion                    Feature Engineering              ML Training
───────────────                   ───────────────────              ───────────
yfinance OHLCV ─┐                pandas-ta indicators ─┐          XGBoost
FRED macro data ─┼→ DataManager → macro changes        ─┼→ MLEngine → RandomForest
VADER sentiment ─┘   (SQLite)     lag / MTF features   ─┘          LightGBM
                                  session encoding                 NeuralNet / DeepNN
                                                                   LSTM / GRU

Strategy & Risk                   Execution                       Cloud Training
───────────────                   ─────────                       ──────────────
StrategyEngine ─→ signals         PaperTradingEngine ─┐           submit_vm.sh
RiskManager ────→ position sizing LiveTradingEngine  ─┼→ console  run_training.sh
generate_trade_log() ─→ backtest  DataService daemon ─┘           nightly_model_training.py
```

### Data Flow

```
yfinance → DataManager (validate + SQLite) → get_data_with_macro()
→ StrategyEngine.add_indicators() → MLEngine.prepare_data() (feature engineering)
→ train_model() → saved model → run_backtest() / predict_next_day() / predict_consensus()
→ RiskManager.calculate_position_size() → paper_trader execution → notifier alerts
```

## Project Structure

```text
src/
  main.py                   # Streamlit app entrypoint
  pages/                    # 9 app pages (dashboard, data explorer, strategy tester,
                            #   AI models, tournament, forecast, trade ledger,
                            #   portfolio, performance tearsheet)
  ml_engine.py              # Training, backtesting, forecasting, model persistence
  strategy_engine.py        # Technical indicators, signal generation, trade simulation
  data_manager.py           # SQLite CRUD, yfinance fetching, data validation
  risk_manager.py           # ATR position sizing, stop/take-profit, kill switches
  tournament_core.py        # Signature generation, task evaluation, result trimming
  param_registry.py         # Frozensets for all parameter registries (drift detection)
  trade_math.py             # Pure FIFO trade matching (no Streamlit dependency)
  trading_engine.py         # Base trading engine with shared scan/execution logic
  trading_console.py        # Interactive REPL for paper/live traders (prompt_toolkit)
  data_service.py           # Standalone data service daemon (Unix socket IPC)
  data_client.py            # IPC client for the data service (auto-start, reconnect)
  macro_data.py             # FRED data + VADER sentiment
  notifier.py               # Discord webhook + email/SMS alerts
  notification_manager.py   # SQLite notifications + Discord forwarding
  reconciliation.py         # Schwab vs DB position/cash/order reconciliation
  nightly_model_training.py # Tournament CLI (parallel workers, timeouts, GCS upload)
  config.py                 # Constants, risk presets, .env loader
  ui_utils.py               # UI state, persistence, HELP_TEXT, grid rendering
  date_time_utils.py        # Market hours, holidays, timezone helpers
  version.py                # APP_VERSION + GIT_SHA (stamped at Docker build time)
tests/
  conftest.py               # Shared fixtures (sample_ohlcv_df, memory_db, sample_task/result)
  test_param_registry.py    # 41 regression tests for parameter registry consistency
  test_*.py                 # 425+ non-slow tests total, 8 slow ML training tests
gcp/
  Dockerfile.train          # CUDA 12.4 + miniconda training image (bakes DB into image)
  cloudbuild.yaml           # Cloud Build: build image + push to Artifact Registry
  run_training.sh           # Container entrypoint (parallel workers, heartbeat, GCS upload)
  submit_vm.sh              # Local orchestrator (build, launch, poll, merge, billing guards)
  ixqt-gcp.sh               # Operator helper (status, watch, log, pull, merge)
  tmux_gcp.sh               # GCP 2x2 tmux workspace (single window, 4 panes)
  count_rows.sh             # Compatibility wrapper for `ixqt-gcp rows`
launchers/                  # CLI launcher scripts for cron workflows
  tmux_ixqt.sh              # Local 2x2 tmux workspace (ixqt-st / ops / ixqt-paper / ixqt-live)
models/                     # Saved model artifacts (.joblib, .xgb.json, .pt) + metadata JSON
data/                       # SQLite DB (trading_bot.db) and ingested data
broker/                     # Schwab integration and auth files
```

## Prerequisites

- Python 3.12 (pandas-ta compatibility requirement)
- Conda recommended (`environment.yml`) or pip (`pyproject.toml`)
- Optional: Schwab API credentials for live/broker workflows
- Optional: GCP project with Artifact Registry + GCS bucket for cloud training

## Installation

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate IXQT312
pip install -e .
```

### Option 2: Pip only

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Create a `.env` file in the project root (loaded automatically by `config.py`):

```env
# Broker (optional — needed for live trading and reconciliation)
SCHWAB_APP_KEY=...
SCHWAB_APP_SECRET=...
SCHWAB_REDIRECT_URI=...
SCHWAB_ACCOUNT_HASH=...

# Notifications (optional — per-channel Discord webhooks)
DISCORD_WEBHOOK_URL=...                    # Fallback for all channels
DISCORD_WEBHOOK_PAPER_TRADES=...           # Paper trade fills
DISCORD_WEBHOOK_LIVE_TRADES=...            # Live trade fills
DISCORD_WEBHOOK_TRAINING=...               # Tournament training updates
DISCORD_WEBHOOK_SYSTEM=...                 # System alerts
```

Broker credentials can also be placed in `broker/schwab.env`.

Optional runtime tuning knobs (data-service fallback policy):

```env
# Allow direct-fetch bypass only after N hard data-service failures per ticker
IXQT_DATA_SERVICE_BYPASS_AFTER=3

# Keep bypass enabled for this many seconds once tripped
IXQT_DATA_SERVICE_BYPASS_COOLDOWN_SEC=120
```

## Usage

```bash
# Launch the Streamlit web app
streamlit run src/main.py
```

## UI Pages

The app has 9 pages across 3 navigation groups:

**Research & Analysis**
- **Dashboard** — Portfolio overview, market snapshot, recent trades
- **Data Explorer** — OHLCV charts with technical indicators, data quality checks
- **Strategy Backtester** — Train models, run backtests, compare against benchmarks (Golden Cross, RSI Reversal, MACD Cross). Backtest details popover shows full config and results. Trade log with trade count, execution realism controls, walk-forward validation.

**AI & Machine Learning**
- **AI Models (DNA)** — Train/evaluate individual models, view feature importance and metrics
- **AI Tournament** — Grid search across parameter space, live results while running, multi-sort/filter, best config export
- **AI Forecast** — Next-day predictions with model consensus, probability distributions

**Trade Execution**
- **Trade Ledger** — Full trade history with filtering
- **Holdings & Portfolio** — Current positions, unrealized P&L, position management
- **Performance Tearsheet** — Closed-trade analytics, per-ticker PnL breakdown, equity curve

## CLI Workflows

```bash
# Data ingestion — fetch/update market data for watchlist
python src/data_collector.py --help

# Nightly tournament training — grid search across models and parameters
python src/nightly_model_training.py --help
python src/nightly_model_training.py --tickers AAPL,NVDA --models XGBoost,LSTM

# Local maintenance: rebalance tournament JSONs (canonical compact+trim only)
./launchers/nightly_training.sh --rebalance
./launchers/nightly_training.sh --rebalance --tickers AAPL,NFLX

# Paper/live trading — interactive console with async event display
python src/paper_trader.py              # Interactive REPL (default)
python src/paper_trader.py --headless   # Headless for cron/automation
python src/paper_trader.py -v           # Verbose (decisions, forecasts)
python src/paper_trader.py -vv          # Debug (everything + third-party)
python src/live_trader.py               # Interactive REPL
python src/live_trader.py --headless    # Headless for cron/automation

# Data service daemon (auto-started by traders; manual start optional)
python src/data_service.py

# Daily reconciliation — Schwab vs DB positions, cash, orders
python src/reconciliation.py --help
```

### Console Commands

When running the interactive trading console (`paper_trader.py` or `live_trader.py`):

| Command | Description |
|---------|-------------|
| `status` | Engine state, scan progress, data service |
| `dashboard` | Account summary (cash, equity, holdings, P&L) |
| `config` | Current configuration and runtime params |
| `holdings` | Current positions with P&L |
| `pending` | Open/pending orders |
| `cancel [scope] [keep-queue]` | Cancel scope (`broker`, `console`, `all`) or abort; `cancel-all` is an alias for `cancel all` |
| `tail [N\|TICKER]` | Recent log entries, optionally filtered |
| `scan` | Trigger immediate scan (skip sleep timer) |
| `set PARAM VALUE` | Change runtime config (e.g., `set verbose 1`, `set scanner simple`) |
| `pause` / `resume` | Pause/resume scanning |
| `kill-switches` | Show kill switch status |
| `data-service [status\|restart\|stop]` | Manage data service daemon |
| `purge-notifications [DAYS]` | Clean old notification records |

## Local tmux Workspace

```bash
# 2x2 local workspace (auto-attach)
# top-left: ixqt-st, top-right: ops, bottom-left: ixqt-paper, bottom-right: ixqt-live
./launchers/tmux_ixqt.sh

# Start fresh
./launchers/tmux_ixqt.sh --reset
```

## Cloud Training (GCP)

Tournament training runs on an ephemeral Spot GPU VM to offload the local machine.

```bash
# Build image + launch VM + poll + merge results
./gcp/submit_vm.sh

# Skip Docker build (reuse existing image)
./gcp/submit_vm.sh --skip-build

# Operator helper
./gcp/ixqt-gcp.sh status    # Combined status (heartbeat + config + manifest + inner progress)
./gcp/ixqt-gcp.sh watch     # Live heartbeat watch
./gcp/ixqt-gcp.sh pull      # Pull results from GCS
./gcp/ixqt-gcp.sh merge     # Signature-based upsert + canonical re-trim into local tournament JSONs
./gcp/ixqt-gcp.sh preempt   # Spot preemption events from Cloud Logging

# tmux 2x2 workspace (single window, 4 panes; auto-detect RUN_ID, auto-attach)
./gcp/tmux_gcp.sh

# Start fresh (replace existing tmux session)
./gcp/tmux_gcp.sh --reset

# Row-count snapshot helper (with retries for mid-write files)
./gcp/ixqt-gcp.sh rows
```

**Infrastructure:** `n1-highmem-4` (4 vCPU, 26GB RAM), 1x T4 GPU, Spot, pd-ssd boot disk. CUDA 12.4 + miniconda image with DB baked in.

**Safety:** Three auto-kill guards (max runtime 4h, heartbeat staleness 10min x3, progress stall 12min). Per-task training timeout (20 min default). Inner heartbeat telemetry for in-flight task progress.
Runs auto-resume from the latest prior run's tournament cache unless `--no-resume` is specified.

**Resume semantics (important):**
- Resume is **signature-cache resume**, not mid-fit checkpoint resume.
- `RESUME_RUN_ID` seeds local `models/*_tournament_v2.json` from prior GCS run results.
- During a run, tournament JSON snapshots are uploaded periodically and merged locally during polling.
- Merge paths re-apply tournament retention policy (top 5k per Model+Res + subgroup guarantees), so stronger rows replace weaker rows when capped.
- If Spot preempts during an active fit, that in-flight fit restarts on next run (no epoch/tree-level continuation).

See [GCP.md](GCP.md) for full monitoring, SSH commands, and architecture details.

## Testing

```bash
# Run all non-slow tests (389+ tests, ~13s)
conda run -n IXQT312 python -m pytest tests/

# Run slow ML training tests (8 tests)
conda run -n IXQT312 python -m pytest tests/ -m slow

# Run specific test file
conda run -n IXQT312 python -m pytest tests/test_param_registry.py -v

# Quick compile check
python -m compileall src
```

Test config is in `pyproject.toml`. Shared fixtures are in `tests/conftest.py`.

## Performance Tuning Notes

Runtime/performance roadmap and profiling notes are tracked in `docs/live/PERF_TUNING_NOTES.md`.

## Development Notes

- Version metadata: `src/version.py` (UI display), `pyproject.toml` (package)
- Widget persistence: `p_`/`w_` dual-key pattern in `ui_utils.py` — see `init_session_state()`
- Parameter registries: `param_registry.py` defines frozensets for all input dicts, signature fields, and popover keys. Regression tests in `test_param_registry.py` catch drift.
- Shared logic locations:
  - Action reversal: `trading_engine.py` (`_is_reversal_action`)
  - Market holidays/sessions: `date_time_utils.py` (`is_us_market_holiday`)
  - Trade math: `trade_math.py` (FIFO matching, no Streamlit dependency)
  - Tournament signatures: `tournament_core.py` (`sig_from_task`, `sig_from_dict`)
- GCP heartbeat exposes both job-level progress and inner active-task progress for long-running jobs
- Tournament JSON periodic save cadence is every 10 new results (separate from progress log cadence at 1% increments)

## Changelog

Version history is in [`CHANGELOG.md`](CHANGELOG.md).

## Contributing

Contributions are welcome. A lightweight workflow:

1. Create a feature branch.
2. Make focused changes with clear commit messages.
3. Run tests and compile check:
```bash
conda run -n IXQT312 python -m pytest tests/
python -m compileall src
```
4. Open a PR with a short summary, impacted pages/scripts, and validation steps.

## License

This project is licensed under the IXQT Proprietary License. See [`LICENSE`](LICENSE).

## Disclaimer

This software is for research and educational use. It is not financial advice. Trading involves risk, including possible loss of principal.
