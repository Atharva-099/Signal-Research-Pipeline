# Signal Research Pipeline 📈

An ML-powered platform for cryptocurrency signal research, validation, and monitoring.

## Features

- **Signal Generation**: Momentum, Mean Reversion, Volatility, Funding Rate signals
- **Statistical Validation**: Walk-forward backtesting, bootstrap confidence intervals, deflated Sharpe ratio
- **ML Discovery**: XGBoost feature importance, SHAP interpretability
- **Regime Detection**: Hidden Markov Model for market regime identification
- **Ensemble Methods**: IC-weighted, stacking, regime-adaptive signal combination
- **Monitoring**: Decay tracking, health scoring, kill criteria

**Live URL**: https://signal-research-pipeline-a99.streamlit.app/

## Quick Start

### 1. Setup Environment

```bash
cd signal-research-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Example

```python
from src.data import PriceFetcher
from src.signals import MomentumSignal
from src.validation.backtester import Backtester

# Fetch data
fetcher = PriceFetcher()
price_df = fetcher.fetch(['BTC', 'ETH', 'SOL'], days=180)

# Generate signal
momentum = MomentumSignal(lookback=20)
signal_df = momentum.compute(price_df)

# Backtest
backtester = Backtester()
result = backtester.run(signal_df, price_df, signal_name='momentum_20d')
print(result)
```

### 3. Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
signal-research-pipeline/
├── config/config.yaml      # Configuration
├── src/
│   ├── data/               # Data fetchers (CoinGecko, Binance)
│   ├── signals/            # Signal generators
│   ├── validation/         # Backtesting & stats
│   ├── ml/                 # ML discovery & ensemble
│   └── monitoring/         # Health tracking
├── dashboard/app.py        # Streamlit dashboard
└── notebooks/              # Analysis notebooks
```

## Signals Available

| Signal | Description |
|--------|-------------|
| `momentum` | N-day price momentum |
| `momentum_multi` | Multi-timeframe momentum |
| `mean_reversion` | Z-score from moving average |
| `rsi` | Relative Strength Index |
| `bollinger` | Bollinger Band position |
| `volatility` | Realized volatility |
| `vol_regime` | Volatility regime indicator |
| `atr` | Average True Range |
| `funding` | Funding rate signal |
| `funding_momentum` | Funding rate momentum |

## Key Metrics

- **IC** (Information Coefficient): Correlation between signal and forward returns
- **IR** (Information Ratio): IC / std(IC) - risk-adjusted IC
- **Sharpe Ratio**: Annualized risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline

## Configuration

Edit `config/config.yaml` to customize:
- Asset universe
- Signal parameters
- Backtest settings
- Validation thresholds
- Monitoring criteria

## Data Sources

- **CoinGecko**: Free price data (50 calls/min)
- **Binance**: OHLCV and funding rates
- No API keys required for basic usage

