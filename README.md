# Single Layer ARIMA-GARCH Risk Parity Portfolio Optimization

## Overview

This project implements an advanced risk parity portfolio optimization strategy that combines traditional risk parity principles with sophisticated time series forecasting using ARIMA-GARCH models. The model forecasts future volatility patterns to create more robust portfolio allocations.

## Key Innovations

### ARIMA-GARCH Volatility Forecasting
- **ARIMA Models**: Capture the autoregressive and moving average components of return series
- **GARCH Models**: Model time-varying volatility and volatility clustering effects
- **21-Day Forecast Horizon**: Projects volatility 21 trading days ahead for forward-looking allocations

### Single Layer Optimization Approach
Unlike traditional hierarchical methods, this implementation uses a single optimization layer across all 21 ETF assets, providing a more integrated and theoretically sound allocation framework.

## Methodology

### Volatility Forecasting Pipeline
1. **ARIMA(1,0,1) Model**: Fits autoregressive integrated moving average model to capture return dynamics
2. **GARCH(1,1) Model**: Models conditional heteroskedasticity in residual volatility
3. **Correlation Preservation**: Maintains historical correlation structure while updating volatilities
4. **Forward-Looking Covariance**: Combines forecasted volatilities with historical correlations

### Risk Parity Optimization
- **Objective**: Equal risk contribution across all assets
- **Constraints**: Full investment (weights sum to 1), no short selling
- **Optimization Method**: Sequential Least Squares Programming (SLSQP)

## Requirements

### Python Libraries
```python
pandas, numpy, scipy
matplotlib (for visualization)
scikit-learn (StandardScaler)
statsmodels (ARIMA models)
arch (GARCH models)
```

### Data File
- **File**: `All-Weather ETF(index) portfolio final backtesting statistics with 军工080925.xlsx`
- **Sheet**: `etf price automation (2)`
- **Assets**: 21 ETFs (columns A-U)
- **Period**: August 2020 to July 2025 (60 monthly periods)

## Code Structure

### Core Functions

#### `forecast_volatility(returns_series, forecast_horizon=21)`
- Fits ARIMA-GARCH model to return series
- Returns 21-day volatility forecast
- Falls back to historical volatility if model fails

#### `forecast_covariance_matrix(returns, forecast_horizon=21)`
- Forecasts individual asset volatilities using ARIMA-GARCH
- Preserves historical correlation matrix
- Combines forecasted volatilities with correlations

#### `calculate_risk_parity_weights(returns)`
- Main optimization function using forecasted covariance matrix
- Implements risk parity objective function
- Returns weights and risk contribution analysis

### Analysis Features

- **60 Monthly Periods**: Rolling window analysis from Aug 2020 to Jul 2025
- **Weekly Performance**: Resampled to weekly frequency for risk metrics
- **Comprehensive Risk Metrics**: Sharpe ratio, Sortino ratio, VaR, CVaR, maximum drawdown
- **Risk Parity Diagnostics**: Measures deviation from perfect risk parity

## Output Files

### Main Results
- `single_layer_arima_garch_results.csv`: Portfolio weights and performance metrics for each period
- `single_layer_weekly_returns.csv`: Weekly portfolio returns
- `single_layer_weekly_values.csv`: Weekly portfolio values (starting at $10,000)
- `single_layer_overall_metrics.csv`: Summary performance statistics

### Diagnostic Files
- `single_layer_forecast_metrics.csv`: ARIMA-GARCH forecasting performance and risk parity deviations
- `single_layer_risk_contributions.csv`: Detailed risk contribution analysis for each asset
- `single_layer_portfolio_value.png`: Portfolio value chart over time
- `single_layer_asset_weights.png`: Asset allocation chart for final period

## Performance Metrics

### Return Metrics
- **Cumulative Return**: Total return over entire period
- **Annualized Return**: Geometric average annual return
- **Weekly Returns**: Time series of weekly performance

### Risk Metrics
- **Annualized Volatility**: Standard deviation of weekly returns annualized
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Sortino Ratio**: Downside risk-adjusted return
- **VaR (95%)**: Value at Risk at 95% confidence
- **CVaR (95%)**: Conditional Value at Risk (expected shortfall)

### Risk Parity Metrics
- **Risk Parity Deviation**: Standard deviation of risk contributions from equal target
- **Maximum Deviation**: Largest individual asset deviation from target risk contribution

## Usage

1. **Data Preparation**: Ensure Excel file is in the correct directory
2. **Run Optimization**:
   ```bash
   python single_layer_arima_garch_risk_parity.py
   ```
3. **Review Results**: Check generated CSV files and charts
4. **Analysis**: Use output files for performance analysis and risk monitoring

## Key Features

### Advanced Volatility Modeling
- Dynamic volatility forecasts rather than relying on historical volatility
- Captures volatility persistence and clustering effects
- More responsive to changing market conditions

### Robust Risk Management
- Comprehensive risk metric calculation
- Weekly frequency for more granular risk assessment
- Multiple downside risk measures (VaR, CVaR, drawdown)

### Diagnostic Capabilities
- Detailed risk contribution analysis
- Optimization convergence monitoring
- Forecasting model performance tracking

## Customization Options

### Model Parameters
```python
# Change forecast horizon
forecast_horizon = 21  # Trading days

# Modify ARIMA order
arima_order = (1, 0, 1)  # (p, d, q)

# Adjust GARCH parameters
garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
```

### Analysis Period
```python
end_dates = pd.date_range(start='2020-08-01', end='2025-07-31', freq='M')
```

### Portfolio Settings
```python
initial_portfolio_value = 10000  # Starting capital
```

## Interpretation Guide

### Optimal Results
- **Low Risk Parity Deviation**: < 0.05 indicates good risk parity achievement
- **Stable Weights**: Consistent allocations across periods
- **Positive Sharpe Ratio**: Positive risk-adjusted returns
- **Controlled Drawdown**: Maximum drawdown < 20%

### Warning Signs
- **High Optimization Errors**: Check convergence messages
- **Extreme Weight Concentrations**: > 30% in single asset
- **Persistent High Deviations**: Risk parity not being achieved
- **Model Failures**: Frequent fallbacks to historical volatility

## Advantages Over Traditional Methods

1. **Forward-Looking**: Uses forecasted rather than historical volatility
2. **Integrated Approach**: Single optimization layer avoids hierarchical biases
3. **Time Series Aware**: Accounts for volatility clustering and persistence
4. **Comprehensive Diagnostics**: Multiple metrics for model validation

## Limitations and Considerations

- **Computational Intensity**: ARIMA-GARCH fitting requires significant computation
- **Model Risk**: Forecasting models may not always outperform historical estimates
- **Stationarity Assumption**: Assumes return characteristics are relatively stable
- **Regular Re-estimation**: Models should be re-estimated periodically for live use

This implementation represents a sophisticated approach to risk parity that combines modern time series forecasting with traditional portfolio optimization principles.
