import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path
file_name = "All-Weather ETF(index) portfolio final backtesting statistics with 军工080925.xlsx"
file_path = os.path.join(script_dir, file_name)

# Check if file exists
if not os.path.exists(file_path):
    print(f"File not found at: {file_path}")
    print("Please check the file name and location.")
    parent_dir = os.path.dirname(script_dir)
    alternative_path = os.path.join(parent_dir, file_name)
    if os.path.exists(alternative_path):
        file_path = alternative_path
        print(f"Found file at: {file_path}")
    else:
        print("Could not find the file automatically.")
        file_path = input("Please enter the full path to the Excel file: ")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

# Read the price data
price_data = pd.read_excel(
    file_path,
    sheet_name='etf price automation (2)',
    header=1,
    index_col=0,
    usecols="A:U",  # 21 assets
    skiprows=0
)

# Clean the data
price_data = price_data[price_data.index.notnull()]
price_data.index = pd.to_datetime(price_data.index, errors='coerce')
price_data = price_data.apply(pd.to_numeric, errors='coerce')

# Generate end dates for the 60 periods
end_dates = pd.date_range(start='2020-08-01', end='2025-07-31', freq="M")


# Function to calculate risk contributions
def calculate_risk_contributions(weights, cov_matrix):
    """
    Calculate the percentage risk contribution of each asset
    """
    portfolio_variance = weights.T @ cov_matrix @ weights
    marginal_risk_contribution = cov_matrix @ weights
    risk_contribution = weights * marginal_risk_contribution
    percentage_risk_contribution = risk_contribution / portfolio_variance
    return percentage_risk_contribution


# Function to fit ARIMA-GARCH model and forecast volatility
def forecast_volatility(returns_series, forecast_horizon=21):
    """
    Fit ARIMA-GARCH model and forecast volatility
    """
    try:
        # Fit ARIMA model
        arima_order = (1, 0, 1)
        arima_model = ARIMA(returns_series, order=arima_order)
        arima_result = arima_model.fit()

        # Get residuals from ARIMA
        residuals = arima_result.resid.dropna()

        # Fit GARCH model on residuals
        garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
        garch_result = garch_model.fit(disp='off')

        # Forecast volatility
        forecast = garch_result.forecast(horizon=forecast_horizon)
        forecast_vol = np.sqrt(forecast.variance.values[-1, :].mean())

        return forecast_vol

    except Exception as e:
        print(f"ARIMA-GARCH model failed: {e}")
        # Fallback to historical volatility if model fails
        return returns_series.std()


# Function to calculate forecasted covariance matrix
def forecast_covariance_matrix(returns, forecast_horizon=21):
    """
    Calculate forecasted covariance matrix using ARIMA-GARCH models
    """
    n_assets = returns.shape[1]

    # Forecast individual volatilities
    vol_forecasts = []
    for asset in returns.columns:
        vol_forecast = forecast_volatility(returns[asset], forecast_horizon)
        vol_forecasts.append(vol_forecast)

    # Use historical correlation structure with forecasted volatilities
    corr_matrix = returns.corr().values
    vol_matrix = np.diag(vol_forecasts)
    forecast_cov = vol_matrix @ corr_matrix @ vol_matrix

    return forecast_cov, vol_forecasts


# Function to calculate risk parity weights with forecasted covariance matrix
def calculate_risk_parity_weights(returns):
    """
    Calculate risk parity weights using forecasted covariance matrix from ARIMA-GARCH
    """
    n_assets = returns.shape[1]

    # Use forecasted covariance matrix from ARIMA-GARCH
    cov_matrix, vol_forecasts = forecast_covariance_matrix(returns)
    print("Using forecasted covariance matrix from ARIMA-GARCH models")

    # Standard risk parity optimization
    def risk_parity_objective(weights):
        portfolio_variance = weights.T @ cov_matrix @ weights
        marginal_risk_contribution = cov_matrix @ weights
        risk_contribution = weights * marginal_risk_contribution / portfolio_variance

        # Objective: equal risk contribution
        target_risk_contribution = portfolio_variance / n_assets
        return np.sum((risk_contribution - target_risk_contribution) ** 2)

    # Constraints: weights sum to 1, no short selling
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess (equal weights)
    x0 = np.ones(n_assets) / n_assets

    # Optimize
    result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        print(f"Optimization warning: {result.message}")

    # Calculate risk contributions to verify risk parity
    risk_contributions = calculate_risk_contributions(result.x, cov_matrix)
    target_risk_contribution = 1 / n_assets

    print(f"Target risk contribution per asset: {target_risk_contribution:.3f}")
    print("Actual risk contributions:")
    for i, asset in enumerate(returns.columns):
        print(f"  {asset}: {risk_contributions[i]:.3f}")

    # Calculate deviation from perfect risk parity
    risk_parity_deviation = np.std(risk_contributions - target_risk_contribution)
    max_deviation = np.max(np.abs(risk_contributions - target_risk_contribution))
    print(f"Risk parity deviation (std): {risk_parity_deviation:.6f}")
    print(f"Maximum deviation: {max_deviation:.6f}")

    return result.x, risk_contributions, risk_parity_deviation, max_deviation, vol_forecasts


# Prepare results storage
all_results = []
portfolio_weekly_returns = pd.Series(dtype=float)
portfolio_weekly_values = pd.Series(dtype=float)
risk_contribution_details = []
forecast_metrics_details = []
initial_portfolio_value = 10000
current_portfolio_value = initial_portfolio_value

# Process each period
for i, end_date in enumerate(end_dates):
    print(f"\nProcessing period {i + 1}/{len(end_dates)} ending {end_date.strftime('%Y-%m-%d')}")

    # Filter data up to the current end date
    period_data = price_data[price_data.index <= end_date]

    # Calculate returns
    returns = period_data.pct_change().dropna()

    # Skip if not enough data
    if len(returns) < 30:
        print(f"Not enough data for period ending {end_date}, skipping...")
        continue

    # Calculate risk parity weights for all assets using forecasted covariance
    print("Calculating risk parity for all assets with ARIMA-GARCH forecast...")
    try:
        weights, risk_contributions, risk_parity_deviation, max_deviation, vol_forecasts = calculate_risk_parity_weights(
            returns)
    except Exception as e:
        print(f"Error calculating risk parity: {e}")
        continue

    # Store forecasting metrics
    forecast_metrics = {
        'End_Date': end_date.strftime('%Y-%m-%d'),
        'Risk_Parity_Deviation': risk_parity_deviation,
        'Max_Deviation': max_deviation
    }

    # Add individual asset volatility forecasts
    for j, asset in enumerate(returns.columns):
        forecast_metrics[f'{asset}_forecast_vol'] = vol_forecasts[j]

    forecast_metrics_details.append(forecast_metrics)

    # Store risk contribution details
    risk_dict = {'End_Date': end_date.strftime('%Y-%m-%d')}
    for j, asset in enumerate(returns.columns):
        risk_dict[f'Risk_Contribution_{asset}'] = risk_contributions[j]
    risk_contribution_details.append(risk_dict)

    # Calculate portfolio performance metrics
    portfolio_returns = returns @ weights

    # RESAMPLE TO WEEKLY FREQUENCY
    weekly_returns = portfolio_returns.resample('W').apply(lambda x: (1 + x).prod() - 1)

    # Calculate maximum drawdown using weekly returns
    cumulative_returns = (1 + weekly_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

    # Calculate Sharpe ratio using weekly returns
    sharpe_ratio = weekly_returns.mean() / weekly_returns.std() * np.sqrt(52) if weekly_returns.std() > 0 else 0

    # Calculate Sortino ratio (downside deviation)
    downside_returns = weekly_returns[weekly_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(52) if len(downside_returns) > 0 else 0
    sortino_ratio = weekly_returns.mean() / downside_deviation * np.sqrt(52) if downside_deviation > 0 else 0

    # Calculate Value at Risk (VaR) at 95% confidence
    var_95 = np.percentile(weekly_returns, 5) if len(weekly_returns) > 0 else 0

    # Calculate Conditional Value at Risk (CVaR) at 95% confidence
    cvar_95 = weekly_returns[weekly_returns <= var_95].mean() if len(weekly_returns) > 0 else 0

    # Store results
    result_dict = {
        'End_Date': end_date.strftime('%Y-%m-%d'),
        'Portfolio_Mean_Return': weekly_returns.mean() if len(weekly_returns) > 0 else 0,
        'Portfolio_Variance': weekly_returns.var() if len(weekly_returns) > 0 else 0,
        'Weekly_Return': weekly_returns.iloc[-1] if len(weekly_returns) > 0 else 0,
        'Weekly_Volatility': weekly_returns.std() * np.sqrt(52) if len(weekly_returns) > 0 else 0,
        'Max_Drawdown': max_drawdown,
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'VaR_95': var_95,
        'CVaR_95': cvar_95,
        'Risk_Parity_Deviation': risk_parity_deviation,
        'Max_Deviation': max_deviation
    }

    # Add individual asset weights
    for j, asset in enumerate(returns.columns):
        result_dict[f'Weight_{asset}'] = weights[j]

    all_results.append(result_dict)

    # Simulate weekly portfolio values for this period
    if i == 0:
        period_weekly_returns = weekly_returns
    else:
        prev_end_date = end_dates[i - 1]
        period_mask = (weekly_returns.index > prev_end_date) & (weekly_returns.index <= end_date)
        period_weekly_returns = weekly_returns[period_mask]

    # Calculate weekly portfolio values
    if len(period_weekly_returns) > 0:
        period_weekly_values = (1 + period_weekly_returns).cumprod() * current_portfolio_value
        current_portfolio_value = period_weekly_values.iloc[-1]

        # Add to the overall weekly returns and values series
        portfolio_weekly_returns = pd.concat([portfolio_weekly_returns, period_weekly_returns])
        portfolio_weekly_values = pd.concat([portfolio_weekly_values, period_weekly_values])

# Create DataFrames from results
results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
forecast_metrics_df = pd.DataFrame(forecast_metrics_details) if forecast_metrics_details else pd.DataFrame()
risk_contribution_df = pd.DataFrame(risk_contribution_details) if risk_contribution_details else pd.DataFrame()

# Calculate overall performance metrics using weekly returns
if len(portfolio_weekly_returns) > 0:
    overall_cumulative_return = (portfolio_weekly_values.iloc[-1] / initial_portfolio_value - 1)
    annualized_return = (1 + overall_cumulative_return) ** (52 / len(portfolio_weekly_returns)) - 1
    annualized_volatility = portfolio_weekly_returns.std() * np.sqrt(52)
    overall_sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

    # Calculate overall maximum drawdown
    overall_cumulative_returns = (1 + portfolio_weekly_returns).cumprod()
    overall_running_max = overall_cumulative_returns.expanding().max()
    overall_drawdown = (overall_cumulative_returns - overall_running_max) / overall_running_max
    overall_max_drawdown = overall_drawdown.min() if len(overall_drawdown) > 0 else 0
else:
    overall_cumulative_return = 0
    annualized_return = 0
    annualized_volatility = 0
    overall_sharpe_ratio = 0
    overall_max_drawdown = 0

# Add overall metrics to results
overall_metrics = pd.DataFrame({
    'Metric': ['Initial_Value', 'Final_Value', 'Cumulative_Return', 'Annualized_Return',
               'Annualized_Volatility', 'Sharpe_Ratio', 'Max_Drawdown'],
    'Value': [initial_portfolio_value,
              portfolio_weekly_values.iloc[-1] if len(portfolio_weekly_values) > 0 else 0,
              overall_cumulative_return, annualized_return, annualized_volatility,
              overall_sharpe_ratio, overall_max_drawdown]
})

# Save to CSV files
try:
    if not results_df.empty:
        results_df.to_csv(os.path.join(script_dir, 'single_layer_arima_garch_results.csv'), index=False)
        print("Saved: single_layer_arima_garch_results.csv")

    if not forecast_metrics_df.empty:
        forecast_metrics_df.to_csv(os.path.join(script_dir, 'single_layer_forecast_metrics.csv'), index=False)
        print("Saved: single_layer_forecast_metrics.csv")

    if not risk_contribution_df.empty:
        risk_contribution_df.to_csv(os.path.join(script_dir, 'single_layer_risk_contributions.csv'), index=False)
        print("Saved: single_layer_risk_contributions.csv")

    if len(portfolio_weekly_returns) > 0:
        portfolio_weekly_returns.to_csv(os.path.join(script_dir, 'single_layer_weekly_returns.csv'),
                                        header=['Weekly_Return'])
        portfolio_weekly_values.to_csv(os.path.join(script_dir, 'single_layer_weekly_values.csv'),
                                       header=['Portfolio_Value'])
        print("Saved: single_layer_weekly_returns.csv")
        print("Saved: single_layer_weekly_values.csv")

    overall_metrics.to_csv(os.path.join(script_dir, 'single_layer_overall_metrics.csv'), index=False)
    print("Saved: single_layer_overall_metrics.csv")

except Exception as e:
    print(f"Error saving files: {e}")

print("\nAll results saved to CSV files:")

# Plot portfolio value over time using weekly data
if len(portfolio_weekly_values) > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_weekly_values.index, portfolio_weekly_values.values)
    plt.title('Single Layer ARIMA-GARCH Risk Parity Portfolio Value (Weekly)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, 'single_layer_portfolio_value.png'))
    plt.show()
    print("Saved: single_layer_portfolio_value.png")

# Plot portfolio weights for the last period
if len(all_results) > 0:
    last_weights = {k: v for k, v in all_results[-1].items() if k.startswith('Weight_')}
    assets = [k.replace('Weight_', '') for k in last_weights.keys()]
    weights = list(last_weights.values())

    plt.figure(figsize=(12, 6))
    plt.bar(assets, weights)
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.title('Asset Weights - Single Layer Approach (Last Period)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'single_layer_asset_weights.png'))
    plt.show()
    print("Saved: single_layer_asset_weights.png")

# Print summary
print("\nSummary of portfolio performance:")
print(f"Initial portfolio value: ${initial_portfolio_value:,.2f}")
print(f"Final portfolio value: ${portfolio_weekly_values.iloc[-1] if len(portfolio_weekly_values) > 0 else 0:,.2f}")
print(f"Cumulative return: {overall_cumulative_return:.2%}")
print(f"Annualized return: {annualized_return:.2%}")
print(f"Annualized volatility: {annualized_volatility:.2%}")
print(f"Sharpe ratio: {overall_sharpe_ratio:.2f}")
print(f"Maximum drawdown: {overall_max_drawdown:.2%}")

# Print risk parity summary
if forecast_metrics_details:
    deviations = [d.get('Risk_Parity_Deviation', 0) for d in forecast_metrics_details if 'Risk_Parity_Deviation' in d]
    max_deviations = [d.get('Max_Deviation', 0) for d in forecast_metrics_details if 'Max_Deviation' in d]

    if deviations:
        print(f"\nRisk Parity Summary:")
        print(f"Average risk parity deviation: {np.mean(deviations):.6f}")
        print(f"Average maximum deviation: {np.mean(max_deviations):.6f}")