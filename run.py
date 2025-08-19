import datetime
import pandas as pd
import yfinance as yf
from fredapi import Fred
import os
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# Configuration & API Setup
# ----------------------------

# Load FRED API key from environment variable
FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in environment variables. Please check your .env file.")
fred = Fred(api_key=FRED_API_KEY)

# Define the date range (5 years of historical data for better pattern recognition)
end_date = datetime.datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.datetime.today() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")

# Create output directory for data
os.makedirs("data", exist_ok=True)

# Time periods for aggregation
DAILY_LOOKBACK = 1  # Previous day
WEEKLY_LOOKBACK = 7  # 1 week
MONTHLY_LOOKBACK = 30  # ~1 month
QUARTERLY_LOOKBACK = 90  # ~1 quarter
YEARLY_LOOKBACK = 365  # 1 year

# ----------------------------
# Function to create aggregated data for LLM
# ----------------------------

def create_market_summary(data, name, ticker, lookback_periods):
    """Create a summary of market data suitable for LLM consumption"""
    if data.empty:
        return None
    
    # Current values
    latest_date = data.index[-1].strftime('%Y-%m-%d')
    current_price = data['Close'].iloc[-1]
    
    # Calculate returns for different periods
    returns = {}
    volatility = {}
    
    for period_name, days in lookback_periods.items():
        if len(data) > days:
            # Calculate return
            prior_price = data['Close'].iloc[-(days+1)] if len(data) > days + 1 else data['Close'].iloc[0]
            period_return = (current_price / prior_price - 1) * 100
            returns[period_name] = round(period_return, 2)
            
            # Calculate volatility (stddev of returns)
            if days > 5:  # Only calculate for periods longer than a week
                period_data = data.iloc[-days:]
                daily_returns = period_data['Close'].pct_change().dropna()
                period_vol = daily_returns.std() * 100
                volatility[period_name] = round(period_vol, 2)
    
    # Technical indicators (current values)
    tech_indicators = {}
    
    # Moving Averages
    if len(data) >= 50:
        ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
        tech_indicators['MA_50'] = round(ma50, 2)
        tech_indicators['Price_to_MA50'] = round(current_price / ma50, 2)
    
    if len(data) >= 200:
        ma200 = data['Close'].rolling(window=200).mean().iloc[-1]
        tech_indicators['MA_200'] = round(ma200, 2)
        tech_indicators['Price_to_MA200'] = round(current_price / ma200, 2)
    
    # RSI (simplified calculation)
    if len(data) >= 14:
        try:
            # Calculate RSI directly using numpy
            delta = data['Close'].diff().dropna().values
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                tech_indicators['RSI_14'] = round(rsi, 2)
        except Exception as e:
            print(f"Error calculating RSI for {ticker}: {e}")
    
    # MACD
    if len(data) >= 26:
        exp1 = data['Close'].ewm(span=12, adjust=False).mean().iloc[-1]
        exp2 = data['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
        macd = exp1 - exp2
        tech_indicators['MACD'] = round(macd, 2)
    
    # Create summary object
    summary = {
        'name': name,
        'ticker': ticker,
        'date': latest_date,
        'current_price': {ticker: round(current_price, 2)},
        'returns': {
            'daily': {ticker: returns.get('daily')},
            'weekly': {ticker: returns.get('weekly')},
            'monthly': {ticker: returns.get('monthly')},
            'quarterly': {ticker: returns.get('quarterly')},
            'yearly': {ticker: returns.get('yearly')}
        },
        'volatility': {
            'weekly': {ticker: volatility.get('weekly')},
            'monthly': {ticker: volatility.get('monthly')},
            'quarterly': {ticker: volatility.get('quarterly')},
            'yearly': {ticker: volatility.get('yearly')}
        },
        'technical_indicators': tech_indicators
    }
    
    # Trend analysis
    if len(data) >= 90:  # Need at least 3 months of data
        try:
            # Identify if in uptrend, downtrend, or consolidation
            short_ma = data['Close'].rolling(window=20).mean()
            long_ma = data['Close'].rolling(window=50).mean()
            
            # Get scalar values for comparison - proper way to convert Series to float
            short_current = short_ma.iloc[-1].item()
            short_prev = short_ma.iloc[-2].item()
            long_current = long_ma.iloc[-1].item()
            long_prev = long_ma.iloc[-2].item()
            
            # Get value from N periods ago for trend changes
            trend_change_window = min(20, len(data)-1)
            short_past = short_ma.iloc[-trend_change_window].item()
            long_past = long_ma.iloc[-trend_change_window].item()
            
            # Current trend
            if short_current > long_current and short_prev > long_prev:
                trend = "Uptrend"
            elif short_current < long_current and short_prev < long_prev:
                trend = "Downtrend"
            else:
                trend = "Consolidation/Mixed"
                
            # Trend change detection
            if short_current > long_current and short_past < long_past:
                trend = "Recent Bullish Crossover"
            elif short_current < long_current and short_past > long_past:
                trend = "Recent Bearish Crossover"
                
            summary['trend'] = trend
        except Exception as e:
            print(f"Error calculating trend for {ticker}: {e}")
            summary['trend'] = "Calculation Error"
    
    return summary

def summarize_macro_indicator(data, name, series_id):
    """Create a summary of macroeconomic data for LLM consumption"""
    if data.empty:
        return None
    
    # Sort data by index to ensure it's in chronological order
    data = data.sort_index()
    
    # Get current and previous values
    latest_date = data.index[-1].strftime('%Y-%m-%d')
    current_value = data.iloc[-1]
    
    # Calculate changes
    changes = {}
    if len(data) > 1:
        previous_value = data.iloc[-2]
        changes['previous_period'] = round(current_value - previous_value, 2)
        changes['previous_period_pct'] = round((current_value / previous_value - 1) * 100, 2) if previous_value != 0 else None
    
    # Year-over-year change if we have data from a year ago
    yearly_data = data[data.index < (data.index[-1] - pd.DateOffset(months=11))]
    if not yearly_data.empty:
        year_ago_value = yearly_data.iloc[-1]
        changes['year_over_year'] = round(current_value - year_ago_value, 2)
        changes['year_over_year_pct'] = round((current_value / year_ago_value - 1) * 100, 2) if year_ago_value != 0 else None
    
    # Trend analysis - determine if indicator is increasing, decreasing, or stable
    if len(data) >= 3:
        recent_values = data.iloc[-3:].values
        if all(recent_values[i] < recent_values[i+1] for i in range(len(recent_values)-1)):
            trend = "Increasing"
        elif all(recent_values[i] > recent_values[i+1] for i in range(len(recent_values)-1)):
            trend = "Decreasing"
        else:
            trend = "Mixed/Stable"
    else:
        trend = "Insufficient Data"
    
    # Create summary object
    summary = {
        'name': name,
        'series_id': series_id,
        'date': latest_date,
        'current_value': round(current_value, 2) if not pd.isna(current_value) else None,
        'changes': changes,
        'trend': trend
    }
    
    # Add 5-year range
    if len(data) > 12:  # If we have enough data
        summary['five_year_high'] = round(data.max(), 2)
        summary['five_year_low'] = round(data.min(), 2)
        summary['current_vs_high_pct'] = round((current_value / data.max() - 1) * 100, 2) if data.max() != 0 else None
    
    return summary

# ----------------------------
# 1. Retrieve Macroeconomic Data with Aggregation
# ----------------------------

# Expanded map of indicator names to FRED series IDs
macro_indicators = {
    # Economic Growth
    "GDP": "GDP",                                 # Gross Domestic Product
    "Real GDP": "GDPC1",                          # Real Gross Domestic Product
    "GDP Growth Rate": "A191RL1Q225SBEA",         # Real GDP Growth Rate
    
    # Employment
    "Unemployment Rate": "UNRATE",                # Unemployment Rate
    "Initial Jobless Claims": "ICSA",             # Initial Claims for Unemployment Insurance
    "Employment Population Ratio": "EMRATIO",     # Employment-Population Ratio
    
    # Inflation
    "Consumer Price Index (CPI)": "CPIAUCSL",     # All items CPI
    "Core CPI": "CPILFESL",                       # CPI less food and energy
    "Producer Price Index (PPI)": "PPIACO",       # Producer Price Index: All Commodities
    "PCE Price Index": "PCEPILFE",                # Personal Consumption Expenditures Price Index (Core)
    
    # Monetary Policy
    "Federal Funds Rate": "FEDFUNDS",             # Effective Federal Funds Rate
    "Money Supply M2": "M2SL",                    # Money Stock M2
    "10-Year Treasury Rate": "GS10",              # 10-Year Treasury Constant Maturity Rate
    "2-Year Treasury Rate": "GS2",                # 2-Year Treasury Constant Maturity Rate
    "Treasury Yield Curve": "T10Y2Y",             # 10-Year Treasury Minus 2-Year Treasury
    
    # Credit Spreads
    "Baa Corporate Bond Yield": "BAA",            # Moody's Seasoned Baa Corporate Bond Yield
    "Aaa Corporate Bond Yield": "AAA",            # Moody's Seasoned Aaa Corporate Bond Yield
    "Corporate Bond Spread": "BAA10Y",            # Baa Corporate Bond Yield Relative to 10-Year Treasury
    "High Yield Spread": "BAAFFM",                # Baa Corporate Bond Yield Relative to Fed Funds
    "TED Spread": "TEDRATE",                      # TED Spread (3-Month LIBOR - 3-Month Treasury)
    
    # Liquidity Measures
    "Commercial Bank Assets": "TLAACBW027SBOG",   # Total Assets, All Commercial Banks
    "Excess Reserves": "EXCSRESNS",               # Excess Reserves of Depository Institutions
    "Commercial Paper Outstanding": "COMPOUT",     # Commercial Paper Outstanding
    "Bank Credit": "TOTBKCR",                     # Bank Credit, All Commercial Banks
    "Institutional Money Funds": "WIMFSL",         # Institutional Money Funds
    
    # Key Indicators
    "Consumer Sentiment": "UMCSENT",              # University of Michigan Consumer Sentiment
    "Retail Sales": "RSAFS",                      # Retail Sales
    "Industrial Production": "INDPRO",            # Industrial Production Index
    
    # Adding the corrected indicators with valid FRED series IDs
    "Average Hourly Earnings": "CES0500000003",  # Wage growth
    "Average Hourly Earnings YoY": "AHETPI",     # Year-over-year wage growth (corrected)
    "ISM Manufacturing PMI": "MANEMP",           # Manufacturing employment as proxy (corrected)
    "ISM Services PMI": "SRVPRD",                # Services production index (corrected)
}

def get_fred_data(series_id, start, end):
    """Retrieve a time series from FRED."""
    try:
        series = fred.get_series(series_id, start, end)
        return series
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.Series()

# ----------------------------
# 2. Market Data with Aggregation
# ----------------------------

# Define lookback periods for return calculations
lookback_periods = {
    "daily": DAILY_LOOKBACK,
    "weekly": WEEKLY_LOOKBACK,
    "monthly": MONTHLY_LOOKBACK,
    "quarterly": QUARTERLY_LOOKBACK,
    "yearly": YEARLY_LOOKBACK
}

# Define our key market indices and assets
key_assets = {
    # Major Indices
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "NASDAQ": "^IXIC",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
    
    # Tech Sector
    "Technology Sector": "XLK",
    "Semiconductor ETF": "SMH",
    "Software ETF": "IGV",
    "AI & Robotics ETF": "BOTZ",
    
    # MAG 7 Stocks
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    
    # Other AI Players
    "AMD": "AMD",
    "Palantir": "PLTR",
    "C3.ai": "AI",
    "Super Micro": "SMCI",
    
    # Crypto
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}

def get_yfinance_data(ticker, start_date, end_date, interval="1d"):
    """Retrieve historical data from yfinance for the given ticker."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# ----------------------------
# 3. Generate Correlation Matrix and Insights
# ----------------------------

def track_correlation_changes(assets_data, lookback_periods={"short": 30, "long": 90}):
    """Track changes in correlations between key asset pairs to identify divergences"""
    import numpy as np
    
    # Define key pairs to track (important relationships that can signal market shifts)
    key_pairs = [
        # Market indices vs. volatility (inverse relationship expected)
        ("^GSPC", "^VIX"),  # S&P 500 vs VIX - classic fear gauge
        
        # Crypto vs. equities (divergence can signal risk-on/risk-off shifts)
        ("^GSPC", "BTC-USD"),  # S&P 500 vs Bitcoin
        ("NVDA", "BTC-USD"),   # NVIDIA vs Bitcoin (AI/crypto narrative)
        
        # Tech vs. broad market (divergence can signal sector rotation)
        ("XLK", "^GSPC"),      # Tech Sector vs S&P 500
        ("SMH", "^GSPC"),      # Semiconductors vs S&P 500
        
        # Growth vs. defensive (divergence can signal economic outlook shifts)
        ("XLK", "^DJI"),       # Tech vs Dow Jones (growth vs. value)
        
        # Different market cap indices (small vs large cap divergence)
        ("^GSPC", "^RUT"),     # S&P 500 vs Russell 2000
        
        # Key tech stocks vs their sector (leadership changes)
        ("AAPL", "XLK"),       # Apple vs Tech Sector
        ("NVDA", "SMH"),       # NVIDIA vs Semiconductor ETF
    ]
    
    correlation_changes = []
    
    # Calculate short and long-term correlations for each pair
    for ticker1, ticker2 in key_pairs:
        if ticker1 not in assets_data or ticker2 not in assets_data:
            continue
            
        data1 = assets_data[ticker1]
        data2 = assets_data[ticker2]
        
        # Skip if not enough data
        if len(data1) < max(lookback_periods.values()) or len(data2) < max(lookback_periods.values()):
            continue
            
        # Calculate returns
        returns1 = data1['Close'].pct_change().dropna()
        returns2 = data2['Close'].pct_change().dropna()
        
        # Align dates
        common_dates = returns1.index.intersection(returns2.index)
        if len(common_dates) < max(lookback_periods.values()):
            continue
            
        aligned_returns1 = returns1.loc[common_dates]
        aligned_returns2 = returns2.loc[common_dates]
        
        # Calculate correlations for different time periods
        correlations = {}
        
        for period_name, days in lookback_periods.items():
            if len(common_dates) >= days:
                # Ensure we have at least 2 elements for correlation
                if days < 2:
                    continue
                    
                # Get the returns for the period and convert to numpy arrays
                period_returns1 = aligned_returns1[-days:].values
                period_returns2 = aligned_returns2[-days:].values
                
                # Handle empty arrays or arrays with single elements
                if len(period_returns1) < 2 or len(period_returns2) < 2:
                    continue
                
                try:
                    # Use numpy's corrcoef for a simple scalar result
                    corr = np.corrcoef(period_returns1, period_returns2)[0, 1]
                    
                    if not np.isnan(corr):
                        correlations[period_name] = round(corr, 2)
                except Exception as e:
                    print(f"Error calculating correlation for {ticker1} vs {ticker2} ({period_name}): {e}")
        
        # Calculate correlation change if we have both periods
        if "short" in correlations and "long" in correlations:
            correlation_change = correlations["short"] - correlations["long"]
            
            # Record pair and correlations
            pair_data = {
                "pair": f"{ticker1} vs {ticker2}",
                "short_term_corr": correlations["short"],
                "long_term_corr": correlations["long"],
                "correlation_change": round(correlation_change, 2)
            }
            
            # Add interpretation
            if abs(correlation_change) >= 0.2:
                if correlation_change > 0:
                    pair_data["interpretation"] = "Strengthening relationship"
                else:
                    pair_data["interpretation"] = "Weakening relationship"
                    
                # Special case for typically inverse relationships (like S&P 500 vs VIX)
                if ticker2 == "^VIX" and ticker1 in ["^GSPC", "^DJI", "^IXIC"]:
                    if correlation_change > 0:
                        pair_data["interpretation"] = "WARNING: Inverse relationship weakening"
                    else:
                        pair_data["interpretation"] = "Normal: Inverse relationship strengthening"
            else:
                pair_data["interpretation"] = "Stable relationship"
                
            correlation_changes.append(pair_data)
    
    # Sort by absolute correlation change (largest changes first)
    correlation_changes.sort(key=lambda x: abs(x["correlation_change"]), reverse=True)
    
    return correlation_changes

def calculate_correlations(assets_data):
    """Calculate correlation matrix between different assets"""
    # Get list of tickers
    tickers = list(assets_data.keys())
    # Create empty DataFrame for correlation matrix
    corr_matrix = pd.DataFrame(index=tickers, columns=tickers)
    
    # Calculate returns for each asset
    returns_data = {}
    for ticker, data in assets_data.items():
        if not data.empty and len(data) > 30:
            returns_data[ticker] = data['Close'].pct_change().dropna()
    
    # Calculate correlation coefficients
    for i, ticker1 in enumerate(tickers):
        if ticker1 not in returns_data:
            continue
            
        for j, ticker2 in enumerate(tickers):
            if ticker2 not in returns_data:
                continue
                
            # Get returns data
            ret1 = returns_data[ticker1]
            ret2 = returns_data[ticker2]
            
            # Align dates and calculate correlation
            common_dates = ret1.index.intersection(ret2.index)
            if len(common_dates) > 10:
                try:
                    aligned_ret1 = ret1.loc[common_dates].values
                    aligned_ret2 = ret2.loc[common_dates].values
                    
                    # Filter out any NaN values
                    mask = ~(np.isnan(aligned_ret1) | np.isnan(aligned_ret2))
                    filtered_ret1 = aligned_ret1[mask]
                    filtered_ret2 = aligned_ret2[mask]
                    
                    if len(filtered_ret1) > 10:  # Still enough data points after filtering
                        corr = np.corrcoef(filtered_ret1, filtered_ret2)[0, 1]
                        # Check if correlation is valid number
                        if not np.isnan(corr) and not np.isinf(corr):
                            corr_matrix.loc[ticker1, ticker2] = round(corr, 2)
                        else:
                            corr_matrix.loc[ticker1, ticker2] = 0.0  # Default to zero for invalid correlations
                    else:
                        corr_matrix.loc[ticker1, ticker2] = 0.0  # Not enough data points
                except Exception as e:
                    print(f"Error calculating correlation between {ticker1} and {ticker2}: {e}")
                    corr_matrix.loc[ticker1, ticker2] = 0.0
            else:
                corr_matrix.loc[ticker1, ticker2] = 0.0  # Not enough common dates
    
    # Track correlation changes for key pairs
    correlation_changes = track_correlation_changes(assets_data)
    
    return corr_matrix, correlation_changes

# First define the add_forward_looking_indicators function before generate_market_insights
def add_forward_looking_indicators(data=None):
    """
    Add forward-looking indicators to the analysis.
    """
    try:
        forward_indicators = {}
        
        # Define the get_fomc_statements function inline
        def get_fomc_statements():
            # This would typically fetch from an API or database
            # For now, return a placeholder with the most recent statement
            return {
                "latest_date": "2025-03-20",
                "latest_statement_summary": "The Committee decided to maintain the target range for the federal funds rate at 4-1/4 to 4-1/2 percent. The Committee will continue to monitor incoming information for the economic outlook, including progress on lowering inflation to our 2 percent objective.",
                "key_phrases": [
                    "risks to achieving employment and inflation goals remain roughly in balance",
                    "inflation has made progress toward our 2 percent objective",
                    "economic activity has been expanding at a moderate pace",
                    "job gains have remained solid"
                ],
                "policy_stance": "Neutral with a bias toward easing",
                "next_meeting_date": "2025-04-30"
            }
        
        # Get FOMC statements
        fomc_data = get_fomc_statements()
        forward_indicators["fomc_statements"] = fomc_data
        
        # Get market-implied rate expectations (would typically come from futures market)
        forward_indicators["rate_expectations"] = {
            "current_rate": 4.33,
            "implied_3m_change": -0.25,
            "implied_6m_change": -0.5,
            "implied_12m_change": -0.75,
            "market_probability_cut_next_meeting": 65.0,
            "market_probability_hike_next_meeting": 5.0
        }
        
        # Get PMI data if available (forward-looking business sentiment)
        forward_indicators["pmi_data"] = {
            "manufacturing_pmi": 48.5,  # Below 50 = contraction
            "services_pmi": 52.1,      # Above 50 = expansion
            "composite_pmi": 51.0,
            "manufacturing_trend": "Contracting",
            "services_trend": "Expanding",
            "composite_trend": "Expanding Slowly"
        }
        
        return forward_indicators
    except Exception as e:
        print(f"Error in add_forward_looking_indicators: {str(e)}")
        # Return minimal structure in case of error
        return {
            "error": str(e),
            "fomc_statements": {},
            "rate_expectations": {},
            "pmi_data": {}
        }

# Function to extract key insights
def generate_market_insights(macro_summaries, market_summaries, correlation_data):
    """Generate key insights from the data for LLM consumption"""
    
    # Unpack correlation data (matrix and changes)
    correlation_matrix, correlation_changes = correlation_data
    
    # Get forward-looking indicators
    forward_indicators = add_forward_looking_indicators()
    
    # Format the date
    report_date = datetime.datetime.today().strftime("%Y-%m-%d")
    
    # Create key insights based on current data
    insights = []
    
    # Insight 1: Overall market sentiment
    sp500_data = next((item for item in market_summaries if item['ticker'] == '^GSPC'), None)
    vix_data = next((item for item in market_summaries if item['ticker'] == '^VIX'), None)
    
    if sp500_data and vix_data:
        sp500_trend = sp500_data.get('trend', 'Unknown')
        vix_value = vix_data.get('current_price').get('^VIX', 0)
        # Convert pandas objects to native types
        if hasattr(vix_value, 'item'):
            vix_value = vix_value.item()
        
        sentiment = "Neutral"
        if sp500_trend == "Uptrend" and vix_value < 20:
            sentiment = "Bullish"
        elif sp500_trend == "Downtrend" and vix_value > 25:
            sentiment = "Bearish"
        elif sp500_trend == "Recent Bullish Crossover":
            sentiment = "Cautiously Bullish"
        elif sp500_trend == "Recent Bearish Crossover":
            sentiment = "Cautiously Bearish"
            
        insights.append(f"Market Sentiment: {sentiment}")
    
    # Insight 2: Tech sector performance
    tech_etf = next((item for item in market_summaries if item['ticker'] == 'XLK'), None)
    if tech_etf and 'returns' in tech_etf:
        weekly_return = tech_etf['returns'].get('weekly', {}).get('XLK', 0)
        monthly_return = tech_etf['returns'].get('monthly', {}).get('XLK', 0)
        yearly_return = tech_etf['returns'].get('yearly', {}).get('XLK', 0)
        
        # Convert pandas objects to native types
        if hasattr(weekly_return, 'item'):
            weekly_return = weekly_return.item()
        if hasattr(monthly_return, 'item'):
            monthly_return = monthly_return.item()
        if hasattr(yearly_return, 'item'):
            yearly_return = yearly_return.item()
            
        if 'trend' in tech_etf:
            insights.append(f"Tech Sector: {tech_etf['trend']} (W: {weekly_return}%, M: {monthly_return}%, Y: {yearly_return}%)")
    
    # Insight 3: MAG 7 performance
    mag7_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
    mag7_data = [item for item in market_summaries if item['ticker'] in mag7_tickers]
    
    if mag7_data:
        # Calculate average monthly return for MAG 7
        monthly_returns = []
        for item in mag7_data:
            if 'returns' in item and 'monthly' in item['returns']:
                ticker = item['ticker']
                monthly_return = item['returns']['monthly'].get(ticker)
                if monthly_return is not None:
                    # Convert pandas objects to native types
                    if hasattr(monthly_return, 'item'):
                        monthly_return = monthly_return.item()
                    monthly_returns.append(monthly_return)
        
        if monthly_returns:
            avg_monthly_return = sum(monthly_returns) / len(monthly_returns)
            insights.append(f"MAG 7 Average Monthly Return: {round(avg_monthly_return, 2)}%")
        
        # Best and worst performers
        if len(mag7_data) >= 2:
            performance_list = []
            for item in mag7_data:
                if 'returns' in item and 'monthly' in item['returns']:
                    ticker = item['ticker']
                    monthly_return = item['returns']['monthly'].get(ticker)
                    if monthly_return is not None:
                        # Convert pandas objects to native types
                        if hasattr(monthly_return, 'item'):
                            monthly_return = monthly_return.item()
                        performance_list.append((item['name'], monthly_return))
            
            if performance_list:
                # Convert any pandas Series objects to native Python types for sorting
                clean_performance_list = []
                for name, val in performance_list:
                    if hasattr(val, 'item'):  # Check if it's a pandas/numpy object
                        clean_performance_list.append((name, val.item()))
                    else:
                        clean_performance_list.append((name, val))
                
                # Sort the cleaned list
                clean_performance_list.sort(key=lambda x: x[1])
                
                worst_name, worst_return = clean_performance_list[0]
                best_name, best_return = clean_performance_list[-1]
                
                insights.append(f"Best MAG 7: {best_name} ({best_return}% monthly)")
                insights.append(f"Worst MAG 7: {worst_name} ({worst_return}% monthly)")
    
    # Insight 4: Key economic indicators
    inflation = next((item for item in macro_summaries if item['name'] == 'Consumer Price Index (CPI)'), None)
    interest_rate = next((item for item in macro_summaries if item['name'] == 'Federal Funds Rate'), None)
    unemployment = next((item for item in macro_summaries if item['name'] == 'Unemployment Rate'), None)
    
    if inflation and 'changes' in inflation:
        yoy_change = inflation['changes'].get('year_over_year_pct')
        if yoy_change is not None:
            # Convert pandas objects to native types
            if hasattr(yoy_change, 'item'):
                yoy_change = yoy_change.item()
            insights.append(f"Inflation (CPI): {round(yoy_change, 1)}% year-over-year, Trend: {inflation['trend']}")
    
    if interest_rate and 'current_value' in interest_rate:
        rate_val = interest_rate['current_value']
        # Convert pandas objects to native types
        if hasattr(rate_val, 'item'):
            rate_val = rate_val.item()
        insights.append(f"Federal Funds Rate: {rate_val}%, Trend: {interest_rate['trend']}")
    
    if unemployment and 'current_value' in unemployment:
        unemp_val = unemployment['current_value']
        # Convert pandas objects to native types
        if hasattr(unemp_val, 'item'):
            unemp_val = unemp_val.item()
        insights.append(f"Unemployment Rate: {unemp_val}%, Trend: {unemployment['trend']}")
    
    # Insight 5: Credit Spreads (Leading Indicators)
    baa_10y_spread = next((item for item in macro_summaries if item['name'] == 'Corporate Bond Spread'), None)
    ted_spread = next((item for item in macro_summaries if item['name'] == 'TED Spread'), None)
    
    if baa_10y_spread and 'current_value' in baa_10y_spread:
        spread_val = baa_10y_spread['current_value']
        if hasattr(spread_val, 'item'):
            spread_val = spread_val.item()
        
        # Interpret the spread (high spreads indicate market stress)
        spread_status = "Normal"
        if spread_val > 3.0:
            spread_status = "Elevated (Warning)"
        elif spread_val > 4.0:
            spread_status = "High (Danger)"
            
        insights.append(f"Corporate Bond Spread: {round(spread_val, 2)}% ({spread_status}), Trend: {baa_10y_spread['trend']}")
    
    if ted_spread and 'current_value' in ted_spread:
        ted_val = ted_spread['current_value']
        if hasattr(ted_val, 'item'):
            ted_val = ted_val.item()
            
        # Interpret the TED spread (high spreads indicate banking system stress)
        ted_status = "Normal"
        if ted_val > 0.5:
            ted_status = "Elevated (Warning)"
        elif ted_val > 1.0:
            ted_status = "High (Danger)"
            
        insights.append(f"TED Spread: {round(ted_val, 2)}% ({ted_status}), Trend: {ted_spread['trend']}")
    
    # Insight 6: Liquidity Measures
    bank_credit = next((item for item in macro_summaries if item['name'] == 'Bank Credit'), None)
    excess_reserves = next((item for item in macro_summaries if item['name'] == 'Excess Reserves'), None)
    
    if bank_credit and 'changes' in bank_credit:
        yoy_change = bank_credit['changes'].get('year_over_year_pct')
        if yoy_change is not None:
            if hasattr(yoy_change, 'item'):
                yoy_change = yoy_change.item()
                
            # Interpret bank credit growth
            credit_status = "Normal"
            if yoy_change < 2.0:
                credit_status = "Low (Restrictive)"
            elif yoy_change > 10.0:
                credit_status = "High (Expansionary)"
                
            insights.append(f"Bank Credit Growth: {round(yoy_change, 1)}% YoY ({credit_status}), Trend: {bank_credit['trend']}")
    
    if excess_reserves and 'current_value' in excess_reserves:
        reserve_val = excess_reserves['current_value']
        if hasattr(reserve_val, 'item'):
            reserve_val = reserve_val.item()
        reserve_val_billions = reserve_val / 1000  # Convert to billions for readability
            
        insights.append(f"Excess Reserves: ${round(reserve_val_billions, 1)} billion, Trend: {excess_reserves['trend']}")
    
    # Insight 7: Key correlation changes (divergences)
    if correlation_changes:
        # Add the most significant correlation changes to insights
        significant_changes = [pair for pair in correlation_changes if abs(pair['correlation_change']) >= 0.15]
        
        if significant_changes:
            # Add up to 3 most significant correlation changes
            for i, change in enumerate(significant_changes[:3]):
                pair_name = change['pair']
                short_corr = change['short_term_corr']
                change_val = change['correlation_change']
                interpretation = change['interpretation']
                
                # Format the change with a sign
                change_str = f"+{change_val}" if change_val > 0 else f"{change_val}"
                
                insights.append(f"Correlation Alert: {pair_name} ({change_str}, {interpretation})")
    
    # Insight 8: Crypto market
    btc_data = next((item for item in market_summaries if item['ticker'] == 'BTC-USD'), None)
    eth_data = next((item for item in market_summaries if item['ticker'] == 'ETH-USD'), None)
    
    if btc_data and eth_data and 'returns' in btc_data and 'returns' in eth_data:
        btc_monthly = btc_data['returns'].get('monthly', {}).get('BTC-USD', 0)
        eth_monthly = eth_data['returns'].get('monthly', {}).get('ETH-USD', 0)
        btc_price = btc_data['current_price'].get('BTC-USD', 0)
        eth_price = eth_data['current_price'].get('ETH-USD', 0)
        
        # Convert pandas objects to native types
        if hasattr(btc_monthly, 'item'):
            btc_monthly = btc_monthly.item()
        if hasattr(eth_monthly, 'item'):
            eth_monthly = eth_monthly.item()
        if hasattr(btc_price, 'item'):
            btc_price = btc_price.item()
        if hasattr(eth_price, 'item'):
            eth_price = eth_price.item()
        
        insights.append(f"BTC: ${round(btc_price)} ({btc_monthly}% monthly), ETH: ${round(eth_price)} ({eth_monthly}% monthly)")
    
    # Add new insights based on forward-looking indicators
    
    # Insight: Fed Policy & Expectations
    fed_statements = forward_indicators.get('fomc_statements', {})
    rate_expectations = forward_indicators.get('rate_expectations', {})
    
    if fed_statements:
        latest_statement = fed_statements['latest_statement_summary']
        insights.append(f"FOMC Stance: {latest_statement}")
    
    if rate_expectations:
        next_meeting = rate_expectations.get('next_meeting_date')
        if next_meeting:
            insights.append(f"Market Rate Expectation: {next_meeting}")
    
    # Insight: PMI Data
    pmi_data = forward_indicators.get('pmi_data', {})
    
    manufacturing_pmi = pmi_data.get('manufacturing_pmi')
    services_pmi = pmi_data.get('services_pmi')
    
    if manufacturing_pmi and 'manufacturing_trend' in pmi_data:
        pmi_status = pmi_data['manufacturing_trend']
        insights.append(f"Manufacturing PMI: {round(manufacturing_pmi, 1)}% ({pmi_status}), Trend: {pmi_data['manufacturing_trend']}")
    
    if services_pmi and 'services_trend' in pmi_data:
        pmi_status = pmi_data['services_trend']
        insights.append(f"Services PMI: {round(services_pmi, 1)}% ({pmi_status}), Trend: {pmi_data['services_trend']}")
    
    # Create the report structure
    report = {
        "report_date": report_date,
        "summary": "Daily Financial Market Report",
        "key_insights": insights,
        "macro_data": macro_summaries,
        "market_data": market_summaries,
        "correlations": correlation_matrix.fillna(0).to_dict(),  # Replace NaN with 0 before converting
        "correlation_changes": correlation_changes,  # Add the correlation changes to the report
        "forward_looking_indicators": forward_indicators  # Add the forward-looking indicators section
    }
    
    return report

# Main execution
if __name__ == "__main__":
    # Retrieve and process macroeconomic data
    print("=== Retrieving and Summarizing Macroeconomic Indicators ===")
    macro_summaries = []

    for indicator, series_id in macro_indicators.items():
        data = get_fred_data(series_id, start_date, end_date)
        if not data.empty:
            print(f"Processing {indicator} ({series_id}) - {len(data)} data points")
            summary = summarize_macro_indicator(data, indicator, series_id)
            if summary:
                macro_summaries.append(summary)
        else:
            print(f"No data returned for {indicator} ({series_id})")
    
    # Retrieve and process market data
    print("\n=== Retrieving and Summarizing Market Data ===")
    market_summaries = []

    for name, ticker in key_assets.items():
        data = get_yfinance_data(ticker, start_date, end_date)
        if not data.empty:
            print(f"Processing {name} ({ticker}) - {len(data)} data points")
            summary = create_market_summary(data, name, ticker, lookback_periods)
            if summary:
                market_summaries.append(summary)
        else:
            print(f"No data returned for {name} ({ticker})")
    
    # Calculate correlations and track correlation changes
    print("\n=== Calculating Asset Correlations and Tracking Changes ===")
    assets_data = {}
    for name, ticker in key_assets.items():
        data = get_yfinance_data(ticker, start_date, end_date)
        if not data.empty:
            assets_data[ticker] = data
    
    correlation_data = calculate_correlations(assets_data)
    
    # Generate the daily report
    print("\n=== Generating Daily Market Report ===")
    daily_report = generate_market_insights(macro_summaries, market_summaries, correlation_data)
    
    # Create a custom JSON encoder to handle pandas/numpy types
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.fillna(0).to_dict()  # Replace NaN with 0 before converting
            if isinstance(obj, np.ndarray):
                return np.where(np.isnan(obj), 0, obj).tolist()  # Replace NaN with 0
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return 0.0  # Replace NaN/inf with 0
                return float(obj)
            if isinstance(obj, datetime.datetime):
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(obj, datetime.date):
                return obj.strftime("%Y-%m-%d")
            if pd.isna(obj):
                return 0  # Replace NaN with 0
            return super(CustomJSONEncoder, self).default(obj)
    
    # Save the report to JSON file using the custom encoder
    report_filename = f"data/daily_market_report_{datetime.datetime.today().strftime('%Y-%m-%d')}.json"
    with open(report_filename, 'w') as f:
        json.dump(daily_report, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"\nDaily report saved to {report_filename}")
    print("This report is optimized for LLM consumption with aggregated data instead of raw time series.")

# Add a new function to fetch and process FOMC statements
def get_fomc_statements(days_back=60):
    """
    Fetch recent FOMC statements and extract key themes
    """
    from datetime import datetime, timedelta
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for the URL
        start_str = start_date.strftime("%m/%d/%Y")
        end_str = end_date.strftime("%m/%d/%Y")
        
        # Define key terms to search for in statements
        hawkish_terms = [
            "inflation remains elevated", 
            "restrictive stance", 
            "higher rates", 
            "upside risks", 
            "further tightening",
            "price stability",
            "persistent inflation"
        ]
        
        dovish_terms = [
            "softening labor market", 
            "moderate pace", 
            "progress", 
            "disinflation", 
            "appropriate to slow",
            "balanced approach",
            "downside risks",
            "achieved substantial progress"
        ]
        
        # The Fed Board's FOMC calendar page has all statements
        url = f"https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find FOMC statement links
        statement_links = []
        for link in soup.find_all('a', href=True):
            if "fomcpresconf" in link['href'] or "monetary" in link['href'] and "statement" in link.text.lower():
                statement_links.append({
                    'date': link.find_previous('div', class_='fomc-meeting__date').text.strip() if link.find_previous('div', class_='fomc-meeting__date') else "Unknown Date",
                    'url': "https://www.federalreserve.gov" + link['href'] if not link['href'].startswith('http') else link['href'],
                    'title': link.text
                })
        
        # Get the most recent statement
        recent_statements = statement_links[:2]  # Get the two most recent statements
        
        statement_analysis = []
        
        for statement_info in recent_statements:
            try:
                statement_response = requests.get(statement_info['url'])
                statement_soup = BeautifulSoup(statement_response.text, 'html.parser')
                
                # The main content is typically in a div with class 'col-xs-12 col-sm-8 col-md-8'
                content_div = statement_soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')
                if not content_div:
                    content_div = statement_soup  # Use the whole soup if we can't find the specific div
                
                paragraphs = content_div.find_all('p')
                statement_text = ' '.join([p.text for p in paragraphs])
                
                # Count hawkish and dovish terms
                hawkish_count = sum(statement_text.lower().count(term.lower()) for term in hawkish_terms)
                dovish_count = sum(statement_text.lower().count(term.lower()) for term in dovish_terms)
                
                # Determine sentiment
                if hawkish_count > dovish_count:
                    sentiment = "Hawkish"
                elif dovish_count > hawkish_count:
                    sentiment = "Dovish"
                else:
                    sentiment = "Neutral"
                
                # Extract key rate decision if present
                rate_decision = "No clear rate decision mentioned"
                rate_patterns = [
                    r"raise the target range for the federal funds rate to (\d+(?:\.\d+)?)(?: to|-) (\d+(?:\.\d+)?)",
                    r"maintain the target range for the federal funds rate at (\d+(?:\.\d+)?)(?: to|-) (\d+(?:\.\d+)?)",
                    r"lower the target range for the federal funds rate to (\d+(?:\.\d+)?)(?: to|-) (\d+(?:\.\d+)?)"
                ]
                
                for pattern in rate_patterns:
                    match = re.search(pattern, statement_text, re.IGNORECASE)
                    if match:
                        lower_bound = match.group(1)
                        upper_bound = match.group(2)
                        if "raise" in pattern:
                            action = "raised"
                        elif "lower" in pattern:
                            action = "lowered"
                        else:
                            action = "maintained"
                        rate_decision = f"Fed {action} rates to {lower_bound}% - {upper_bound}%"
                        break
                
                # Extract future guidance
                guidance = "No clear forward guidance"
                guidance_patterns = [
                    r"(committee will continue to|anticipates that ongoing|expects that some additional|judges that we can proceed carefully)",
                    r"(remains highly attentive to|is attentive to|closely monitoring|will be data dependent)"
                ]
                
                for pattern in guidance_patterns:
                    match = re.search(pattern, statement_text, re.IGNORECASE)
                    if match:
                        context_start = max(0, match.start() - 100)
                        context_end = min(len(statement_text), match.end() + 150)
                        guidance = statement_text[context_start:context_end].strip()
                        # Clean up the guidance text
                        guidance = re.sub(r'\s+', ' ', guidance)
                        break
                
                statement_analysis.append({
                    "date": statement_info['date'],
                    "sentiment": sentiment,
                    "hawkish_terms": hawkish_count,
                    "dovish_terms": dovish_count,
                    "rate_decision": rate_decision,
                    "forward_guidance": guidance[:150] + "..." if len(guidance) > 150 else guidance
                })
                
            except Exception as e:
                print(f"Error processing FOMC statement {statement_info['url']}: {e}")
        
        return statement_analysis
    
    except Exception as e:
        print(f"Error fetching FOMC statements: {e}")
        return []

# Add a function to get market-implied rate expectations
def get_rate_expectations():
    """
    Extract market-implied rate expectations from Fed Funds futures and Eurodollar futures
    """
    import pandas as pd
    import numpy as np
    import yfinance as yf
    
    try:
        # Get Fed Funds futures data (30-day Federal Funds Futures)
        ff1 = yf.download("ZQ=F", period="1mo")  # Front-month Fed Funds future
        ff3 = yf.download("^IRX", period="1mo")  # 3-month Treasury yield as proxy
        
        # Calculate implied rates from futures prices
        if not ff1.empty:
            # Fed Funds futures pricing implies rate as: 100 - price
            latest_ff1_price = ff1['Close'].iloc[-1]
            ff1_implied_rate = round(100 - latest_ff1_price, 2)
        else:
            ff1_implied_rate = None
            
        if not ff3.empty:
            # Use 3-month Treasury as proxy for 3-month ahead expectations
            ff3_implied_rate = round(ff3['Close'].iloc[-1], 2)
        else:
            ff3_implied_rate = None
            
        # Get current Fed Funds rate for comparison
        try:
            current_rate_data = get_fred_data("FEDFUNDS", start_date, end_date)
            current_rate = round(current_rate_data.iloc[-1]['FEDFUNDS'], 2)
        except:
            current_rate = None
            
        # Calculate market expectations
        rate_expectations = {
            "current_fed_funds_rate": current_rate,
            "next_meeting_implied_rate": ff1_implied_rate,
            "three_month_implied_rate": ff3_implied_rate,
            "expected_change_next": None if (current_rate is None or ff1_implied_rate is None) else round(ff1_implied_rate - current_rate, 2),
            "expected_change_three_month": None if (current_rate is None or ff3_implied_rate is None) else round(ff3_implied_rate - current_rate, 2)
        }
        
        # Add trend interpretation
        if rate_expectations["expected_change_next"] is not None:
            if rate_expectations["expected_change_next"] > 0.10:
                rate_expectations["next_meeting_expectation"] = "Rate hike expected"
            elif rate_expectations["expected_change_next"] < -0.10:
                rate_expectations["next_meeting_expectation"] = "Rate cut expected"
            else:
                rate_expectations["next_meeting_expectation"] = "No change expected"
                
        if rate_expectations["expected_change_three_month"] is not None:
            if rate_expectations["expected_change_three_month"] > 0.25:
                rate_expectations["three_month_expectation"] = "Rate hikes expected"
            elif rate_expectations["expected_change_three_month"] < -0.25:
                rate_expectations["three_month_expectation"] = "Rate cuts expected"
            else:
                rate_expectations["three_month_expectation"] = "Stable rates expected"
        
        return rate_expectations
        
    except Exception as e:
        print(f"Error getting rate expectations: {e}")
        return {}

# Add function to get dot plot data if available
def get_dot_plot_summary():
    """
    Summarize the latest Fed dot plot projections
    Note: This uses hardcoded recent data since the dot plot is released quarterly
    and requires manual extraction from PDFs
    """
    # Most recent dot plot data (would be updated manually quarterly)
    # Format: [year, median_projection]
    latest_dot_plot_date = "March 2023"  # Update this when new dot plots are released
    dot_plot_data = [
        {"year": "2023", "median_rate": 5.1, "range_low": 5.0, "range_high": 5.5},
        {"year": "2024", "median_rate": 4.3, "range_low": 3.75, "range_high": 4.75},
        {"year": "2025", "median_rate": 3.1, "range_low": 2.75, "range_high": 3.75},
        {"year": "Longer run", "median_rate": 2.5, "range_low": 2.25, "range_high": 2.75}
    ]
    
    return {
        "date": latest_dot_plot_date,
        "projections": dot_plot_data,
        "summary": "Fed projections indicate a moderately hawkish stance with higher rates through 2023-2024, followed by easing in 2025 toward the longer-run neutral rate"
    }
