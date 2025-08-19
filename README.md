# That's My Quant

## Overview

This project collects and aggregates financial market data for AI/LLM-powered financial analysis. It transforms raw time series data into concise, structured daily reports optimized for large language model consumption.

Instead of ingesting millions of data points from raw time series, this tool creates a daily snapshot of key market indicators, trends, and relationships in a JSON format that's easy for LLMs to process and reason about as if they needed to perform at the top levels of quant analysis.

## Key Features

- **Macroeconomic Data**: Collects and summarizes key indicators from FRED (Federal Reserve Economic Data)
- **Market Data**: Aggregates pricing and performance data for major indices, sectors, and individual stocks
- **Technical Indicators**: Calculates and summarizes key technical indicators (MA, RSI, MACD)
- **Correlation Analysis**: Identifies relationships between different market assets
- **On-Chain Metrics**: Retrieves and analyzes blockchain data for cryptocurrencies
- **Focus Areas**: Special emphasis on tech stocks (MAG 7), AI-related companies, and cryptocurrencies
- **LLM Optimization**: All data is transformed into a concise, structured format ideal for language model processing

## Data Structure

The daily report JSON file contains the following sections:

1. **Key Insights**: Concise text summaries of the most important market trends and indicators
2. **Macro Data**: Economic indicators like GDP, inflation, interest rates, unemployment
3. **Market Data**: Performance metrics for individual stocks and indices
4. **Correlation Matrix**: Showing relationships between different market assets
5. **On-Chain Data**: Blockchain metrics for cryptocurrencies

## Example Insights

The report generates insights like:
- Overall market sentiment (bullish, bearish, neutral)
- Tech sector performance with weekly/monthly/yearly trends
- MAG 7 (major tech companies) average returns and best/worst performers
- Current inflation and interest rate trends
- Cryptocurrency price and performance metrics
- Blockchain-specific metrics like active addresses and exchange flows

## Usage

1. **Setup**: Ensure you have the required dependencies:
   ```
   pip install fredapi pandas yfinance numpy requests
   ```

2. **API Key**: Replace the FRED API key in the script with your own key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)

3. **Run Daily**: Execute the script to generate the latest market report:
   ```
   python run.py
   ```

4. **LLM Integration**: Import the generated JSON file in your LLM pipeline for financial analysis

## Data Sources

- **FRED**: Federal Reserve Economic Data for macroeconomic indicators
- **Yahoo Finance**: Market data for stocks, indices, ETFs, and cryptocurrencies
- **Blockchair**: On-chain metrics for Bitcoin and Ethereum

## Report Structure

The daily JSON report is organized as follows:

```json
{
  "report_date": "YYYY-MM-DD",
  "summary": "Daily Financial Market Report",
  "key_insights": [
    "Market Sentiment: Cautiously Bearish",
    "Tech Sector: Downtrend (W: 1.08%, M: -8.89%, Y: 30.75%)",
    "MAG 7 Average Monthly Return: -14.64%",
    "BTC Exchange Flow: -1205.32 (Net outflow from exchanges (potentially bullish))",
    "ETH Active Addresses: 547,261",
    // Additional insights...
  ],
  "macro_data": [
    // Economic indicators with trends and changes
  ],
  "market_data": [
    // Individual asset summaries with technical indicators
  ],
  "correlations": {
    // Correlation matrix between assets
  },
  "onchain_data": [
    // Blockchain metrics like active addresses, transaction counts, fees, etc.
  ]
}
```

## On-Chain Metrics

For cryptocurrencies, the report includes:

- **Active Addresses**: Number of unique active addresses in the last 24h
- **Transaction Count**: Number of on-chain transactions in the last 24h 
- **Average Transaction Fees**: Transaction costs in USD
- **Exchange Flow**: Net movement of assets to/from exchanges (indicator of potential selling/buying pressure)
- **Large Transactions**: Number of significant value transfers (whale activity)

## For LLM Analysis

When analyzing this data as an LLM:

1. Start with the `key_insights` for a high-level overview
2. Examine `macro_data` to understand the economic environment
3. Review `market_data` for specific assets of interest
4. Use the `correlations` to identify relationships between assets
5. Analyze `onchain_data` for crypto-specific trends
6. Look for divergences between related metrics
7. Pay attention to trend changes in both macro indicators and market data

## Customization

The script can be customized to:
- Add or remove specific economic indicators
- Modify the list of tracked stocks and assets
- Change the lookback periods for performance calculations
- Adjust technical indicator parameters
- Add additional on-chain metrics or cryptocurrencies

## Use Cases

This data is particularly valuable for:
- Market trend analysis
- Long-term investment planning
- Avoiding major market drawdowns
- Technical trading strategy development
- Economic cycle analysis
- Sector rotation strategy
- Cryptocurrency trading signals 