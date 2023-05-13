import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import datetime
import re

def get_sp500_symbols():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    symbols = []
    for row in table.findAll('tr')[1:]:
        symbol = row.findAll('td')[0].text.strip()
        symbols.append(symbol)
    return symbols

def get_stock_data(symbol, start, end):
    stock = yf.Ticker(symbol)
    return stock.history(start=start, end=end)

def get_stock_financials(symbol):
    stock = yf.Ticker(symbol)
    return stock.financials, stock.quarterly_financials, stock.balancesheet, stock.quarterly_balancesheet

def get_stock_earnings(symbol):
    stock = yf.Ticker(symbol)
    return stock.earnings, stock.quarterly_earnings

def get_stock_info(symbol):
    stock = yf.Ticker(symbol)
    return stock.info

def moving_average_cross_strategy(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
    signals['positions'] = signals['signal'].diff()
    return signals

def mean_reversion_test(data, confidence_level=0.05):
    test_result = adfuller(data['Close'], autolag='AIC')
    p_value = test_result[1]
    return p_value < confidence_level

def calculate_ratios(financials, stock_info):
    pe_ratio = stock_info['trailingPE']
    ep_ratio = 1 / pe_ratio if pe_ratio else None
    book_value = financials.loc['Total Stockholder Equity'][0]
    market_cap = stock_info['marketCap']
    book_to_price_ratio = book_value / market_cap if market_cap else None
    peg_ratio = stock_info['pegRatio']
    return pe_ratio, ep_ratio, book_to_price_ratio, peg_ratio

def calculate_debt_equity_ratio(balance_sheet):
    total_debt = balance_sheet.loc['Total Liab'][0]
    total_equity = balance_sheet.loc['Total Stockholder Equity'][0]
    return total_debt / total_equity

def calculate_earnings_quality(earnings):
    earnings_diff = np.diff(earnings['Earnings'])
    earnings_diff_pct = np.abs(earnings_diff) / earnings['Earnings'][:-1]
    return np.mean(earnings_diff_pct)

def calculate_sentiment(symbol, rolling_days=[1, 7, 31]):
    url = f'https://finance.yahoo.com/quote/{symbol}/community?p={symbol}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    messages = soup.find_all('div', {'class': 'C($c-fuji-grey-l) Mb(2px) Fz(14px) Lh(20px) Pend(8px)'})
    sentiment_scores = []
    for message in messages:
        analysis = TextBlob(re.sub(r'\W+', ' ', message.text))
        sentiment_scores.append(analysis.sentiment.polarity)
    sentiment_moving_averages = {}
    for days in rolling_days:
        sentiment_moving_averages[days] = pd.Series(sentiment_scores).ewm(span=days).mean().iloc[-1]
    return sentiment_moving_averages

start_date = '2022-01-01'
end_date = '2023-01-01'
symbols = get_sp500_symbols()
results = []

# for symbol in symbols:
#     try:
#         data = get_stock_data(symbol, start_date, end_date)
#         financials, _, balance_sheet, _ = get_stock_financials(symbol)
#         earnings, _ = get_stock_earnings(symbol)
#         stock_info = get_stock_info(symbol)
        
#         mac_signals = moving_average_cross_strategy(data, 60, 200)
#         mean_reversion_result = mean_reversion_test(data)
#         pe_ratio, ep_ratio, book_to_price_ratio, peg_ratio = calculate_ratios(financials, stock_info)
#         debt_equity_ratio = calculate_debt_equity_ratio(balance_sheet)
#         earnings_quality = calculate_earnings_quality(earnings)
#         sentiment_scores = calculate_sentiment(symbol)

#         results.append({
#             'symbol': symbol,
#             'moving_average_cross_signals': mac_signals,
#             'mean_reversion': mean_reversion_result,
#             'PE_ratio': pe_ratio,
#             'EP_ratio': ep_ratio,
#             'Book_to_Price_Ratio': book_to_price_ratio,
#             'PEG_ratio': peg_ratio,
#             'Debt_to_Equity_Ratio': debt_equity_ratio,
#             'Earnings_Quality': earnings_quality,
#             'Sentiment_1_day': sentiment_scores[1],
#             'Sentiment_7_days': sentiment_scores[7],
#             'Sentiment_31_days': sentiment_scores[31]
#         })
#         print(f"Processed {symbol}")
#     except Exception as e:
#         print(f"Error processing {symbol}: {e}")

# Continue from previous code
# Define the metrics
metrics = ['moving_average_cross_signals', 'mean_reversion', 'PE_ratio', 'EP_ratio', 'Book_to_Price_Ratio', 'PEG_ratio', 
           'Sentiment_1_day', 'Sentiment_7_days', 'Sentiment_31_days', 'Debt_to_Equity_Ratio', 'Earnings_Quality']

long_positions_dict = {}
short_positions_dict = {}

for metric in metrics:
    if metric in ['PE_ratio', 'PEG_ratio', 'Debt_to_Equity_Ratio']:
        # For these metrics, lower values are better for long positions, and higher values are better for short positions
        sorted_df = result_df.sort_values(by=metric, ascending=True)
        long_positions = sorted_df.head(10)
        short_positions = sorted_df.tail(10)
    elif metric in ['EP_ratio', 'Book_to_Price_Ratio', 'Earnings_Quality']:
        # For these metrics, higher values are better for long positions, and lower values are better for short positions
        sorted_df = result_df.sort_values(by=metric, ascending=False)
        long_positions = sorted_df.head(10)
        short_positions = sorted_df.tail(10)
    else: # 'Sentiment_1_day', 'Sentiment_7_days', 'Sentiment_31_days'
        # For sentiment analysis, positive values indicate bullish (long) sentiment and negative values indicate bearish (short) sentiment
        sorted_df_long = result_df.sort_values(by=metric, ascending=False)
        sorted_df_short = result_df.sort_values(by=metric, ascending=True)
        long_positions = sorted_df_long.head(10)
        short_positions = sorted_df_short.head(10)

    long_positions_dict[metric] = long_positions
    short_positions_dict[metric] = short_positions

for metric in metrics:
    print(f"Metric: {metric}")

    print("\nTop 10 Long Positions:")
    display(long_positions_dict[metric])

    print("\nTop 10 Short Positions:")
    display(short_positions_dict[metric])
    
    print("\n---------------------------\n")


result_df = pd.DataFrame(results)
