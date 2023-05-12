# Interview_Trading_Code
## Setup Requirements 
Install [python](https://www.python.org) 3.10 following the documentation
Open terminal and install relevant packages by following the listed commands for each project.
```
    python3 -m pip install --user --upgrade pip
    pip install requirements.txt
    
```
To run each file it will be this
```
python3 {name_of_project}.py
```
## Completed Projects

### Monte Carlo Simulation
Monte Carlo Simulation Project simulating the random weighting of portfolios of stock and selecting the optimal portfolio based on sharpe score 
increase based on historical closing prices 

• Utilized yfinance to extract historical data into Pandas Data Frame

• Improved portfolio’s initial Sharpe score by 79.45% with final score of 1.107703

• Initialize weights to uniform distribution from 0 to 1, to perform Monte Carlo simulation

```
pip install monte_carlo_requiremnts.txt
source env/bin/activate
```
### Quantitave Intrinsic Value 
Calculating  intrinsic value metrics of the whole S&P 500 and selecting the top 50 stocks to invest and display the metrics onto excel 

• Programmed API calls to iexcloud for stock quote data

• Utilized Pandas to create data frames to extrapolate the data to calculate metrics for investing strategy

• Used xlsxwriter to create an excel file that lists the top 50 stocks and their associated metrics in a table

```
pip install intrinsic_value_requirements.txt
source env/bin/activate
```
### Unoptimized Mean Recersion BTC
This project utilizes Mean reversion, zscore, dynamic stop loss, and dynamic profit target to Back Test a BTC strategy on historical OHLCV data and display the sharpe score and cumlative returns graph. 
```
pip install mean_requirements.txt
source env/bin/activate
```
### Prosperity Graph Generator
•	Developed a Python script to load, manipulate and analyze financial data from various order books using the pandas library.

•	Computed key financial indicators like mid-price, profit and loss, moving averages, Bollinger Bands, RSI, and MACD.

•	Visualized these indicators using matplotlib to create graphs for each product, including individual metrics and comparisons with other products.

•	Created a function to rank top arbitrage opportunities by calculating and comparing the spread between different products.

•	Exported the analyzed data and visualizations to an Excel file, arranging data and images in a clear and organized manner for easy interpretation.

•	Wrote utility functions for generating graph titles and y-axis labels dynamically based on the product under analysis.


This project is a graph generator that takes histroical order book data from Prosperity Trading challenge and creates graphs in jupyter notebook and in Excel to help us make investment descions. 
Some inisghts and graphs displayed:

Top 5 pairs trading oppurtunities 

1. Mean k-depth book prices
2. order book Volume
3. moving averages 
4. correlation spread graphs

```
pip install graph_requirements.txt
source env/bin/activate
```
### Final Round Prsoperity Code
This code is the final solution that I made for the final round of the IMC Trading Prosperity Coding Challenge. 
```
# Doesnt run, just code
```
## Active Projects (not completed)

### Transformer Market Making Model
This Model will produce buy, hold, and sell signals. Needs to find solution for sentiment analysis  
• Preprocessing market, data, orderbook data, and time-series-calculations for feature engineering

• Colleting and preprocessing macro-economic-data and new data for feature engineering

• Developing multi-feature transformer model to produce trading signals

• Separating features for training and validation sets and train and validate model

• Developing Bayesian Optimization to create a wrapper for my transformer model to find the best parameters

```
# just code for now ...
```
### S&P 500 Heart Beat
The final output of this project is a series of data frames, each representing the top 10 stocks for long and short positions according to a specific metric. This enables investors to make informed decisions based on various financial ratios and indicators.

•	Designed and developed a Python-based analytical tool for evaluating S&P 500 companies on various financial metrics and sentiment analysis.

•	Implemented strategies such as Moving Average Crossover, Mean Reversion, and Sentiment Analysis to generate investment signals.

•	Utilized Yahoo Finance API for real-time and historical market data collection.

•	Calculated various financial ratios such as P/E, E/P, Book-to-Price, PEG, and Debt-to-Equity to evaluate company performance.

•	Constructed a sentiment analysis model to gauge market sentiment on a rolling basis (1, 7, and 31 days) using Natural Language Processing (NLP).

•	Executed statistical tests to identify potential mean-reverting series for profitable trading opportunities.

•	Created dynamic reports displaying top 10 long and short opportunities for each evaluation metric.

•	Utilized Python libraries such as Pandas, NumPy, BeautifulSoup, and yfinance for data scraping, manipulation, and analysis.


```
# just code for now ...
```

### Sentiment Analysis 
•	Collected and preprocessed a year's worth of textual data and historical stock prices using web scraping and data cleaning techniques.

•	Conducted sentiment analysis using BERT, VADER, and GPT, and compared their performance against human input as a baseline.

•	Fine-tuned pre-trained BERT and GPT models on labeled stock-related data to improve accuracy in sentiment prediction.

•	Designed and implemented a simple trading strategy based on sentiment scores from the various analysis methods.

•	Calculated and visualized profit and loss (PnL) for long trades across each sentiment analysis method to evaluate and compare their performance in stock trading.

•	Technologies and libraries used: Python, pandas, numpy, matplotlib, scikit-learn, transformers, torch, and vaderSentiment.


```
# just code for now ...
```
