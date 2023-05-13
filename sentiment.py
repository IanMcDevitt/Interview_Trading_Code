import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
import yfinance as yf

# ---- Data Collection ----
# Specify the stock and date range
stock = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'

# Download historical stock prices
stock_prices = yf.download(stock, start=start_date, end=end_date)

# Scrape news data and merge with stock prices
news_df = ...  # Scrape news data (code not included)


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL of the webpage you want to scrape
url = 'https://www.example.com/news'

# Send a GET request to the webpage
response = requests.get(url)

# Parse the webpage's content
soup = BeautifulSoup(response.content, 'html.parser')

# Find the news articles on the webpage
articles = soup.find_all('div', class_='article')

# Define lists to store the data
titles = []
dates = []
contents = []

# Extract the data from the articles
for article in articles:
    title = article.find('h2').text
    date = article.find('span', class_='date').text
    content = article.find('p').text

    titles.append(title)
    dates.append(date)
    contents.append(content)

# Create a DataFrame with the data
news_df = pd.DataFrame({
    'title': titles,
    'date': dates,
    'content': contents
})

# Convert the 'date' column to datetime format
news_df['date'] = pd.to_datetime(news_df['date'])

merged_df = pd.merge(stock_prices, news_df, on='timestamp', how='inner')

# ---- Preprocessing ----
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    words = text.split()
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a single string
    text = ' '.join(words)
    
    return text

# Apply the preprocessing to the 'text' column
merged_df['preprocessed_text'] = merged_df['text'].apply(preprocess_text)



# ---- Sentiment Analysis (VADER example) ----
analyzer = SentimentIntensityAnalyzer()
merged_df['vader_sentiment_scores'] = merged_df['preprocessed_text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

# ---- Trading Strategy ----
def simple_trading_strategy(sentiment_scores, positive_threshold, negative_threshold):
    position = 0
    entry_price = 0
    pnl = 0

    trading_results = []

    for index, row in sentiment_scores.iterrows():
        sentiment_score = row['vader_sentiment_scores']
        stock_price = row['Close']  # Adjust column name if necessary

        if position == 0 and sentiment_score >= positive_threshold:
            position = 1
            entry_price = stock_price
        elif position == 1 and sentiment_score <= negative_threshold:
            pnl += stock_price - entry_price
            position = 0

        trading_results.append({'timestamp': row['timestamp'], 'position': position, 'pnl': pnl})

    return pd.DataFrame(trading_results)

from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch

# ---- Preparing the BERT model ----
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Suppose you have a dataset with the columns 'preprocessed_text' and 'label'
class StockNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.text = data.preprocessed_text
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Create a data loader
data = ...  # Load your labeled stock news data
max_len = 128  # Define the maximum length for the sequences
dataset = StockNewsDataset(data, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define a loss function and an optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=0.001)

# Fine-tune the BERT model
for epoch in range(10):  # Number of epochs
    model.train()
    for i, batch in enumerate(dataloader):
        ids = batch['ids']
        mask = batch['mask']
        targets = batch['targets']

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


import openai
import pandas as pd

# Set the OpenAI API key
openai.api_key = 'your-openai-api-key'

# Define a function to get the sentiment of a text using GPT-3
def get_sentiment(text):
    # Prompt GPT-3 to rate the sentiment of the text
    prompt = f"The following is a piece of news text. Please rate its sentiment on a scale from -1 (very negative) to 1 (very positive):\n\n\"{text}\"\n\nSentiment rating:"
    
    # Generate a response from GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.1,
        max_tokens=10
    )

    # Extract the sentiment rating from the response
    sentiment = float(response.choices[0].text.strip())

    return sentiment

# Get the sentiment of each news article
merged_df['gpt_sentiment_scores'] = merged_df['text'].apply(get_sentiment)





# Apply the trading strategy
positive_threshold = 0.5
negative_threshold = -0.5
trading_results = simple_trading_strategy(merged_df, positive_threshold, negative_threshold)

# ---- Visualization ----
plt.figure(figsize=(12, 6))
plt.plot(trading_results['timestamp'], trading_results['pnl'], label="VADER")
plt.xlabel('Time')
plt.ylabel('PnL')
plt.legend()
plt.title('PnL for Long Trades - Sentiment Analysis with VADER')
plt.show()
