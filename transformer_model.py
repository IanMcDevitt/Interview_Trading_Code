import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MinMaxScaler
import requests
import datetime
from functools import partial
from bayes_opt import BayesianOptimization

import websocket
import json
import pandas as pd
import numpy as np
import datetime
import requests
from fredapi import Fred
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MinMaxScaler

# Fetch news data using News API
def fetch_news_data(symbol, api_key, days=30):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    url = f"https://newsapi.org/v2/everything?q={symbol}&from={start_date}&to={end_date}&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    return news_data

news_api_key = "YOUR_NEWS_API_KEY"
news_data = fetch_news_data("AAPL", news_api_key)

# Extract textual features using BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_textual_features(news_data):
    features = []
    for article in news_data["articles"]:
        text = article["title"] + " " + article["description"]
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]
        features.append(cls_token.detach().numpy())
    return np.vstack(features)

news_features = extract_textual_features(news_data)

# Preprocessing and aligning data
def align_data(market_data_df, macro_data_df, news_features_df):
    market_data_df['timestamp'] = pd.to_datetime(market_data_df['timestamp']).dt.tz_convert('UTC')
    market_data_df.set_index('timestamp', inplace=True)
    
    macro_data_df['timestamp'] = pd.to_datetime(macro_data_df['timestamp']).dt.tz_convert('UTC')
    macro_data_df.set_index('timestamp', inplace=True)
    
    news_features_df['timestamp'] = pd.to_datetime(news_features_df['timestamp']).dt.tz_convert('UTC')
    news_features_df.set_index('timestamp', inplace=True)

    market_data_resampled = market_data_df.resample('1T').ffill()
    macro_data_resampled = macro_data_df.resample('1T').ffill()
    news_features_resampled = news_features_df.resample('1T').ffill()

    merged_data = pd.concat([market_data_resampled, macro_data_resampled, news_features_resampled], axis=1)
    merged_data.sort_index(inplace=True)

    return merged_data

# Replace with your actual data
market_data_df = ...
macro_data_df = ...
news_features_df = ...

aligned_data = align_data(market_data_df, macro_data_df, news_features_df)

# Prepare data for the Transformer model
class MarketDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_data_loaders(train_features, train_labels, val_features, val_labels, batch_size):
    train_dataset = MarketDataset(train_features, train_labels)
    val_dataset = MarketDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Transformer Model Architecture
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

# Training and Evaluation Functions
def setup_training_components(learning_rate, weight_decay):
    model = TransformerModel(...)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluate on the validation set
        val_loss, val_accuracy = evaluate_model(model, val_loader)

def evaluate_model(model, val_loader):
    model.eval()
    loss_sum = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

    val_loss = loss_sum / len(val_loader)
    val_accuracy = correct_predictions / total_predictions

    return val_loss, val_accuracy

# Hyperparameter Optimization
def objective_func(params, train_features, train_labels, val_features, val_labels):
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])

    train_loader, val_loader = create_data_loaders(train_features, train_labels, val_features, val_labels, batch_size)
    model, optimizer, criterion = setup_training_components(learning_rate, weight_decay)
    train_model(model, train_loader, val_loader, optimizer, criterion, epochs)
    val_loss, val_accuracy = evaluate_model(model, val_loader)

    return val_accuracy

# Replace with your actual preprocessed data
train_features = ...
train_labels = ...
val_features = ...
val_labels = ...

# Run Bayesian Optimization
bounds = {
    'learning_rate': (1e-4, 1e-2),
    'weight_decay': (1e-5, 1e-2),
    'batch_size': (32, 256),
    'epochs': (10, 100),
}

optimizer = BayesianOptimization(
    f=partial(objective_func, train_features=train_features, train_labels=train_labels, val_features=val_features, val_labels=val_labels),
    pbounds=bounds,
    random_state=42,
)

optimizer.maximize(init_points=5, n_iter=25)



def authenticate_and_subscribe(ws, channels):
    auth = {"action": "auth", "params": "YOUR_POLYGON_API_KEY"}
    ws.send(json.dumps(auth))

    for channel in channels:
        ws.send(json.dumps({"action": "subscribe", "params": f"AM.{channel}"}))
        ws.send(json.dumps({"action": "subscribe", "params": f"A.{channel}"}))

def fetch_macro_data(api_key):
    fred = Fred(api_key=api_key)

    # Collect macro-economic data
    gdp = fred.get_series("GDPC1")  # Real Gross Domestic Product
    inflation = fred.get_series("CPIAUCSL")  # Consumer Price Index for All Urban Consumers: All Items
    unemployment = fred.get_series("UNRATE")  # Unemployment Rate
    interest_rate = fred.get_series("FEDFUNDS")  # Effective Federal Funds Rate

    # Combine the macro-economic data into a single DataFrame
    macro_data = pd.concat([gdp, inflation, unemployment, interest_rate], axis=1)
    macro_data.columns = ["GDP", "Inflation", "Unemployment", "Interest Rate"]

    return macro_data

def main():
    channels = ["T.AAPL", "T.GOOG", "T.MSFT", "T.AMZN", "T.TSLA"]

    # Set up WebSocket connection to polygon.io
    ws = websocket.WebSocketApp("wss://socket.polygon.io/stocks",
                                on_message=on_message)

    # Authenticate and subscribe to relevant channels
    ws.on_open = lambda ws: authenticate_and_subscribe(ws, channels)

    # Start WebSocket connection
    ws.run_forever()

    # Fetch macro-economic data
    macro_data_df = fetch_macro_data(fred_api_key)

    # Fetch news data for all symbols
    news_data = {}
    for symbol in channels:
        news_data[symbol] = fetch_news_data(symbol, news_api_key)

    # Extract textual features from news data
    news_features = {}
    for symbol, data in news_data.items():
        news_features[symbol] = extract_textual_features(data)

    # Create a DataFrame for news features
    news_features_df = pd.DataFrame(news_features)

    # Data cleaning: Fill missing values using interpolation
    market_data_df.interpolate(method='linear', inplace=True)
    macro_data_df.interpolate(method='linear', inplace=True)
    order_book_df.interpolate(method='linear', inplace=True)

    # Data resampling: Resample market data, order book data, and macro-economic data to daily frequency
    daily_market_data_df = market_data_df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    order_book_df = order_book_df.resample('D').agg({
        'bid_price': 'mean',
        'bid_size': 'mean',
        'ask_price': 'mean',
        'ask_size': 'mean',
    })
    macro_data_df = macro_data_df.resample('D').fillna(method='ffill')

    # Data normalization: Normalize market data, order book data, and macro-economic data
    scaler = MinMaxScaler()
    normalized_market_data = scaler.fit_transform(daily_market_data_df)
    normalized_order_book_data = scaler.fit_transform(order_book_df)
    normalized_macro_data = scaler.fit_transform(macro_data_df)

    # Convert normalized data back to DataFrames
    normalized_market_data_df = pd.DataFrame(normalized_market_data, columns=daily_market_data_df.columns, index=daily_market_data_df.index)
    normalized_order_book_df = pd.DataFrame(normalized_order_book_data, columns=order_book_df.columns, index=order_book_df.index)
    normalized_macro_data_df = pd.DataFrame(normalized_macro_data, columns=macro_data_df.columns, index=macro_data_df.index)

    # Data alignment: Merge market data, order book data, and macro-economic data on the common index (timestamp)
    merged_data = pd.merge(normalized_market_data_df, normalized_order_book_df, left_index=True, right_index=True, how='inner')
    merged_data = pd.merge(merged_data, normalized_macro_data_df, left_index=True, right_index=True, how='inner')


# Combine time-series features, news features, and aligned data
final_features_df = pd.concat([time_series_features_df, news_features_df, merged_data], axis=1)

if __name__ == '__main__':
    main()
