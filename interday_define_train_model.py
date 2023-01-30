import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import datetime as dt
import torch
import torch.nn as nn
import torch.nn.functional as F 
from pytickersymbols import PyTickerSymbols
import warnings

import torch
import torch_directml
dml = torch_directml.device()

warnings.filterwarnings("ignore")


class Model_interday(nn.Module):
    """The neural network model for interday trading
    it takes seven input layers:
    one for the stock time series 
    one for the embeddings of the latest news,
    one for the Sentiment of the latest news,
    one layer for the embeddings most liked tweets, 
    one layer for the Sentiment of the most liked tweets,
    one layer for the actual market data,
    it will be connected to a LSTM layer and then to a fully connected layer
    it will output a tensor with the probability of the stock being profitable in the next x timesteps
    more than 64 timesteps wont make sense, sind the layer will be to big"""
    
    def __init__(self, state_size,news_per_day, embedding_size, sentiment_size, market_size, time_horizon, hidden_size=32, num_layers=10):
        super(Model_interday, self).__init__()
        self.state_size = state_size
        #predict wich timesteps will be profitable
        self.time_horizon = time_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        #LSTM layer for the time series
        self.lstm_ts_state = nn.LSTM(input_size=state_size, hidden_size=hidden_size, num_layers=10)
        #Convolutional layer for the embeddings converting a 3D input to an 2D output
        #taking a max of 10 news per day in account, so this value is fixed
        self.emb_conv2d = nn.Conv2d(in_channels=10, out_channels=hidden_size, kernel_size=(1,3), stride=(1,1))
        self.emb_pool = nn.AdaptiveAvgPool2d((1,1))
        #Linear layer for the sentiment
        #also only one per day, so a fixed value and since the sentiment is simple, a linear layer is enough
        self.sen_state = nn.Linear(sentiment_size*10*state_size, hidden_size)
        #linear layer for the market data
        self.market_state = nn.Linear(market_size, hidden_size)
        #dropout and linear layer before the fully connected layer
        self.dropout = nn.Dropout(0.05)
        self.fc1 = nn.Linear(hidden_size, 64)
        #fully connected layer for all the inputs
        self.fc2 = nn.Linear(6*64, 128)
        self.fc3 = nn.Linear(128, 256)
        #lstm in the middle of the fully connected layer
        self.lstm_middle = nn.LSTM(input_size=256, hidden_size=128, num_layers=num_layers)
        self.fc5 = nn.Linear(128, 64)
        #last sigmoid layer // think about using a other activation function
        #think about using tanh 
        self.fc6 = nn.Linear(64, self.time_horizon)
    
    def forward(self, state_ts,state_market,state_tweets_sen, state_tweets_emb, state_news_sen, state_news_emb):
        """Forward pass through the
        state_ts (tensor) = the time series of the stockprices
        state_market (tensor) = the market data
        state_tweets_emb (tensor) = the embeddings of the tweets
        state_tweets_sen (tensor) = the sentiment of the tweets
        state_news_emb (tensor) = the embeddings of the news
        state_news_sen (tensor) = the sentiment of the news
        returns the output of the fully connected layer"""
        
        #LSTM for the time series, then linear layer and dropout
        lstm_ts_state, (h_ts, c_ts) = self.lstm_ts_state(state_ts)
        lstm_ts_state = F.relu(self.fc1(lstm_ts_state))
        lstm_ts_state = self.dropout(lstm_ts_state)
        
        #Conv for the embeddings of the tweets, then linear layer and dropout
        conv_state_tweets_emb = self.emb_conv2d(state_tweets_emb)
        conv_state_tweets_emb = self.emb_pool(conv_state_tweets_emb).view(state_tweets_emb.size(0), -1) 
        conv_state_tweets_emb = F.relu(self.fc1(conv_state_tweets_emb))
        conv_state_tweets_emb = self.dropout(conv_state_tweets_emb)

        
        #Two linear layers for the sentiment of the and dropout
        #reshape first
        state_tweets_sen = state_tweets_sen.view(state_tweets_sen.size(0), -1)
        linear_state_tweets_sen = F.relu(self.sen_state(state_tweets_sen))
        linear_state_tweets_sen = F.relu(self.fc1(linear_state_tweets_sen))
        linear_state_tweets_sen = self.dropout(linear_state_tweets_sen)
        
        #LSTM for the embeddings of the news, then linear layer and dropout
        conv_state_news_emb = self.emb_conv2d(state_tweets_emb)
        conv_state_news_emb = self.emb_pool(conv_state_news_emb).view(conv_state_news_emb.size(0), -1) 
        conv_state_news_emb = F.relu(self.fc1(conv_state_news_emb))
        conv_state_news_emb = self.dropout(conv_state_news_emb)
        
        #Two linear layers for the sentiment of the and dropout
        #reshape first
        state_news_sen = state_news_sen.view(state_news_sen.size(0), -1)
        linear_state_news_sen = F.relu(self.sen_state(state_news_sen))
        linear_state_news_sen = F.relu(self.fc1(linear_state_news_sen))
        linear_state_news_sen = self.dropout(linear_state_news_sen) 
        
        #linear layer for the market data
        market_state = self.market_state(state_market)
        market_state = F.relu(self.fc1(market_state))
        market_state = self.dropout(market_state)

        #concatenate all the layers
        #// has to be changed
        x = torch.cat((lstm_ts_state, conv_state_tweets_emb, linear_state_tweets_sen, conv_state_news_emb, linear_state_news_sen, market_state), 1) 
        
        #fully connected layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        #lstm in the middle of the fully connected layer
        x, (h, c) = self.lstm_middle(x)
        x = self.fc5(x)
        x = self.fc6(x)
        #tanh activation function
        x = torch.tanh(x)
        return x

def calculate_windows(timeseries, window_size):
    first_dim = timeseries.shape[0]
    output_tensor = np.zeros((first_dim, *timeseries.shape[1:], window_size))
    for i in range(first_dim):
        for j in range(window_size):
            if i + j >= first_dim:
                break
            output_tensor[i, ..., j] = timeseries[i + j, ...]
    output_tensor = torch.tensor(output_tensor, dtype=torch.float)
    return output_tensor

def calculate_predictions(timeseries, time_horizon):
    # Initialize the output tensor with zeros and the shape (timeseries.shape[0], time_horizon)
    output_tensor = np.zeros((timeseries.shape[0], time_horizon))
    # Iterate over the first dimension of the input tensor
    for i in range(timeseries.shape[0]):
        # Iterate over the time horizon (j = 1, 2, ..., time_horizon)
        for j in range(1, time_horizon + 1):
            # If the current index i + j is greater than the length of the first dimension of the input tensor, then break
            if i + j >= timeseries.shape[0]:
                break
            # If the element at index i + j is greater than the element at index i, then set the corresponding element in the output tensor to 1
            # otherwise set it to -1
            output_tensor[i, j-1] = 1 if timeseries[i + j] > timeseries[i] else -1
    # Convert the output tensor to a PyTorch tensor
    output_tensor = torch.tensor(output_tensor, dtype=torch.float)
    # Return the output tensor
    return output_tensor
    

def get_data_light(start_date, end_date, stock_symbol,time_horizon, state_size, batch_size):
    """returns the output in tensors of the batch size
    start_date (string) = the start date of the data
    time_horizon (int) = the time horizon of the model
    end_date (string) = the end date of the data (- time_horizon))
    stock_symbol (string) = the stock symbol
    
    to train the data withouth the news or tweets, the tensors should be filled with 0.5

    
    returns a iterable object with tuple of tensors (keeps the memory usage small):
    stock_ts (tensor) =     the time series of the stock prices
    market_data (tensor) =    the market data of the S&P 500 and the stock (volume etc.)
    predicted_stock_ts (tensor) =   for every day, the next 7 days are predicted whether the stock price 
                                    will go up (1) or down(-1)
    tensors_tweets_sentiment (tensor) = tensor consisting only of 0.1 (small bias , if its 0 then the weights would not count and would be 
                                        unpredictable)
    tensors_tweets_embeddings (tensor) =    tensor consisting only of 0.1 (small bias , if its 0 then the weights would not count and would be 
                                            unpredictable)
    tensors_articles_embeddings (tensor) =  tensor consisting only of 0.1 (small bias , if its 0 then the weights would not count and would be 
                                            unpredictable)
    tensors_articles_sentiment (tensor) =   tensor consisting only of 0.1 (small bias , if its 0 then the weights would not count and would be 
                                            unpredictable)
    -> tuple(time series, market data, predicted stock ts, tweets sentiment, tweets embeddings, articles embeddings, articles sentiment)
                          
    """
    
    #get stock prices
    stock_data = yf.download(stock_symbol, period="max", interval="1d", progress=False,show_errors=False)
    #get the time series of the stock prices
    stock_ts = list(stock_data["Adj Close"].values)
    #transform the time series into a tensor
    tensor_stock_ts = torch.tensor(stock_ts, dtype=torch.float)
    
    #get the market data of the stock
    market_data_stock = stock_data.drop(["Adj Close", "Close"], axis=1)
    #transform the market data into a tensor
    tensor_market_data_stock = torch.tensor(market_data_stock.values, dtype=torch.float)

    #get the market data
    market_data_sp500 = yf.download("^GSPC", period="max", interval="1d", progress=False,show_errors=False)
    #cut the market data to the same length as the stock data
    if len(market_data_sp500) > len(stock_data):
        market_data_sp500 = market_data_sp500[:len(stock_data)]
    if len(market_data_sp500) < len(stock_data):
        stock_data = stock_data[:len(market_data_sp500)]
    #get the time series of the market data
    market_data_sp500 = market_data_sp500.values
    #transform the time series into a tensor
    tensor_market_data_sp500 = torch.tensor(market_data_sp500, dtype=torch.float)
    
    #combine the market data of the stock and the market data of the sp500
    tensor_market_data = torch.cat((tensor_market_data_sp500, tensor_market_data_sp500), 1)
    del tensor_market_data_sp500
    del tensor_market_data_stock
    tensor_market_data.to(dml)
    #calc number days 
    global num_days
    num_days = len(stock_ts)
    #tensors_tweets_sentiment = torch.zeros((num_days, 10, 1,state_size))+0.1
    #tensors_articles_sentiment = torch.zeros((num_days, 10, 1,state_size))+0.1
    #tensors_tweets_embeddings = torch.zeros((num_days, 10, 768,state_size))+0.1
    #tensors_articles_embeddings = torch.zeros((num_days, 10, 768,state_size))+0.1
    #all to .to(dml)
    tensor_stock_ts.to(dml)
    #get the prediction tensor, and create the batches
    #for each timestep the prediction ist whether for the time_zone  timesteps the stock price will be 
    #higher or lower than the current price, and it returns a tensor of the size (num_days,time_horizon, 1)
    #it returns tuples of the batch size, so the optimal input is num_days/batch_size+time_horizon = 0
    global int_number_of_batches
    int_number_of_batches = int((num_days - time_horizon)/batch_size)
    tensor_predicted = calculate_predictions(tensor_stock_ts,time_horizon)
    #cut the tensor to the right size
    tensor_predicted = tensor_predicted[:int_number_of_batches*batch_size]
    tensor_predicted.to(dml)
    tensor_stock_ts = calculate_windows(tensor_stock_ts, state_size)
    tensor_market_data = calculate_windows(tensor_market_data, state_size)
    tensor_market_data.to(dml)
    tensor_market_data.to(dml)
    for i in range(int_number_of_batches):
        tensor_stock_ts_batch = tensor_stock_ts[i*batch_size:(i+1)*batch_size]
        tensor_market_data_batch = tensor_market_data[i*batch_size:(i+1)*batch_size]
        tensor_predicted_batch = tensor_predicted[i*batch_size:(i+1)*batch_size]
        #tensors_tweets_sentiment_batch = tensors_tweets_sentiment[i*batch_size:(i+1)*batch_size]
        #tensors_articles_sentiment_batch = tensors_articles_sentiment[i*batch_size:(i+1)*batch_size]
        #tensors_tweets_embeddings_batch = tensors_tweets_embeddings[i*batch_size:(i+1)*batch_size]
        #tensors_articles_embeddings_batch = tensors_articles_embeddings[i*batch_size:(i+1)*batch_size]
        yield tensor_stock_ts_batch, tensor_market_data_batch, tensor_predicted_batch#, tensors_tweets_sentiment_batch, tensors_tweets_embeddings_batch, tensors_articles_embeddings_batch, tensors_articles_sentiment_batch
    
    return 
      
    
global int_number_of_batches
global num_days

stock_data = PyTickerSymbols()
list_stocks = stock_data.get_all_stocks()
list_stocks = [ list_stocks[i]["symbols"] for i in range(len(list_stocks))]
list_stocks = [item for sublist in list_stocks for item in sublist]
list_stocks = [item["yahoo"] for item in list_stocks]
list_stocks = list(set(list_stocks))

int_number_of_batches = 0
time_horizon = 5
batch_size = 128
state_size = 60
model = Model_interday(state_size = state_size,news_per_day=10, embedding_size = 192, sentiment_size = 1, market_size = state_size*12, time_horizon = 5, hidden_size=32, num_layers=10)
#use PCA to reduce the dimensionality of the embeddings because they are wasting a lot of space
#save the model
torch.save(model.state_dict(), "model_lstm_embeddings_big.pt")
optimizer = optim.Adam(model.parameters())
criterion = nn.HingeEmbeddingLoss()
model = model.to(dml)
list_done = []
list_losses = []     
list_stocks = list_stocks[200:]   
for stock_symbol in list_stocks:
    try:
        print(stock_symbol)
        #load the model
        model.load_state_dict(torch.load("model_lstm_embeddings_big.pt"))
        model = model.to(dml)
        today = dt.date.today().strftime("%Y-%m-%d")
        test_df = yf.download(stock_symbol, start="2000-01-01", end=today, interval="1d", progress=False,show_errors=False)
        if len(test_df)< 100:
                del test_df
                continue
        del test_df
        start_date = "2000-01-01"
        end_date = today
        test = get_data_light(start_date, end_date, stock_symbol,time_horizon, state_size, batch_size)
        batch = test.__next__()
        tensor_stock_ts = batch[0].to(dml)
        market_data = batch[1].to(dml)
        market_data = market_data.reshape((batch_size, state_size*market_data.shape[1]))
        labels = batch[2].to(dml)
        tensors_tweets_sentiment = torch.zeros((batch_size, 10, 1,state_size)).to(dml)+0.1
        tensors_tweets_embeddings = torch.zeros((batch_size, 10, 192,state_size)).to(dml) + 0.1
        tensors_articles_sentiment = torch.zeros((batch_size, 10, 1,state_size)).to(dml)+0.1
        tensors_articles_embeddings = torch.zeros((batch_size, 10, 192,state_size)).to(dml) + 0.1
        optimizer.zero_grad()
        outputs = model(state_ts = tensor_stock_ts, state_market = market_data, state_tweets_sen = tensors_tweets_sentiment, state_tweets_emb = tensors_tweets_embeddings, state_news_sen = tensors_articles_sentiment, state_news_emb = tensors_articles_embeddings)
        outputs.to(dml)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        for epoch in range(int_number_of_batches-1):
                batch = test.__next__()
                tensor_stock_ts = batch[0].to(dml)
                market_data = batch[1].to(dml)
                market_data = market_data.reshape((batch_size, state_size*market_data.shape[1]))
                predicted_stock_ts = batch[2].to(dml)
                labels = predicted_stock_ts.to(dml)
                optimizer.zero_grad()
                outputs = model(state_ts = tensor_stock_ts, state_market = market_data, state_tweets_sen = tensors_tweets_sentiment, state_tweets_emb = tensors_tweets_embeddings, state_news_sen = tensors_articles_sentiment, state_news_emb = tensors_articles_embeddings)
                outputs.to(dml)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                del batch
        #save the model
        torch.save(model.state_dict(), "model_lstm_embeddings_big.pt")
        #save the stocks the model has been trained in a python
        list_done.append(stock_symbol)
        with open("list_done.py", "w") as f:
                f.write(str(list_done))
        #save the losses
        list_losses.append(loss.item())
        with open("list_losses.py", "w") as f:
                f.write(str(list_losses))
        del test
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
        continue
