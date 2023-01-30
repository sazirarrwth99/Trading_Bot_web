import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime as dt
from pytickersymbols import PyTickerSymbols
import time
import torch
import torch_directml
dml = torch_directml.device()

def calculate_windows(timeseries, window_size):
    first_dim = timeseries.shape[0]
    #output_tensor = np.zeros((first_dim, *timeseries.shape[1:], window_size)) do iz in torch
    output_tensor = torch.zeros((first_dim, *timeseries.shape[1:], window_size),device=dml)
    for i in range(first_dim):
        for j in range(window_size):
            if i + j >= first_dim:
                break
            output_tensor[i, ..., j] = timeseries[i + j, ...]
    output_tensor = torch.tensor(output_tensor, dtype=torch.float)
    return output_tensor

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


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Model_Portefolio(nn.Module):
    """The neural network for the portefolio,
    it gets outputs of the NNs of the stocks, and it gets some general market data 
    the portefolio consist of X stocks, the S&P500 and a risk free asset with
    a fixed yearly return of 
    
    gets: 
    for initialisation:
        portefolio_size (int) = size of the portefolio
        
        market_data_size (int) = size of the market data
        time_horizon (int) = time horizon of the predictions
        hidden_size (int) = size of the hidden layer
    for forward:
        predicictios of the stocks (tensor, shape: (X,time horizon)) = stock price predictions )
        market_data (tensor, shape:(market_data_size,time_horizon)) = S&P500, general market data
        old portefolio (tensor) = old portefolio
    return:
        new portefolio (tensor) = new portefolio
        
    attention: for good performance, the hidden size should be bigger then the portefolio size
    """
    
    #buy dax as index or dont invest

    def __init__(self,portefolio_size, market_data_size, time_horizon, hidden_size, num_layers):
        super(Model_Portefolio, self).__init__()
        self.market_data_size = market_data_size
        self.portefolio_size = portefolio_size
        self.time_horizon = time_horizon
        self.hidden_size = hidden_size
        
        #first linear layer for the stocks
        self.linear_layer_stocks = nn.Linear(self.time_horizon, self.time_horizon*5)
        #Convolutional layer for the stocks, compress the time predictions to one value for each stock
        self.conv_layer = nn.Conv1d(in_channels=self.time_horizon*5, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        #linear layer to set to the hidden size from portefolio size
        self.linear_layer_stocks_portefolio = nn.Linear(self.portefolio_size, self.hidden_size)
        #linear layer to set to the hidden size from market data size
        self.linear_layer_market_portefolio = nn.Linear(self.market_data_size, self.hidden_size)
        #add a new layer 
        self.fc1 = nn.Linear(self.hidden_size*3, self.hidden_size)
        #lstm layer
        self.lstm = nn.LSTM(self.hidden_size, hidden_size*2, num_layers)
        #linear layer to set to the hidden size from market data size
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        #dropout layer
        self.dropout = nn.Dropout(0.1)
        #final portefolio layer
        self.fc3 = nn.Linear(hidden_size, self.portefolio_size)
        #layers to minize the changes
        self.fc4 = nn.Linear(self.portefolio_size, self.portefolio_size*3)
        self.fc5 = nn.Linear(self.portefolio_size*3, self.portefolio_size)
        self.fc6 = nn.Linear(self.portefolio_size, self.portefolio_size*2)
        #output layer
        self.fc7 = nn.Linear(self.portefolio_size*2, self.portefolio_size)
        
        
    def forward(self, tensor_stock_predictions, tensor_old_portefolio,tensor_market_data):
        
        #first linear layer for the stocks 
        tensor_stock_predictions = self.linear_layer_stocks(tensor_stock_predictions)
        #transpose the tensor to fit the convolutional layer
        tensor_stock_predictions = tensor_stock_predictions.transpose(0,1)
        #Convolutional layer for the stocks, compress the time predictions to one value for each stock
        tensor_stock_predictions = self.conv_layer(tensor_stock_predictions)
        tensor_stock_predictions = self.max_pool(tensor_stock_predictions)
        tensor_stock_predictions = tensor_stock_predictions.reshape((self.portefolio_size))
        #linear layer to set to the hidden size
        tensor_stock_predictions = F.relu(self.linear_layer_stocks_portefolio(tensor_stock_predictions))
        #layer for the old portefolio
        tensor_portefolio = F.relu(self.linear_layer_stocks_portefolio(tensor_old_portefolio))
        #layer for the market data
        tensor_market_data = F.relu(self.linear_layer_market_portefolio(tensor_market_data))
        #bring the tensors together
        x = torch.cat((tensor_stock_predictions, tensor_portefolio, tensor_market_data), 0)
        #add a new layer
        #dropout layer
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        #lstm layer
        x = x.view(1, 1, -1)
        x, (h_n, c_n) = self.lstm(x)
        x = x.view(-1, 64).reshape(64)
        #add a new layer
        x = F.relu(self.fc2(x))
        #output layer
        x = self.fc3(x)
        #calculate the changes
        x = x - tensor_old_portefolio
        #give input to the layers to minize the changes
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        #add the old portefolio again
        x = x + tensor_old_portefolio
        #give input to the layers to minize the changes
        x = F.relu(self.fc6(x))
        #output layer
        x = self.fc7(x)
        #softmax layer
        return F.softmax(x, dim=0)  


class Agent:
    """ Stock Trading Bot """
    
    def __init__(self, state_size,portefolio_size,market_data_size,time_horizon,hidden_size, num_layers,news_per_day = 10,embedding_size = 192 ,stock_model_name=None,model_name=None):
    
        #run on GPU
        self.dml = torch_directml.device()
        # agent attributes
        self.portefolio_size = portefolio_size  
        self.market_data_size = market_data_size
        self.time_horizon = time_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_horizon = time_horizon
        
        self.state_size = state_size
        self.time_horizon = time_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.news_per_day = news_per_day

        
        self.cost_per_trade = 0.001
        #Bux average trade fee 0.15%
        self.trade_fee = 0.0015
        self.amount = 100000
        self.risk_penalty = 0.0001
        #10% Dax return
        self.not_invested_penalty = 0.0002
        self.loose_money_penalty = 0.0001
        #2 % interest rate
        self.risk_free_return = 0.00005
        stock_data = PyTickerSymbols()   
        list_stocks = stock_data.get_all_stocks()
        list_stocks = [ list_stocks[i]["symbols"] for i in range(len(list_stocks))]
        list_stocks = [item for sublist in list_stocks for item in sublist]
        list_stocks = [item["yahoo"] for item in list_stocks]
        self.list_stocks = list(set(list_stocks))
        self.cost_per_trade = self.cost_per_trade/self.amount
        # generate model
        self.model_name = model_name
        #load the RL model if it exists
        if self.model_name is not None:
            try:
                self.model = torch.load(self.model_name) 
            except:
                self.model = Model_Portefolio(portefolio_size = self.portefolio_size, market_data_size = self.market_data_size, time_horizon = self.time_horizon, hidden_size=self.hidden_size, num_layers=self.num_layers)
                #save the model
                torch.save(self.model, self.model_name)
        else:
            self.model = Model_Portefolio(portefolio_size = self.portefolio_size, market_data_size = self.market_data_size, time_horizon = self.time_horizon, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.model.to(self.dml)
        self.optimizer = optim.Adam(self.model.parameters())
        #get stock model
        self.stock_model_name = stock_model_name
        if self.stock_model_name is not None:
            self.stock_model = Model_interday(state_size = self.state_size,news_per_day=self.news_per_day, embedding_size = self.embedding_size, sentiment_size = 1, market_size = state_size*self.market_data_size, time_horizon = self.time_horizon, hidden_size=32, num_layers=10)
            #load dict of parameters
            self.stock_model.load_state_dict(torch.load(self.stock_model_name))
        else:
            raise Exception("No stock model found")
        self.stock_model.to(self.dml)
    def get_status(self):
        """Get the status of the agent (epsilon, etc.)
        Explanation of the status:
        - loss: loss function
        - optimizer: optimizer
        - model: model
        """
        return {
            "optimizer": self.optimizer,
            "model": self.model,
            "stock model": self.stock_model,

        }

    def get_action_probs(self, state, is_eval=False):
            """Take action from given possible set of actions
            in this case the portefolio is returned
            """
            # Get the policy for the current state
            return self.model(state[0],state[1],state[2])
        
    def update_stock_model(self):
            self.stock_model = Model_interday(state_size = self.state_size,news_per_day=self.news_per_day, embedding_size = self.embedding_size, sentiment_size = 1, market_size = state_size*self.market_data_size, time_horizon = self.time_horizon, hidden_size=32, num_layers=10)
            #load dict of parameters
            self.stock_model.load_state_dict(torch.load(self.stock_model_name))
            self.stock_model.to(self.dml)
 
        
    def create_batches(self,stocks = None,batch_size= 64):
        if stocks == None:
            #take portefolio_size random stocks of the stocks list
            stocks = random.sample(self.list_stocks, self.portefolio_size*2)
            #get the data for the stocks, get a random date 
            today = dt.date.today()
            start_date = random.choice([dt.date(random.randint(2015, today.year), random.randint(1, 12), random.randint(1, 28)) for i in range(1000)])
            #plus 6 months
            end_date  = start_date + dt.timedelta(days=360)
            start_date = start_date.strftime("%Y-%m-%d")    
            end_date = end_date.strftime("%Y-%m-%d")
            stocks_data = yf.download(stocks, start = start_date, end = end_date,interval="1d", progress=False,show_errors=False)
            stocks_data = stocks_data.dropna(how="all", axis=1)
            stocks_data = stocks_data.fillna(method="ffill")
            stocks_data = stocks_data["Adj Close"].values
            if stocks_data.shape[0] < 110:
                print("not enough rows")
                return None
            if stocks_data.shape[1] < self.portefolio_size:
                print("not enough columns")
                return None
            #cut to the columns that have more than 110 rows
            stocks_data = stocks_data[:,np.sum(np.isnan(stocks_data),axis=0) < 110]
            #cut to the first portefolio_size-1 columns
            stocks_data = stocks_data[:,:self.portefolio_size-1]
            market_data_sp500 = yf.download("^GSPC", start = start_date, end = end_date, interval="1d", progress=False,show_errors=False).values
            market_data_sp500 = torch.tensor(market_data_sp500,device=self.dml)
            market_data_sp500
            ts_data = torch.tensor(stocks_data,device=self.dml)
            #iterate from the 90th row to 20th last row
            #iterate a random portefolio_size 
            #market_data_sp500 = torch.cat((market_data_sp500, market_data_sp500), 1)
            ts_data = calculate_windows(ts_data, 60)
            market_data_sp500 = calculate_windows(market_data_sp500, 60)
            tensors_tweets_sentiment = torch.zeros((self.portefolio_size-1, 10, 1,self.state_size),device=self.dml)
            tensors_tweets_embeddings = torch.zeros((self.portefolio_size-1, 10, 192,self.state_size),device=self.dml)
            tensors_articles_sentiment = torch.zeros((self.portefolio_size-1, 10, 1,self.state_size),device=self.dml)
            tensors_articles_embeddings = torch.zeros((self.portefolio_size-1, 10, 192,self.state_size),device=self.dml)
            #initialize a random portefolio
            old_portefolio = torch.rand((self.portefolio_size))
            old_portefolio.to(self.dml)
            old_portefolio = old_portefolio/torch.sum(old_portefolio)
            batches_return = []
            #get_action_probs
            for i in range(90,len(stocks_data)-20):
                ts_batch = ts_data[i:i+1].reshape(self.portefolio_size-1,self.state_size)
                market_batch = market_data_sp500[i:i+1].reshape(self.state_size*market_data_sp500.shape[1])
                #make the market batch the same tensor but X times 
                market_batch_repeat = market_batch.repeat(self.portefolio_size-1, 1)
                #cat the tensor with itself
                market_batch_repeat = torch.cat((market_batch_repeat, market_batch_repeat), 1)
                print("1")
                ts_batch.to(self.dml)
                market_batch.to(self.dml)
                market_batch_repeat.to(self.dml)
                predictions = self.stock_model(state_ts=ts_batch,state_market = market_batch_repeat,state_tweets_sen=tensors_tweets_sentiment, state_tweets_emb = tensors_tweets_embeddings, state_news_sen=tensors_articles_sentiment, state_news_emb=tensors_articles_embeddings)
                new_row = torch.zeros((1, 5),device=self.dml)
                predictions = torch.cat((new_row, predictions), dim=0)  
                #get the old stock prices, the ith row 
                print("9")
                old_stock_prices = stocks_data[i]
                old_stock_prices = torch.tensor(old_stock_prices)
                old_stock_prices.to(self.dml)
                #append the first element as 1 as the risk free asset
                old_stock_prices = torch.cat((torch.tensor([1]),old_stock_prices))
                old_stock_prices.to(self.dml)
                print("10")
                #get the new stock prices, the ith+1 row
                new_stock_prices = stocks_data[i+1]
                new_stock_prices = torch.tensor(new_stock_prices)
                new_stock_prices.to(self.dml)
                print("11")
                #append the first element as 1 + risk_free_return as the risk free asset
                new_stock_prices = torch.cat((torch.tensor([1+self.risk_free_return]),new_stock_prices))
                new_stock_prices.to(self.dml)
                #get the new portefolio
                print("12")
                best_portefolio = self.create_optimal_portefolio(old_portefolio,old_stock_prices,new_stock_prices)
                best_portefolio.to(self.dml)
                #create the old portefolio by randomly addding numbers from the best_portefolio
                old_portefolio = best_portefolio + torch.rand((self.portefolio_size))*0.01
                old_portefolio = old_portefolio/torch.sum(old_portefolio)
                last_elements = market_batch[-market_data_sp500.shape[1]:]
                state = (predictions,old_portefolio,last_elements)
                batches_return.append([state,old_stock_prices,new_stock_prices, best_portefolio])
            return batches_return


    def create_optimal_portefolio(self,old_portefolio,old_stock_prices,new_stock_prices):
        """creates the optimal portefolio for the current state, assuming that the stock prices for the next day are known
        """
        #calculate for each stock if its return is less than the trading fee
        new_portefolio = old_portefolio*(old_portefolio*(new_stock_prices - old_stock_prices) > -old_portefolio*self.trade_fee - self.cost_per_trade)
        new_portefolio[0] = new_portefolio[0] +  1 - torch.sum(new_portefolio)
        #check if it to sell any asset to buy the best one 
        new_portefolio = new_portefolio*(new_portefolio*(new_stock_prices - old_stock_prices) < new_portefolio*(new_stock_prices - old_stock_prices).max()  -2*(new_portefolio*self.trade_fee + self.cost_per_trade))
        new_portefolio[(new_stock_prices - old_stock_prices).argmax()] = new_portefolio[(new_stock_prices - old_stock_prices).argmax()] +  1 - torch.sum(new_portefolio)   
        return new_portefolio 

    def train(self,batch_states_max_rewards):
        """takes a batch of states and trains the model towards the maximal reward
        """
        #deep policy gradient was to complex to implement at the time
        #so the model is trained with the maximal reward of the batch
        #the model is trained with the maximal reward of the batch
        
        for state,old_stock_prices,new_stock_prices, best_portefolio in batch_states_max_rewards:
        
            #get the current new portefolio
            old_portefolio = state[1]
            new_portefolio = self.model(tensor_stock_predictions = state[0],tensor_old_portefolio = old_portefolio,tensor_market_data = state[2])
            #new portefolio value - old portefolio value - transaction costs
            #the risrk free has to be at least as big as the trading fees and is the reason why 0 is added
            delta_stock_prices = (new_stock_prices - old_stock_prices)/old_stock_prices
            reward = torch.sum(new_portefolio*delta_stock_prices) - self.trade_fee*torch.sum(torch.abs(new_portefolio - old_portefolio))  - torch.sum(torch.abs(new_portefolio - old_portefolio)[1:self.portefolio_size] != 0)*self.cost_per_trade
            #add a small penalty for money that is not invested
            reward = reward - self.not_invested_penalty*new_portefolio[0]
            #add a second small penalty if more than 30% of the money is invested in one stock or not invested at all
            if new_portefolio.max() > 0.3:
                reward = reward - self.risk_penalty*new_portefolio.max()*5
            #add a small penalty for loosing money
            reward = reward - self.loose_money_penalty*torch.sum(new_portefolio*(new_portefolio*(new_stock_prices - old_stock_prices)<0)*(new_stock_prices - old_stock_prices))            
            max_reward = torch.sum(best_portefolio*delta_stock_prices)
            loss = F.mse_loss(reward, max_reward)
            #backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()     
            #get monetary gain
            gain = torch.sum(new_portefolio*delta_stock_prices)*self.amount   
        return loss.item(),gain
                   
    def save_model(self):
        """Saves the model"""
        torch.save(self.model, str(self.model_name + ".pt"))
    def load_model(self):
        """Loads the model"""
        self.model = torch.load(str(self.model_name + ".pt"))
        

#state/window size
state_size = 60
#initialize the agent
agent = Agent(state_size = state_size, portefolio_size = 40, market_data_size = 12, time_horizon =5,hidden_size = 32, num_layers = 10, news_per_day = 10, embedding_size = 192, stock_model_name = "model_lstm_embeddings_big.pt",model_name="RL_Agent_Dax_length")
#create a test stock model
agent.save_model()
for epoch in range(1,10000):   
    print("starting epoch: ",epoch)
    try:
        while True:
            try:
                print("creating batches")
                train_batch = agent.create_batches()
                print("batches created")
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print("error")
                print(e)
                continue
        print("loading model")
        agent.load_model()
        print("model loaded")
        print("updating stock model")
        agent.update_stock_model()
        print("stock model updated")
        print("training")
        loss_,gain_ = agent.train(train_batch)
        print("training done")
        print("saving model")
        agent.save_model()
        print("model saved")
        print("epoch: ",epoch ,"loss: ",loss_)
        #get monetary gain
        print("gain: ",gain_)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as e:
        print(e)
        time.sleep(1)
        continue