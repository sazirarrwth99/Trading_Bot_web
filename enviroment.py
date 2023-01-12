import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import os
import numpy as np
from tqdm import tqdm
import os
import math
import torch
import numpy as np
import pandas as pd
from agent import *


class Enviroment:
    """The environment class for the trading bot, where he interacts
    
    data_paths (list) = places where the first data is stored, has to be a string first
                  can describe the path to a folder or a file
    data_indicator (string) = the indicator for the data, default is .csv but can be changed
                                for example: _market_data.csv can be used to mark all the 
                                data that is used for the model, has to end with .csv
    window_size (int) = the size of the window for the model (timesteps)
    reward_dict (dict, optional) =  sets the reward funktion for the agent, see calc_reward for more :
                                    risk_free_return (float) = float, default is 0.0
                                    risk_aversion (float) = profit * (1-(volatility/price)*risk_aversion)), default is 0.01
                                    trading_fees (float) = profit - trading_fees, per trade, default is 0.0
                                    trading_fees_percentage = profit*(1-trading_fees_percentage), per volumina, default is 0.0
                                    sharpe_ratio_value = profit * (1-sharpe_ratio)*sharpe_ratio_value, default is 0.0
                                    
    """
    
    def __init__(self, data_path, data_indicator = ".csv", window_size = 10):
        self.data_paths = [data_path]
        self.window_size = window_size
        self.data_indicator = data_indicator
        #dictionary with the stored dataframes and the name of the dataframe/ file its from
        self.dict_dataframes = {}
        self.agents = {}
        self.actual_agent = None
        self.actual_agent_name = None
        self.calculated_dataframe = {}
        self.reward_dict = {}
        self.risk_aversion = 0.0
        self.trading_fees = 0.0
        self.trading_fees_percentage = 0.0
        self.risk_free_return = 0.0
        self.sharpe_ratio_value = 0.0
        
    def set_trading_fees(self, dict_rewards):
        """sets the trading fees for the agents
        #takes a dictionary with the fees
        """
        if "trading_fees" in dict_rewards:
            self.trading_fees = dict_rewards["trading_fees"]
        if "trading_fees_percentage" in dict_rewards:
            self.trading_fees_percentage = dict_rewards["trading_fees_percentage"]
        if "risk_free_return" in dict_rewards:
            self.risk_free_return = dict_rewards["risk_free_return"]
        if "sharpe_ratio_value" in dict_rewards:
            self.sharpe_ratio_value = dict_rewards["sharpe_ratio_value"]
        if "risk_aversion" in dict_rewards:
            self.risk_aversion = dict_rewards["risk_aversion"]  
    
    def calc_reward(self,return_adj, list_data, timestemp):
        """ Takes a profit and applies the trading fees,
            uses the dict_trading_fees but can be changed to individual fees 
            or pricing of risk
            gets: 
                return_adj(float) = return adjusted for trading fees
                list_data(list) = list of the data
                timestemp(int) = timestemp of the data
            params:
                risk_free_return (float) = float, default is 0.0
                risk_aversion (float) = profit * (1-(volatility/price)*risk_aversion)), default is 0.01
                trading_fees (float) = profit - trading_fees, per trade, default is 0.0
                trading_fees_percentage = profit*(1-trading_fees_percentage), per volumina, default is 0.0
                sharpe_ratio_value = profit * (1-sharpe_ratio)*sharpe_ratio_value, default is 0.0

            calculates:
                volatility (float) = standard deviation of the data
                sharpe_ratio (float) = float, default is 0.0
            

        """
        volatility = max(np.std(list_data[timestemp-self.window_size:timestemp]) , 0.0001)
        #check if volatility if a float, not an nan,
        # and not higher than the max of 50% of the price 
        selling_price = list_data[timestemp]
        return_adj_percentage = return_adj/selling_price
        if math.isnan(volatility) or volatility == 0.0 or volatility > (selling_price/2):
            sharpe_ratio_factor = 1
        else:
            sharpe_ratio = (return_adj_percentage - self.risk_free_return) / volatility
            #best case sharp ratio is below 1
            sharpe_ratio_factor = (1 - max(min(sharpe_ratio,1),0))*self.sharpe_ratio_value
        #calculate the reward
        return return_adj_percentage * (1 - (volatility/selling_price)*self.risk_aversion) * (1 - sharpe_ratio_factor)
                      
        
            
            
    
    def get_status(self):
        """prints the status of the environment
        """
        print("data: ", self.data)
        print("window_size: ", self.window_size)
        print("data_indicator: ", self.data_indicator)
        print("dict_files: ", self.dict_files)
        print("agents: ", self.agents)
        
    def append_new_data(self, new_data):
        """appends new data to the data list
        """
        self.data_paths.append(new_data)
        
    def set_window_size(self, window_size):
        """sets the window size
        """
        self.window_size = window_size
        
    def get_data(self):
        """creates a dictionary with the data and the name of the data
        """
        for i in self.data_paths:
            #check if the data is a folder or a file
            if os.path.isdir(i):
                #get all the files in the folder
                for file in os.listdir(i):
                    #check if the file is a csv file
                    if file.endswith(self.data_indicator):
                        #read the csv file and add it to the dictionary
                        self.dict_dataframes[file] = pd.read_csv(i + "/" + file)
            else:
                #read the csv file and add it to the dictionary
                self.dict_dataframes[i] = pd.read_csv(i)
            
    def print(self):
        """prints the dataframes actually in storage
        """
        print(self.dict_dataframes)
    
    def create_agent(self, model_name):
        """creates an agent and appends it to the agent dictionary
        model name (string) = the name of the model equals the name of the agent
        sets the actual agent to the new agent
        """
        self.agents[model_name] = Agent(self.window_size, model_name = model_name)
        self.actual_agent = self.agents[model_name]
        self.actual_agent_name = model_name
        
    def set_actual_agent(self, model_name):
        """sets the actual agent to the model name
        """
        self.actual_agent = self.agents[model_name]
        self.actual_agent_name = model_name
        
    def preprocess_dataframe(self, dataframe, columns = ["CLOSE"], how = "minmax", scaler = None):
        """preprocesses the dataframe, 
        look in preprocess_data for more info
        """
        #remove duplicate rows
        dataframe = dataframe.drop_duplicates(subset = columns)
        #remove NaN values if NaN in the columns
        dataframe = dataframe.dropna(subset = columns)
        #for safety fill the NaN in the column with the interpolation
        try:
            dataframe = dataframe.interpolate(method = "linear", columns = columns)
        except Exception as e:
            print(e)
            print("could not interpolate the data")
            print("maybe a error")
    
        #scale the data
                    
        if how == "minmax":
            Scaler = MinMaxScaler()
        elif how == "standard":
            Scaler = StandardScaler()
        elif how == "robust":
            Scaler = RobustScaler()
        elif how == "maxabs":
            Scaler = MaxAbsScaler()
        
        #scale 
        dataframe[columns] = Scaler.fit_transform(dataframe[columns])
        
        return dataframe
        
    def preprocess_data(self, data_name = "all", columns = ["CLOSE"],
                        how = "minmax", scaler = None):
        """preprocess the data, 
        data_name(string, list) if data_name is all, then all the data is preprocessed, if its 
                                a string, then only the data with that name is preprocessed, if
                                its a list, then all the dataframes in the list are preprocessed
        columns (list) = the columns that are used for the preprocessing, default is ["CLOSE"]
        """
        #check if the data_name is a list or a string
        if type(data_name) == list:
            #loop over the list and preprocess the data
            for i in data_name:
                self.dict_dataframes[i] = self.preprocess_dataframe(self.dict_dataframes[i], columns = columns, 
                                                          how = how, scaler = scaler)
        elif data_name == "all":
            #loop over the dictionary and preprocess the data
            for i in self.dict_dataframes:
                self.dict_dataframes[i] = self.preprocess_dataframe(self.dict_dataframes[i], columns = columns, 
                                                          how = how, scaler = scaler)
        else:
            #preprocess the data
            self.dict_dataframes[data_name] = self.preprocess_dataframe(self.dict_dataframes[data_name], columns = columns, 
                                                          how = how, scaler = scaler)
            
    def get_state(self,data, t, n_days):
        """Returns an state representation ending at time t, starting n_days before t
        data (list) = the data that is used for the state
        t (int) = the time
        n_days (int) = the number of days that are used for the state
        returns a tensor with the state
        """
        sigmoid = torch.nn.Sigmoid()
        d = t - n_days + 1
        if d >= 0:
            block = data[d: t + 1]
        else:
            block = -d * [data[0]] + data[0: t + 1]  # pad with t0   
        res = [block[i + 1] - block[i] for i in range(n_days - 1)]
        return sigmoid(torch.tensor([res]))

    def train(self, datafiles,columns = ["CLOSE"], date_column = "Date", batch_size = 32, verbose = False, save_df = False):   
        """
        datafiles (list or string) = the datafiles that are used for training,
                                      , if all the data is used, then datafiles = "all"
        Trains an agent with an environment.
        columns (list) = the columns that are used for the training, default is ["CLOSE"]
        batch_size (int) = the size of the batch, default is 32
        verbose (bool) = if true, then the training is verbose, default is False
        returns a dataframe the trades of the agent, and a dict with information about the training
        """
        #make list
        if type(datafiles) == str:
            if datafiles == "all":
                datafiles = list(self.dict_dataframes.keys())
            else:
                datafiles = [datafiles]
        
        for datafile in datafiles:
            #start with a profit of 0, and no stocks owned
            profit = 0
            self.actual_agent.inventory = []
            name_agent = self.actual_agent.model_name
            #create a column with the trades
            dataframe = self.dict_dataframes[datafile]
            dataframe[str("trade" + str(name_agent))] = ""
            #append a column with the profit
            dataframe[str("profit" + str(name_agent))] = 0
            #append a column with the reward
            dataframe[str("reward" + str(name_agent))] = 0
            list_data = dataframe[columns].values.tolist()
            #gets the first element in the list of lists
            #!!! later when training with multiple columns, this needs to be changed
            list_data = [i[0] for i in list_data]
            #iterate over the dataframe
            print("training the model " + str(name_agent) + " with the data: " + str(datafile))
            #get initial state
            window_size = self.window_size
            state = self.get_state(list_data, 0, window_size + 1)
            for timestemp in tqdm(range(len(list_data)-1)):
                #get the next state
                next_state = self.get_state(list_data, timestemp + 1, window_size + 1)
                #set the inventory
                
                #get the action
                action = self.actual_agent.get_action(state)
                #get the reward
                reward = 0
                #prevent confusion
                action_done = ""
                if action == 1:
                    #buy
                    self.actual_agent.inventory.append(list_data[timestemp])
                    dataframe.iloc[timestemp, dataframe.columns.get_loc(str("trade" + str(name_agent)))] = "buy"
                    action_done = "buy"
                elif action == 2 and len(self.actual_agent.inventory) > 0:
                    #sell
                    bought_price = self.actual_agent.inventory.pop(0)
                    dataframe.iloc[timestemp, dataframe.columns.get_loc(str("trade" + str(name_agent)))] = "sell"
                    return_adj = list_data[timestemp]*(1-self.trading_fees_percentage) - self.trading_fees - bought_price
                    profit += return_adj
                    dataframe.iloc[timestemp, dataframe.columns.get_loc(str("profit" + str(name_agent)))] = profit
                    reward = self.calc_reward(return_adj, list_data, timestemp)
                    dataframe.iloc[timestemp, dataframe.columns.get_loc(str("reward" + str(name_agent)))] = reward
                    action_done = "sell"
                else:
                    #hold
                    dataframe.iloc[timestemp, dataframe.columns.get_loc(str("trade" + str(name_agent)))] = "hold"
                    #make a small negative reward for holding stocks because the money could 
                    #be invested otherwise
                    reward = - list_data[timestemp] * len(self.actual_agent.inventory) * self.risk_free_return
                    action_done = "hold"
                done = True if timestemp == len(list_data) - 2 else False
                #add the state to the memory
                self.actual_agent.remember(state, action, reward, next_state, done)
                
                if len(self.actual_agent.memory) > batch_size:
                    loss = self.actual_agent.replay(batch_size)
                    
                if verbose:
                    print("time: " + str(dataframe[date_column].iloc[timestemp]) + " action: " + str(action_done) + " price: " + str(round(list_data[timestemp],2)) + " reward: " + str(round(reward,2)) + " profit: " + str(round(profit,2)))
                state = next_state
                #append the dataframe to the dict
                self.calculated_dataframe[datafile] = dataframe
                #save the model for every batches
                if timestemp % (batch_size) == 0:
                    self.actual_agent.save_model()
            
            print("total profit: " + str(round(profit,2)))
            
            if save_df:
                self.calculated_dataframe.to_csv(str(datafile) + str(name_agent) + "calc.csv")
        
    def set_model_agent(self, model_agent):
        """sets the model of the actural agent
        updates the agent on the dict
        """
        self.actual_agent.model = model_agent
        self.dict_agents[self.actual_agent.actual_agent_name] = model_agent   
            
    def predict(self,timeseries,t=1, verbose=False):
        """gets a timeseries and gets the next action
        has to be a list an bigger than the window_size
        returns the action
        1 = buy
        2 = sell
        0 = hold"""
        state = self.get_state(timeseries, t, self.window_size)
        action = self.actual_agent.get_action(state)
        if verbose:
            if action == 1:
                print("buy")
            if action == 2:
                print("sell")
            if action == 0:
                print("hold")
        return action