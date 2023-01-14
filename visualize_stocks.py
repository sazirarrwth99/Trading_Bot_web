import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import datetime as dt
import time
import plotly.graph_objects as go
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import os
import logging
import numpy as np
from tqdm import tqdm
import os
import math
import torch
import numpy as np
import pandas as pd
from agent import *
from enviroment import *
import os

enviroment = Enviroment("/app/trading_bot_web/AAPL_daily_yf_test.csv")
enviroment.data_indicator = ".csv"
#enviroment.append_new_data("Close_Minute/")
enviroment.get_data()
enviroment.window_size = 1024
enviroment.create_agent(model_name="agent_lstm_appl_train")
stock_symbol = "AAPL"
start_time = dt.datetime.now() - dt.timedelta(days=5)
stock_data = yf.download(stock_symbol, start=start_time, interval="1m",period="5d")
while True:
    actual_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.title("Live Trading Bot --BETA--")
    #set streamlit theme to light
    
    stock_symbol = st.text_input("Enter a stock symbol (e.g. GOOG, AAPL): ")
    if stock_symbol is None or stock_symbol == "":
        stock_symbol = "AAPL" # set a default symbol
    cash_ = st.text_input("Enter a portefolio value (e.g. 10000): ")
    if cash_ is None or cash_ == "":
        cash_ = 10000 # set a default symbol
    # Get the stock data
    try:
        start_time = dt.datetime.now() - dt.timedelta(days=5)
        while stock_data["Close"].max() < 10:
            stock_data = yf.download(stock_symbol, start=start_time, interval="1m",period="5d")
        timeseries = list(stock_data["Close"].values)
        list_actions = [ enviroment.predict(timeseries, t=i) for i in tqdm(range(1023, len(timeseries)))]
        #generate profits
        cash = int(cash_)
        stocks = 0
        portefolio_value = cash
        portefolio_values = []
        for stock_num in range(1023,len(timeseries)):
            if list_actions[stock_num-1023] == 1:
                if cash > 0:
                    stocks = cash/timeseries[stock_num]
                    cash = 0
            elif list_actions[stock_num-1023] == 2:
                if stocks > 0:
                    #cash = stocks*timeseries[stock_num] 
                    cash = stocks*timeseries[stock_num] 
                    stocks = 0
            portefolio_value = cash + stocks*timeseries[stock_num]
            portefolio_values.append(portefolio_value)     
        #create a list with the dates
        stock_data["Date"] = pd.to_datetime(stock_data.index)
        dates_min = pd.to_datetime(stock_data["Date"].iloc[1023:])
        dates_min = list(dates_min)    
        start_price = stock_data["Close"].iloc[-len(portefolio_values)]
        
        #plot  with plotly
        colors = ["#003366","#99ccff"]
        traces = [
            go.Scatter(x=stock_data.index[-len(portefolio_values):], y=(stock_data["Close"].iloc[:-len(portefolio_values)]/start_price)*int(cash_), name="Value Apple Stocks without trading", line=dict(color=colors[0])),
            go.Scatter(x=dates_min, y=portefolio_values, name="RL Portefolio value", line=dict(color=colors[1]))
        ]
        layout = go.Layout(title='Portfolio Value over Time',
                   xaxis=dict(title='Date', tickangle=45, showgrid=True, showline=True, showticklabels=True,  tickmode='auto', nticks=0, tickfont=dict(size=14, color='black'), side='bottom', mirror='all', gridcolor='#bdbdbd', gridwidth=1, zeroline=True, zerolinecolor='#969696', zerolinewidth=1, linecolor='#636363', linewidth=1, ticks='outside', tickcolor='#636363', ticklen=5, tickwidth=1),
                   yaxis=dict(title='Portfolio Value [€]', showgrid=True, showline=True, showticklabels=True,  tickmode='auto', nticks=0, tickfont=dict(size=14, color='black'), side='left', mirror='all', gridcolor='#bdbdbd', gridwidth=1, zeroline=True, zerolinecolor='#969696', zerolinewidth=1, linecolor='#636363', linewidth=1, ticks='outside', tickcolor='#636363', ticklen=5, tickwidth=1, tickangle=0),
                   hovermode='x',
                   showlegend=True,
                   plot_bgcolor='white',
        )      
        fig = go.Figure(data=traces, layout=layout)
        #set size 
        fig.update_layout(  
            autosize=False,
            width=1000,
            height=500,
        )
        #make bigger plot
        st.plotly_chart(fig, use_container_width=True)
                   
    except ValueError:
        st.error("Invalid stock symbol.")
    #except Exception as e:
        st.error("An error occurred while retrieving the data.")
        print(e)
    st.write("Last updated at: " + actual_time)
    st.balloons()
    time.sleep(10) # refresh every minute
    st.experimental_rerun()
