import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from keras.models import load_model
import streamlit as st

#Sensitive information
def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to predict the next day's closing price
def predict_next_day_price(model, df):
    past_100_days = df['Close'].tail(100).values.reshape(-1, 1)
    min_value = past_100_days.min()
    max_value = past_100_days.max()
    past_100_days_scaled = (past_100_days - min_value) / (max_value - min_value)
    x_next_day = np.array(past_100_days_scaled)
    x_next_day = np.reshape(x_next_day, (1, x_next_day.shape[0], 1))
    predicted_next_day_scaled = model.predict(x_next_day)
    predicted_next_day = predicted_next_day_scaled * (max_value - min_value) + min_value
    return predicted_next_day[0, 0]



# Green color style for the predict button
predict_button_style = """
    <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 24px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease 0s;
        }
        div.stButton > button:hover {
            background-color: #45a049;
            color: white;
        }
    </style>
"""

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 10px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Checkbox to toggle sidebar visibility
show_sidebar = True
# show_sidebar = st.checkbox("", value=True, key="show_sidebar_button")

if show_sidebar:
    # Button to select between International and Indian stocks
    selected_section = st.sidebar.selectbox("Choose Market Type:", ("International Stocks", "Indian Stocks","Commodities Stocks"))

    # Sidebar for selecting stock symbol
    st.sidebar.subheader(f'Select Stock Symbol ({selected_section}):')

    if selected_section == "International Stocks":
        stock_symbols = ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'MSFT', 'FB', 'NFLX', 'NVDA', 'INTC', 'AMD',
                     'IBM', 'ORCL', 'CSCO', 'ADBE', 'PYPL', 'CRM', 'BABA', 'TWTR', 'UBER', 'LYFT']
    elif selected_section == "Indian Stocks":
        stock_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFC.NS', 'INFY.NS', 'SBIN.NS', 'ICICIBANK.NS',  'HINDUNILVR.NS', 'KOTAKBANK.NS', 'HDFCBANK.NS', 'LT.NS', 'AXISBANK.NS',
            'ITC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'WIPRO.NS', 'ONGC.NS', 'SUNPHARMA.NS','BHARTIARTL.NS', 'HCLTECH.NS', 'ULTRACEMCO.NS']
    elif selected_section == "Commodities Stocks":
        stock_symbols = ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F', 'PA=F', 'ZC=F', 'ZW=F', 'ZS=F', 'SB=F', 'KC=F', 'CT=F', 'CC=F']

    selected_stock = st.sidebar.selectbox('Select Stock Symbol:', stock_symbols)
    
    # Title of the app
    st.title('Stock Forecasting Master')

    # User inputs
    stock = st.text_input('Enter Stock Symbol (e.g. AAPL, TSLA, AMZN...)', value=selected_stock, key="stock_input")
    start_date = st.date_input('Start Date', datetime(2018, 1, 31))
    end_date = datetime.now()

    # Check if the user input for stock is not empty, then override the selected stock
    if stock:
        selected_stock = stock
        
        stock = selected_stock
        

    # Predict Button Functionality
    st.markdown(predict_button_style, unsafe_allow_html=True)
    if st.button('Predict', key="predict_button"):
        st.subheader(selected_stock)
        try:
            # Download stock data
            df = yf.download(selected_stock, start=start_date, end=end_date)
            
            if df.empty:
                st.error(f"No data available for stock symbol '{stock}' in the selected date range.")
            else:
                
                # if not os.path.exists("CSV"):
                #     os.makedirs("CSV")

                # csv_file_path = os.path.join("CSV", f"{selected_stock}_data.csv")
                # df.to_csv(csv_file_path, index=False)
                
                # df = pd.read_csv(csv_file_path)
                
                df1 = df.reset_index()['Close']

                # Describing Data
                st.subheader('Data Description')
                st.write(df.describe())

                # Start and End dates
                st.subheader('Start and End Dates')
                st.write(df.head(), df.tail())

                # Closing price vs Time chart
                st.subheader('Closing Price vs Time Chart')
                fig = plt.figure(figsize=(12, 6))
                plt.plot(df1)
                st.pyplot(fig)

                # Moving Average 200
                st.subheader('Closing Price vs Time Chart with MA200')
                ma200 = df1.rolling(200).mean()
                fig = plt.figure(figsize=(12, 6))
                plt.plot(df1)
                plt.plot(ma200)
                st.pyplot(fig)
                
                # Moving Average 50 and 200
                st.subheader('Closing price VS Time chart with MA50 and MA200')
                ma50 = df1.rolling(50).mean()
                ma200 = df1.rolling(200).mean()
                fig = plt.figure(figsize=(12, 6))
                plt.plot(df1)
                plt.plot(ma200)
                plt.plot(ma50)
                st.pyplot(fig)

                # Calculate RSI indicator
                rsi_values = calculate_rsi(df['Close'])

                # Plot RSI indicator
                st.subheader('Relative Strength Index (RSI)')
                fig = plt.figure(figsize=(12, 6))
                plt.plot(df.index, rsi_values, label='RSI', color='purple')
                plt.axhline(70, linestyle='--', color='red', alpha=0.5)
                plt.axhline(30, linestyle='--', color='green', alpha=0.5)
                plt.fill_between(df.index, y1=70, y2=30, color='gray', alpha=0.1)
                plt.title('Relative Strength Index (RSI)')
                plt.xlabel('Date')
                plt.ylabel('RSI')
                plt.legend()
                st.pyplot(fig)

                # Split data into training and testing
                data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
                data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

                if not data_training.empty and not data_testing.empty:
                    from sklearn.preprocessing import MinMaxScaler

                    # Feature scaling
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    data_training_array = scaler.fit_transform(data_training)

                    # Load the LSTM model
                    model = load_model('LSTM_Model.h5')

                    # Testing part
                    past_100_days = data_training.tail(100)

                    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
                    input_data = scaler.fit_transform(final_df)

                    x_test = []
                    y_test = []

                    for i in range(100, input_data.shape[0]):
                        x_test.append(input_data[i - 100: i])
                        y_test.append(input_data[i, 0])

                    x_test, y_test = np.array(x_test), np.array(y_test)
                    y_predicted = model.predict(x_test)
                    scaler = scaler.scale_

                    scale_factor = 1 / scaler[0]
                    y_predicted = y_predicted * scale_factor
                    y_test = y_test * scale_factor

                    # Final Graph
                    st.subheader('Predictions vs Original')
                    fig2 = plt.figure(figsize=(12, 6))
                    plt.plot(y_test, 'b', label='Original Price')
                    plt.plot(y_predicted, 'r', label='Predicted Price')
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.legend()
                    st.pyplot(fig2)

                    # Next Day
                    predicted_next_day = predict_next_day_price(model, df)

                    st.subheader('Predicted Price for the Next Day')
                    st.write(f'The predicted closing price for the next day is: **{predicted_next_day:.4f}**')

                    # Predicted Price for the Next Day with Randomness
                    st.subheader('Predicted Price Range for the Next Day ')
                    random_factor = np.random.uniform(-3, 3)
                    predicted_next_day_with_randomness = predicted_next_day + random_factor
                    price_range_next_day = (predicted_next_day_with_randomness - 2, predicted_next_day_with_randomness + 2)
                    user_range_next_day = st.slider("The Range for the next day:", min_value=price_range_next_day[0], max_value=price_range_next_day[1], value=(price_range_next_day[0], price_range_next_day[1]))
                    
                    # Next 10 days
                    past_100_days = df['Close'].tail(100).values.reshape(-1, 1)

                    min_value = past_100_days.min()
                    max_value = past_100_days.max()

                    next_10_days_dates = pd.date_range(df.index[-1], periods=11, freq='B')[1:]

                    predicted_prices_with_dates = []

                    for i in range(10):

                        past_100_days_scaled = (past_100_days - min_value) / (max_value - min_value)

                        x_next_day = np.array(past_100_days_scaled)
                        x_next_day = np.reshape(x_next_day, (1, x_next_day.shape[0], 1))

                        predicted_next_day_scaled = model.predict(x_next_day)

                        predicted_next_day = predicted_next_day_scaled * (max_value - min_value) + min_value

                        predicted_prices_with_dates.append((next_10_days_dates[i], predicted_next_day[0, 0]))

                        past_100_days = np.concatenate([past_100_days, predicted_next_day.reshape(-1, 1)])

                    st.subheader('Predicted Prices for the Next 10 Days')
                    for date, price in predicted_prices_with_dates:
                        st.write(f'{date.strftime("%Y-%m-%d")}: {price:.4f}')

                    # Create a graph for the next 10 days predicted prices
                    predicted_dates, predicted_prices = zip(*predicted_prices_with_dates)
                    fig3 = plt.figure(figsize=(12, 6))
                    plt.plot(predicted_dates, predicted_prices, marker='o', linestyle='-', color='b')
                    plt.title('Predicted Prices for the Next 10 Days')
                    plt.xlabel('Date')
                    plt.ylabel('Predicted Price')
                    plt.xticks(rotation=45)
                    st.pyplot(fig3)

                    # Generate date range for the next 30 days
                    next_30_days_dates = pd.date_range(df.index[-1], periods=31, freq='B')[1:]
                    predicted_prices_with_dates_30_days = []

                    # Predict the next 30 days
                    for i in range(30):

                        past_100_days_scaled = (past_100_days - min_value) / (max_value - min_value)

                        x_next_day = np.array(past_100_days_scaled)
                        x_next_day = np.reshape(x_next_day, (1, x_next_day.shape[0], 1))

                        predicted_next_day_scaled = model.predict(x_next_day)

                        predicted_next_day = predicted_next_day_scaled * (max_value - min_value) + min_value

                        predicted_prices_with_dates_30_days.append((next_30_days_dates[i], predicted_next_day[0, 0]))

                        past_100_days = np.concatenate([past_100_days, predicted_next_day.reshape(-1, 1)])

                    st.subheader('Predicted Prices for the Next 30 Days')
                    for date, price in predicted_prices_with_dates_30_days:
                        st.write(f'{date.strftime("%Y-%m-%d")}: {price:.4f}')
                    

                    # Create a graph for the next 30 days' predicted prices
                    predicted_dates_30_days, predicted_prices_30_days = zip(*predicted_prices_with_dates_30_days)
                    fig4 = plt.figure(figsize=(12, 6))
                    plt.plot(predicted_dates_30_days, predicted_prices_30_days, marker='o', linestyle='-', color='b')
                    plt.title('Predicted Prices for the Next 30 Days')
                    plt.xlabel('Date')
                    plt.ylabel('Predicted Price')
                    plt.xticks(rotation=45)
                    st.pyplot(fig4)

                else:
                    st.warning("Insufficient data for training and testing. Please check your date range.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
