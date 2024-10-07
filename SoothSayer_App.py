import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
#import plotly.express as px
from prophet import Prophet
#from prophet.plot import plot_plotly, plot_components_plotly
#import statsmodels.api as sm

#from sklearn.model_selection import train_test_split
#st.set_option('deprecation.showPyplotGlobalUse', False)


transaction_01 = pd.read_csv('https://github.com/Agnika0987/Demand-Forecasting-System/tree/main/Transactional_data_retail_01.csv', low_memory=False)

transaction_01['InvoiceDate'] = pd.to_datetime(transaction_01['InvoiceDate'], format='%d %B %Y')

# Format the parsed dates to 'dd-mm-yyyy'
transaction_01['InvoiceDate'] = transaction_01['InvoiceDate'].dt.strftime('%d-%m-%Y')
transaction_02  = pd.read_csv('https://github.com/Agnika0987/Demand-Forecasting-System/tree/main/Transactional_data_retail_02.csv', low_memory=False)

transactions = pd.concat([transaction_01, transaction_02], ignore_index=True)

transactions.drop_duplicates(inplace=True)

transactions['Customer ID'] = transactions['Customer ID'].fillna('Guest')


transactions['Customer ID'] = transactions['Customer ID'].astype(str)
transactions['Customer ID'] = transactions['Customer ID'].str.replace(r'\.0$', '', regex=True)


transactions['InvoiceDate'] = pd.to_datetime(transactions['InvoiceDate'], format='%d-%m-%Y', errors='coerce')   # format='%d %B %Y')
transactions['Time'] = transactions['InvoiceDate'].dt.strftime('%H:%M')
transactions['Year'] = transactions['InvoiceDate'].dt.year
transactions['Month'] = transactions['InvoiceDate'].dt.month
transactions['Week'] = transactions['InvoiceDate'].dt.isocalendar().week

customer_data = pd.read_csv('https://github.com/Agnika0987/Demand-Forecasting-System/tree/main/CustomerDemographics.csv')
customer_data['Customer ID'] = customer_data['Customer ID'].astype(str)
customer_data['Customer ID'] = transactions['Customer ID'].str.replace(r'\.0$', '', regex=True)

product_info = pd.read_csv('https://github.com/Agnika0987/Demand-Forecasting-System/tree/main/ProductInfo.csv')
product_info['Description'] = product_info['Description'].fillna('No Description')

transactional_data = transactions.merge(customer_data, on='Customer ID', how='left').merge(product_info, on='StockCode', how='left')

transactional_data['TotalPrice'] = transactional_data["Quantity"] *transactional_data['Price']

## dropping columns to avoid duplicates 
transactions_dropped = transactional_data.drop(columns=['Country','Description','Price'])

# Remove duplicates
transactional_data1 = transactions_dropped.drop_duplicates()

# Top 10 Stockcodes

top_stock_codes = transactional_data1.groupby('StockCode').agg(TotalQuantitySold=('Quantity', 'sum')).reset_index()
 
# Sort by TotalQuantitySold in descending order and select the top 10
top_10_stock_codes = top_stock_codes.sort_values(by='TotalQuantitySold', ascending=False).head(10)
 
stock_code_to_description = product_info.set_index('StockCode')['Description'].to_dict()

top_10_stock_codes['Description'] = top_10_stock_codes['StockCode'].map(stock_code_to_description)

top_revenue_products = transactional_data1.groupby('StockCode').agg(TotalRevenue=('TotalPrice', 'sum')).reset_index()
 
# Sort by TotalRevenue in descending order and select the top 10
top_10_revenue_products = top_revenue_products.sort_values(by='TotalRevenue', ascending=False).head(10)
 
top_10_revenue_products['Description'] = top_10_revenue_products['StockCode'].map(stock_code_to_description)

unique_stock_codes = top_10_stock_codes['StockCode'].unique()

#filter the data based on above stock codes 

filtered_transactions = transactional_data1[transactional_data1['StockCode'].isin(unique_stock_codes)]

agg_stock_code_qty = filtered_transactions.groupby(['StockCode','Year', 'Month','Week']).agg( TotalQuantitySold=('Quantity', 'sum')).reset_index()
agg_stock_code_price=filtered_transactions.groupby(['StockCode','Year', 'Month','Week']).agg( TotalSumPrice=('TotalPrice', 'sum')).reset_index()

#add date ass index for timeseries 
agg_stock_code_qty['Date'] = pd.to_datetime(agg_stock_code_qty['Year'].astype(int).astype(str) + '-W' + agg_stock_code_qty['Week'].astype(str)+'-1', format='%Y-W%W-%w')
agg_stock_code_qty.sort_values('Date', inplace=True)
agg_stock_code_qty.set_index('Date', inplace=True)



page = st.sidebar.radio("Navigation", ('Top Stocks', 'Forecasting'))

if page == 'Top Stocks':
    # Sidebar filters
    selected_criteria = st.sidebar.selectbox('Select a Criteria', ['Quantity', 'Revenue'])
    if selected_criteria == 'Quantity':
        filtered_df = top_10_stock_codes
    else: 
        filtered_df = top_10_revenue_products
    
    # Plotting top 10 sales by Stock Code
    st.subheader('Top 10 Stock Codes by Criteria')

    # Create a Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(filtered_df.columns),
                    fill_color='lightskyblue',
                    line_color='darkslategray',
                    align='center'),
        cells=dict(values=[filtered_df[col] for col in filtered_df.columns],
                   fill_color='white',
                   line_color='darkslategray',
                   align='center'))
    ])


    fig.update_layout(title="Top 10 Stock Codes", height=400)


    st.plotly_chart(fig, use_container_width=True) # Display Plotly figure

elif page == 'Forecasting':
    
    st.title("Demand Forecasting")
    st.sidebar.title("Input Options")

    # Get unique stock codes for selection
    stock_codes = agg_stock_code_qty['StockCode'].unique()

    # Stock selection
    stock_code = st.sidebar.selectbox("Select a Stock Code:", stock_codes)

    filtered_df = pd.DataFrame(agg_stock_code_qty[agg_stock_code_qty['StockCode'] == stock_code])
    
    train_size = int(len(filtered_df) * 0.8)
    train_df = filtered_df.iloc[:train_size]
    test_df = filtered_df.iloc[train_size:,:]

    train = train_df['TotalQuantitySold']  # This will be a Series
    test = test_df['TotalQuantitySold']
    # decomposition = seasonal_decompose(filtered_df['TotalQuantitySold'], model='additive', period=52)
    # # Displaying the seasonal-trend decomposition
    # st.subheader('Seasonal-Trend Decomposition (STL)')
    # fig_stl = decomposition.plot()
    # st.pyplot(fig_stl)

    # # Plot function for actual data
    # def plot_actual_data():
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(train.index, train, label='Train Data')
    #     plt.plot(test.index, test, label='Test Data', color='orange')
    #     plt.title(f'Total Quantity Sold for {stock_code}')
    #     plt.xlabel('Date')
    #     plt.ylabel('Total Quantity Sold')
    #     plt.legend()
    #     st.pyplot(plt)
    
    def plot_error_distributions(train_actual, train_predictions, test_actual, test_predictions):
        # Ensure the inputs are numeric and print their types
        train_actual = pd.to_numeric(train_actual, errors='coerce')
        train_predictions = pd.to_numeric(train_predictions, errors='coerce')
        test_actual = pd.to_numeric(test_actual, errors='coerce')
        test_predictions = pd.to_numeric(test_predictions, errors='coerce')
    
        train_actual = train_actual.dropna()
        train_predictions = train_predictions.dropna()
        test_actual = test_actual.dropna()
        test_predictions = test_predictions.dropna()
    
        min_length = min(len(train_actual), len(train_predictions), len(test_actual), len(test_predictions))
        train_actual = train_actual.iloc[:min_length]
        train_predictions = train_predictions.iloc[:min_length]
        test_actual = test_actual.iloc[:min_length]
        test_predictions = test_predictions.iloc[:min_length]
        
        # Check for remaining discrepancies in length
        if len(train_actual) != len(train_predictions):
            print("Warning: Train actual and predictions lengths do not match!")
    
        # Calculate errors
        try:
            train_errors = train_actual - train_predictions
        except Exception as e:
            print("Error during calculation of train_errors:", e)
            return
        
        try:
            test_errors = test_actual - test_predictions
        except Exception as e:
            print("Error during calculation of test_errors:", e)
            return
        
        
        col1, col2 = st.columns(2)

        with col1:
            fig, axes = plt.subplots(figsize=(6, 4))
            axes.hist(train_errors, bins=30, color='blue', alpha=0.7)
            axes.set_title('Training Error Distribution')
            axes.set_xlabel('Error')
            axes.set_ylabel('Frequency')
            st.pyplot(fig)

        with col2:
            fig, axes = plt.subplots(figsize=(6, 4))
            axes.hist(test_errors, bins=30, color='orange', alpha=0.7)
            axes.set_title('Testing Error Distribution')
            axes.set_xlabel('Error')
            axes.set_ylabel('Frequency')
            st.pyplot(fig)
        
    ####ARIMA Model
    def arima_model():
        arim = pm.auto_arima(train, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, w=52,
                                    start_P=0, seasonal=True, d=None, D=1, trace=True, error_action='ignore',
                                    suppress_warnings=True, stepwise=True)
        predictions = arim.predict(n_periods=15)  # Forecast for 15 weeks
        
        train_predictions = arim.predict(n_periods=len(train))
        test_predictions = arim.predict(n_periods=len(test))
        
        
        # plt.figure(figsize=(6, 4))
        # plt.plot(train.index, train, label='Train Data')
        # plt.plot(test.index, test, label='Test Data', color='orange')
        # future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(weeks=1), periods=15, freq='W')
        # plt.plot(future_dates, predictions, label='ARIMA Predictions', color='green')
        # plt.title('ARIMA Model Predictions for 15 Weeks')
        # plt.legend()
        # st.pyplot(plt)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Test Data', line=dict(color='orange')))
        #fig.add_trace(go.Scatter(x=test.index, y=test_predictions, mode='lines', name='Test DataP', line=dict(color='black')))
        future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(weeks=1), periods=15, freq='W')
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='ARIMA Predictions', line=dict(color='green')))
        fig.update_layout(
            title='ARIMA Model Predictions for 15 Weeks',
            xaxis_title='Date',
            yaxis_title='Values',
            legend_title='Legend',
            height=400,
            width=700
        )
        st.plotly_chart(fig)
        
        plot_error_distributions(train, train_predictions , test, test_predictions)

    ######ETS Model
    def ets_model():
        ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=4)
        ets_model_fit = ets_model.fit()
        predictions = ets_model_fit.forecast(15)  # Forecast for 15 weeks
        
        train_predictions = ets_model_fit.forecast(len(train))
        test_predictions = ets_model_fit.forecast(len(test))
        
        plot_error_distributions(train, train_predictions, test, predictions)
    
        # plt.figure(figsize=(6, 4))
        # plt.plot(train.index, train, label='Train Data')
        # plt.plot(test.index, test, label='Test Data', color='orange')
        # future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(weeks=1), periods=15, freq='W')
        # plt.plot(future_dates, predictions, label='ETS Predictions', color='green')
        # plt.title('ETS Model Predictions for 15 Weeks')
        # plt.legend()
        # st.pyplot(plt)
        
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Test Data', line=dict(color='orange')))
        #fig.add_trace(go.Scatter(x=test.index, y=test_predictions, mode='lines', name='Test DataP', line=dict(color='black')))
        future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(weeks=1), periods=15, freq='W')
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='ETS Predictions', line=dict(color='green')))
        fig.update_layout(
            title='ETS Model Predictions for 15 Weeks',
            xaxis_title='Date',
            yaxis_title='Values',
            legend_title='Legend',
            height=400,
            width=700
        )
        st.plotly_chart(fig)
    
    ####Prophet Model
    def prophet_model():
        prophet_train = train.reset_index().rename(columns={'Date': 'ds', 'TotalQuantitySold': 'y'})
        prophet_test = test.reset_index().rename(columns={'Date': 'ds', 'TotalQuantitySold': 'y'})
        
        model = Prophet()
        model.fit(prophet_train)
        
  
        future = model.make_future_dataframe(periods=15, freq='W')
        forecast = model.predict(future)
        
        train_forecast = model.predict(prophet_train[['ds']])  
        test_predictions = forecast['yhat'].tail(15) 
        test_actual = prophet_test['y']
        plot_error_distributions(prophet_train['y'], train_forecast['yhat'], test_actual, test_predictions)
    
    
        
        # plt.figure(figsize=(6, 4))
        # plt.plot(prophet_train['ds'], prophet_train['y'], label='Train Data')
        # plt.plot(prophet_test['ds'], prophet_test['y'], label='Test Data', color='orange')
        # plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Predictions', color='green')
        # plt.title('Prophet Model Predictions for 15 Weeks')
        # plt.legend()
        # st.pyplot(plt)
        
        
                
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prophet_train['ds'], y=prophet_train['y'], mode='lines', name='Train Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=prophet_test['ds'], y= prophet_test['y'], mode='lines', name='Test Data', line=dict(color='orange')))
        #fig.add_trace(go.Scatter(x=test.index, y=test_predictions, mode='lines', name='Test DataP', line=dict(color='black')))
        #future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(weeks=1), periods=15, freq='W')
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Prediction', line=dict(color='green')))
        fig.update_layout(
            title='Prophet Model Predictions for 15 Weeks',
            xaxis_title='Date',
            yaxis_title='Values',
            legend_title='Legend',
            height=400,
            width=700
        )
        st.plotly_chart(fig)
    
    
        
    # Display stock data
    st.subheader(f"Demand Overview for {stock_code}")
    
    st.sidebar.subheader("Model Selection")

    model_choice = st.sidebar.selectbox("Choose a model:", ["ARIMA", "ETS", "Prophet"])
    
    if model_choice == "ARIMA":
        arima_model()
    elif model_choice == "ETS":
        ets_model()
    elif model_choice == "Prophet":
        prophet_model()

