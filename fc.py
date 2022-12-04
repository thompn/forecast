import warnings
warnings.simplefilter('ignore')

import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import mean_absolute_error, mean_squared_error

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', \
    help='The file to read')
parser.add_argument('date_column', \
    help='The name of the date column')
parser.add_argument('column_to_predict', \
    help='This is the column in the data containing the volume you wish to predict')
args = parser.parse_args()

# read the file into a df
with open(args.filename, 'r') as f:
    df = pd.read_csv(f)

# import datetime and get todays date in tidy
# string format yyyy-mm-dd
from datetime import datetime as dt
tddt = dt.strftime(dt.today(), '%Y-%m-%d')

# transform the date column to a datetime type
df['date'] = pd.to_datetime(df['date'])

# group by the date, to aggregate daily
# and sum the total volumes to get a daily aggregated
# call volume, ignoring the country split
df = df.groupby(args.date).agg({args.column_to_predict:'sum'}).reset_index()

# rename the columns to be accepted by prophet
df.columns = ['ds','y']

# instantiate prophet
m = Prophet()

# fit the model to the call data in df
model = m.fit(df)

# use the fitted model to make a prediction +365 periods on a daily basis
future = m.make_future_dataframe(periods=365, freq='D')

# predict the forecast and save to forecast df
forecast = m.predict(future)

# Combine the two dataframes using the 'ds' column as the key
df_combined = pd.concat([df, forecast], axis=1, join='inner')

# Calculate the residuals by subtracting the predicted values from the actual values
residuals = df_combined['y'] - df_combined['yhat']

# Calculate the MAE and MSE
mae = mean_absolute_error(df_combined['y'], df_combined['yhat'])
mse = mean_squared_error(df_combined['y'], df_combined['yhat'])

# filter only date and volumes to output
to_save = forecast[['ds','yhat']]
to_save.columns = ['date','forecast_value']

# start forecast output from tomorrow
forecast_ready = to_save[to_save['date'] > tddt]

# save forecast output to csv file
forecast_ready.to_csv(f'forecast_output_{tddt}.csv')

# Create a PDF file to add plots to
with PdfPages(f'forecast_{tddt}_plots.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(18,5))
    ax.plot(df['ds'], df['y'])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volume', fontsize=14)
    plt.title('Total Actual Volumes - Aggregated Daily - Used to Predict', fontsize=16)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    fig, ax2 = plt.subplots(figsize=(18,8))
    m.plot(forecast, ax=ax2 )
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Forecasted Volume', fontsize=14)
    plt.title('Aggregated Forecasted Volume', fontsize=16)
    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    m.plot_components(forecast)
    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    fig, ax = plt.subplots(figsize=(18,8))
    ax.plot(forecast_ready['ds'], forecast_ready['yhat'])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Forecasted Volume', fontsize=14)
    plt.title('Aggregated Forecasted Volume', fontsize=16)
    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    plt.plot(residuals)
    plt.tight_layout()
    plt.title('Residuals')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

# print the evaluation metrics
print(f'MAE: {mae}')
print(f'RMSE: {mse}')
