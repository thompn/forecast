import sys
import warnings
warnings.simplefilter('ignore')

import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import mean_absolute_error, mean_squared_error

from datetime import datetime as dt

tddt = dt.strftime(dt.today(), '%Y-%m-%d')

class Forecaster:
    def __init__(self, filename, date_column, column_to_predict):
        self.filename = filename
        self.date_column = date_column
        self.column_to_predict = column_to_predict

    def read_file(self):
        with open(self.filename, 'r') as f:
            df = pd.read_csv(f)
        return df

    def transform_data(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df = df.groupby(self.date_column).agg({self.column_to_predict:'sum'}).reset_index()
        df.columns = ['ds','y']
        return df

    def fit_model(self, df):
        m = Prophet()
        model = m.fit(df)
        return model

    def predict_model(self, model):
        future = model.make_future_dataframe(periods=365, freq='D')
        forecast = model.predict(future)
        return forecast

    def combine_dataframes(self, df, forecast):
        df_combined = pd.concat([df, forecast], axis=1, join='inner')
        return df_combined

    def calculate_residuals(self, df_combined):
        residuals = df_combined['y'] - df_combined['yhat']
        return residuals

    def calculate_metrics(self, df_combined):
        mae = mean_absolute_error(df_combined['y'], df_combined['yhat'])
        mse = mean_squared_error(df_combined['y'], df_combined['yhat'])
        return mae, mse

    def format_forecast_output(self, forecast):
        to_save = forecast[['ds','yhat']]
        to_save.columns = ['date','forecast_value']
        tddt = dt.strftime(dt.today(), '%Y-%m-%d')
        forecast_ready = to_save[to_save['date'] > tddt]
        return forecast_ready

    def save_forecast_output(self, forecast_ready):
        forecast_ready.to_csv(f'forecast_output_{tddt}.csv')

    def save_plots(self, model, forecast, forecast_ready, residuals):
        with PdfPages(f'forecast_{tddt}_plots.pdf') as pdf:
            fig, ax = plt.subplots(figsize=(18,5))
            ax.plot(df['ds'], df['y'])
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Volume', fontsize=14)
            plt.title('Total Actual Volumes - Aggregated Daily - Used to Predict', fontsize=16)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            fig, ax2 = plt.subplots(figsize=(18,8))
            model.plot(forecast, ax=ax2 )
            a = add_changepoints_to_plot(fig.gca(), model, forecast)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Forecasted Volume', fontsize=14)
            plt.title('Aggregated Forecasted Volume', fontsize=16)
            plt.tight_layout()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            model.plot_components(forecast)
            plt.tight_layout()
            plt.title('Components')
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            fig, ax = plt.subplots(figsize=(18,8))
            ax.plot(forecast_ready['date'], forecast_ready['forecast_value'])
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

if __name__ == "__main__":
    filename, date_column, column_to_predict = sys.argv[1], sys.argv[2], sys.argv[3]
    forecaster = Forecaster(filename, date_column, column_to_predict)
    df = forecaster.read_file()
    df = forecaster.transform_data(df)
    model = forecaster.fit_model(df)
    forecast = forecaster.predict_model(model)
    df_combined = forecaster.combine_dataframes(df, forecast)
    residuals = forecaster.calculate_residuals(df_combined)
    mae, mse = forecaster.calculate_metrics(df_combined)
    forecast_ready = forecaster.format_forecast_output(forecast)
    forecaster.save_forecast_output(forecast_ready)
    forecaster.save_plots(model, forecast, forecast_ready, residuals)
    print(f'MAE: {mae}')
    print(f'RMSE: {mse}')