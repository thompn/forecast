
# Forecast Daily Total Volumes

This code uses the [Prophet](https://facebook.github.io/prophet/) library to forecast daily total volumes using data from a file specified as an argument.

For now, this only covers daily simplistic forecasting.

## Requirements

-   Python 3.6 or higher
-   Pandas
-   Prophet
-   Matplotlib
-   Sklearn

## Usage

First set up your virtual environment. Copy the following code into a terminal:

`python -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

Now you're ready to run, use the following command to run the forecast:

`python main.py filename column_to_predict`

where `filename` is the path to the csv file containing the data to be used to make the forecast. The file should contain a `date` column with dates in `yyyy-mm-dd` format and a `volume` column with the value you wish to predict.

## Output

The code will output two files:

-   `forecast_output_yyyy-mm-dd.csv`: a csv file containing the forecasted total inbound volumes for each day starting from the day after the last day in the input data.
-   `forecast_yyyy-mm-dd_plots.pdf`: a pdf file containing four plots:
    -   A plot of the actual total volumes used to make the forecast.
    -   A plot of the forecasted total volumes.
    -   A plot showing the components of the forecast, including the trend and weekly and yearly seasonality.
    -  The residuals of the fitted model

The code will also print the mean absolute error (MAE) and mean squared error (MSE) of the forecast to the terminal.