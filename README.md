
# Forecast Total Inbound Call Volumes

This code uses the [Prophet](https://facebook.github.io/prophet/) library to forecast daily total inbound call volumes using data from a file specified as an argument.

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

`python main.py filename` 

where `filename` is the path to the file containing the data to be used to make the forecast. The file should contain a `date` column with dates in `yyyy-mm-dd` format and a `total_inbound_calls` column with the number of inbound calls for each date.

## Output

The code will output two files:

-   `forecast_output_yyyy-mm-dd.csv`: a csv file containing the forecasted total inbound call volumes for each day starting from the day after the last day in the input data.
-   `forecast_yyyy-mm-dd_plots.pdf`: a pdf file containing four plots:
    -   A plot of the actual total inbound call volumes used to make the forecast.
    -   A plot of the forecasted total inbound call volumes.
    -   A plot showing the components of the forecast, including the trend and weekly and yearly seasonality.
    -  The residuals of the fitted model

The code will also print the mean absolute error (MAE) and mean squared error (MSE) of the forecast to the terminal.