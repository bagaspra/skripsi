import pandas as pd
from matplotlib import pyplot
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statistics

# Read csv file of our train dataset as a univariate 
# (single variable) series, with datetime (column 0) 
# as the row index
hourly_sentiment_series = pd.read_csv('hourly_users_sentiment_subset.csv', 
                                      index_col=0,
                                      parse_dates=True,
                                      squeeze=True)
print(hourly_sentiment_series.head())

# Check data is indexed as DatetimeIndex
print(hourly_sentiment_series.index)

# Preview the data to get an idea of the values and sample size
print(hourly_sentiment_series.head())
print(hourly_sentiment_series.tail())
print(hourly_sentiment_series.shape)

# Plot the data to check if stationary (constant mean and variance), 
# as many time series models require the data to be stationary
pyplot.plot(hourly_sentiment_series)
pyplot.show()

# Difference the data to make it more stationary 
# and plot to check if the data looks more stationary
# Differencing subtracts the next value by the current value
# Best not to over-difference the data, 
# as this could lead to inaccurate estimates
# Make sure to leave no missing values, as this could cause 
# problems when modeling later
hourly_sentiment_series_diff1 = hourly_sentiment_series.diff().fillna(hourly_sentiment_series)
pyplot.plot(hourly_sentiment_series_diff1)
pyplot.show()

hourly_sentiment_series_diff2 = hourly_sentiment_series_diff1.diff().fillna(hourly_sentiment_series_diff1)
pyplot.plot(hourly_sentiment_series_diff2)
pyplot.show()

# Check ACF and PACF plots to determine number of AR terms and 
# MA terms in ARMA model, or to spot seasonality/periodic trend
# Autoregressive forecast the next timestamp's value by
# regressing the previous values
# Moving Average forecast the next timestamp's value by
# averaging the previous values 
# Autoregressive Integrated Moving Average is useful 
# for non-stationary data, plus has an additional seasonal 
# differencing parameter for seasonal non-stationary data
# ACF and PACF plots include 95% Confidence Interval bands
# Anything outside of the CI shaded bands is a 
# statistically significant correlation 
# If we see a significant spike at lag x in the ACF 
# that helps determine the number of MA terms
# If we see a significant spike at lag x in the PACF 
# that helps us determine the number of AR terms
plot_acf(hourly_sentiment_series_diff2)
pyplot.show()

plot_pacf(hourly_sentiment_series_diff2)
pyplot.show()

# Depending on ACF and PACF, create ARMA/ARIMA model 
# with AR and MA terms
# This will infer the frequency, so make sure there are 
# no gaps between datetimes
ARMA1model_hourly_sentiment = ARIMA(hourly_sentiment_series, order=(5,2,1)).fit(transparams=False)
# If the p-value for a AR/MA coef is > 0.05, it's not significant
# enough to keep in the model
# Might want to re-model using only significant terms
print(ARMA1model_hourly_sentiment.summary())

# Predict the next 5 hours (5 time steps ahead), 
# which is the test/holdout set
ARMA1predict_5hourly_sentiment = ARMA1model_hourly_sentiment.predict('2/6/2019  7:00:00 PM','2/6/2019  11:00:00 PM', typ='levels')
print('Forecast/preditions for 5 hours ahead ', ARMA1predict_5hourly_sentiment)

# Back transform so we can compare de-diff'd predicted values 
# with the de-diff'd/original actual values
# This is automatically done when predicting (specify typ='levels'), 
# so no need to manually de-diff
# Nevertheless, let's demo how we de-transform 2 rounds of diffs
# using cumulative sum with original data given
#diff2 back to diff1
undiff1 = hourly_sentiment_series_diff2.cumsum().fillna(hourly_sentiment_series_diff2)
#undiff1 back to original data
undiff2 = undiff1.cumsum().fillna(undiff1)
print(all(round(hourly_sentiment_series,6)==round(undiff2,6))) # Note: very small differences
print('Original values ', hourly_sentiment_series.head())
print('De-differenced values ', undiff2.head())

# Plot actual vs predicted
# First let's get 2 versions of the time series: 
# All values with the last 5 being actual values 
# All values with last 5 being predicted values
hourly_sentiment_full_actual = pd.read_csv('hourly_users_sentiment_sample.csv',
                                           index_col=0, 
                                           parse_dates=True,
                                           squeeze=True)
print(hourly_sentiment_full_actual.tail())
indx_row_values = hourly_sentiment_full_actual.index[19:24]
print(indx_row_values)
predicted_series_values = pd.Series(ARMA1predict_5hourly_sentiment, 
                                    index=['2019-02-06 19:00:00',
                                           '2019-02-06 20:00:00',
                                           '2019-02-06 21:00:00',
                                           '2019-02-06 22:00:00',
                                           '2019-02-06 23:00:00'])
print(predicted_series_values)
hourly_sentiment_full_predicted = hourly_sentiment_series.append(predicted_series_values)
print(hourly_sentiment_full_predicted.tail())
# Now let's plot actual vs predicted
pyplot.plot(hourly_sentiment_full_predicted, c='orange', label='predicted')
pyplot.plot(hourly_sentiment_full_actual, c='blue', label='actual')
pyplot.legend(loc='upper left')
pyplot.show()

# Calculate the MAE to evaluate the model and see if there's 
# a big difference between actual and predicted values
actual_values_holdout = hourly_sentiment_full_actual.iloc[19:24]
predicted_values_holdout = hourly_sentiment_full_predicted.iloc[19:24]
prediction_errors = []
for i in range(len(actual_values_holdout)):
    err = actual_values_holdout[i]-predicted_values_holdout[i]
    prediction_errors.append(err)

print('Prediction errors ', prediction_errors)
mean_absolute_error = statistics.mean(map(abs, prediction_errors))
print('Mean absolute error ', mean_absolute_error)

# You could also look at RMSE

# Would you accept this model as it is?
# There are a few problems to be aware of:
# Data might not be stationary - even though looked 
# fairly stationary to our judgement, a test would 
# help better determine this

# Test (using Dickey-Fuller test) to check if 2 rounds 
# of differencing resulted in stationary data or not
test_results = adfuller(hourly_sentiment_series_diff2)
# Print p-value: 
# If > 0.05 accept the null hypothesis, as the data
# is non-stationary
# If <= 0.05 reject the null hypothesis, as the data
# is stationary
print('p-value ', test_results[1])

'''
-Need to better transform these data:
 You could look at stabilizing the variance by applying  
 the cube root for neg and pos values and then 
 difference the data 
-You might compare models with different AR and MA terms
-This is a very small sample size of 24 timestamps, 
 so might not have enough to spare for a holdout set 
 To get more use out of your data for training, rolling over time 
 series or timestamps at a time for different holdout sets
 allows for training on more timestamps; doesn't stop the model from 
 capturing the last chunk of timestamps stored in a single holdout set
-The data only looks at 24 hours in one day
 Would we start to capture more of a trend in hourly sentiment if we 
 collected data over several days?
 How would you go about collecting more data?

 Take on the challenge and further improve this model:
 You have been given a head start, now take this example
 and improve on it!

 To study time series further:
-Look at model diagnostics
-Using AIC to search best model parameters 
-Handle any datetime data issues
-Try other modeling techniques

 Learn more during a short, intense bootcamp:
 Time Series to be introduced in Data Science Dojo's 
 post bootcamp material
 Data Science Dojo's bootcamp also covers some other key 
 machine learning algorithms and techniques and takes you through 
 the critical thinking process behind many data science tasks
 Check out the curriculum: https://datasciencedojo.com/bootcamp/curriculum/
'''
