# Install and load packages
#install.packages("forecast")
library(forecast)
#install.packages("tseries")
library(tseries)

# Set your working directory to where your script and
# data files sit on your local computer
setwd("C:\\Your\\Directory\\Here")

# Read csv file of train dataset as a univariate 
# (single variable) series, with datetime 
# (column 1) as the row index
hourly_sentiment_series <- read.csv(file="hourly_users_sentiment_subset.csv",
                                    sep=",",
                                    row.names=1,
                                    header=TRUE)
head(hourly_sentiment_series)

# Check data is indexed with rows/index as the datetime values
rownames(hourly_sentiment_series)

# Preview the data to get an idea of the values and sample size
head(hourly_sentiment_series)
tail(hourly_sentiment_series)
dim(hourly_sentiment_series)

# Plot the data to check if stationary (constant mean and variance), 
# as many time series models require the data to be stationary
plot(hourly_sentiment_series$users_sentiment_score, type="l", xlab="Datetime", ylab="Users Sentiment Score")

# Difference the data to make it more stationary 
# and plot to check if the data looks more stationary
# Differencing subtracts the next value by the current value
# Best not to over-difference the data, 
# as this could lead to inaccurate estimates
# Make sure to leave no missing values, as this could cause 
# problems when modeling later
hourly_sentiment_series_diff1 <- diff(hourly_sentiment_series$users_sentiment_score)
plot(hourly_sentiment_series_diff1, type="l", xlab="Datetime", ylab="Users Sentiment Score")

hourly_sentiment_series_diff2 = diff(hourly_sentiment_series_diff1)
plot(hourly_sentiment_series_diff2, type="l", xlab="Datetime", ylab="Users Sentiment Score")

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
acf(hourly_sentiment_series_diff2)
pacf(hourly_sentiment_series_diff2)

# Depending on ACF and PACF, create ARMA/ARIMA model 
# with AR and MA terms
# auto.arima will automatically choose best terms
ARMA1model_hourly_sentiment <- auto.arima(hourly_sentiment_series, d=2)
# If the p-value for a AR/MA coef is > 0.05, it's not significant
# enough to keep in the model
# Might want to re-model using only significant terms
ARMA1model_hourly_sentiment

# Predict the next 5 hours (5 time steps ahead), 
# which is the test/holdout set
ARMA1predict_5hourly_sentiment <- predict(ARMA1model_hourly_sentiment, n.ahead=5)
ARMA1predict_5hourly_sentiment

# Back transform so we can compare de-diff'd predicted values 
# with the de-diff'd/original actual values
# This is automatically done when forecasting, so no need to 
# manually de-diff
# Nevertheless, let's demo how we de-transform 2 rounds of diffs
# using cumulative sum with original data given
#diff2 back to diff1
undiff1<-cumsum(c(hourly_sentiment_series_diff1[1],hourly_sentiment_series_diff2))
all(round(hourly_sentiment_series_diff1)==round(undiff1))
#undiff1 back to original data
undiff2<-cumsum(c(hourly_sentiment_series$users_sentiment_score[1],undiff1))
all(round(undiff2,6)==round(hourly_sentiment_series,6)) #Note: very small differences
head(hourly_sentiment_series$users_sentiment_score)
head(undiff2)

# Plot actual vs predicted
# First let's get 2 versions of the time series: 
# All values with the last 5 being actual values 
# All values with last 5 being predicted values
hourly_sentiment_full_actual <- read.csv(file="hourly_users_sentiment_sample.csv",
                                         sep=",",
                                         row.names=1,
                                         header=TRUE)
tail(hourly_sentiment_full_actual)
indx_row_values <- row.names(hourly_sentiment_full_actual)[20:24]
indx_row_values
ARMA1predict_5hourly_sentiment[[1]]
predicted_df <- data.frame(indx_row_values,ARMA1predict_5hourly_sentiment[[1]])
hourly_sentiment_series_df <- read.csv(file="hourly_users_sentiment_subset.csv",
                                       sep=",",
                                       header=TRUE)
predicted_df <- setNames(predicted_df, names(hourly_sentiment_series_df))
hourly_sentiment_full_predicted <- rbind(hourly_sentiment_series_df,predicted_df)
hourly_sentiment_full_predicted <- data.frame(hourly_sentiment_full_predicted, row.names=1)
tail(hourly_sentiment_full_predicted)
# Now let's plot actual vs predicted
plot(hourly_sentiment_full_predicted$users_sentiment_score, type="l", col="orange", xlab="Datetime", ylab="Users Sentiment Score")
lines(hourly_sentiment_full_actual$users_sentiment_score, type="l", col="blue")
legend("topleft", legend=c("predicted", "actual"), col=c("orange", "blue"), lty=1, cex=0.5)

# Calculate the MAE to evaluate the model and see if there's 
# a big difference between actual and predicted values
actual_values_holdout <- hourly_sentiment_full_actual$users_sentiment_score[20:24]
predicted_values_holdout <- hourly_sentiment_full_predicted$users_sentiment_score[20:24]
prediction_errors <- c()
for(i in 1:(length(actual_values_holdout))){
  err <- actual_values_holdout[i] - predicted_values_holdout[i]
  prediction_errors <- append(prediction_errors,err)
}
prediction_errors
mean_absolute_error <- mean(abs(prediction_errors))
mean_absolute_error

# You could also look at RMSE

# Would you accept this model as it is?
# There are a few problems to be aware of:
# Data might not be stationary - even though looked 
# fairly stationary to our judgement, a test would 
# help better determine this

# Test (using Dickey-Fuller test) to check if 2 rounds 
# of differencing resulted in stationary data or not
# Print p-value: 
# If > 0.05 accept the null hypothesis, as the data
# is non-stationary
# If <= 0.05 reject the null hypothesis, as the data
# is stationary
adf.test(hourly_sentiment_series_diff2)

#-Need to better transform these data:
# You could look at stabilizing the variance by applying  
# the cube root for neg and pos values and then 
# difference the data 
#-You might compare models with different AR and MA terms
#-This is a very small sample size of 24 timestamps, 
# so might not have enough to spare for a holdout set 
# To get more use out of your data for training, rolling over time 
# series or timestamps at a time for different holdout sets
# allows for training on more timestamps; doesn't stop the model from 
# capturing the last chunk of timestamps stored in a single holdout set
#-The data only looks at 24 hours in one day
# Would we start to capture more of a trend in hourly sentiment if we 
# collected data over several days?
# How would you go about collecting more data?

# Take on the challenge and further improve this model:
# You have been given a head start, now take this example
# and improve on it!

# To study time series further:
#-Look at model diagnostics
#-Use AIC to search best model parameters 
#-Handle any datetime data issues
#-Try other modeling techniques

# Learn more during a short, intense bootcamp:
# Time Series to be introduced in Data Science Dojo's 
# post bootcamp material
# Data Science Dojo's bootcamp also covers some other key 
# machine learning algorithms and techniques and takes you through 
# the critical thinking process behind many data science tasks
# Check out the curriculum: https://datasciencedojo.com/bootcamp/curriculum/"
