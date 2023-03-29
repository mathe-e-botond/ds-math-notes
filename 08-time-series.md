# **8 Time series analysis and forecasting**

In this chapeter we will explore time series analysis and forecasting techniques.


## **8.1 SARIMAX**

SARIMAX is a commonly applied time series analysis technique, it is actually a combination of multiple techiques:

* AR: Autoregression
* MA: Moving Average
* I: Adding differencing to ARMA
* S: Seasanility added it ARIMA
* X: External parameter added (moving from single variate to multi variate)

## **8.1.1 The AR model**

The AR model assumes a relationship in the time series between a point in time and a given lag. This can be quantified by calculating the correlation between the time series and it's shifted version (trimming the edges). This concept is called autocorrelation. 

# References:

**Advanced forecasting with Python**, 2021, Joos Korstanje