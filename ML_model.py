import time
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import itertools
import holidays
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import os

#onlymetric = anomaly(metricdata,s,metric2,date,country,frequency)
def anomaly(group_data,segment,metric,date,country,frequency):
    group_data.rename(columns={date: 'ds', metric: 'y'}, inplace=True)
    group_data = group_data.replace([np.inf, -np.inf], np.nan)
    group_data = group_data.fillna(0)
    #...........prophet............

    #m = Prophet(yearly_seasonality=True,daily_seasonality=True,weekly_seasonality=True,holidays=h)
    m = Prophet(yearly_seasonality=True,daily_seasonality=True,weekly_seasonality=True)
    #m.add_country_holidays(country_name=country) #prophet dict for particular countries
    m.fit(group_data)
    m.train_holiday_names #checking holidays in trained country

    future_dates = m.make_future_dataframe(periods=0, freq=frequency) #for predicting in future dates #period=0 is for zero extra days prediction
    forecast = m.predict(future_dates) #prediting the m(Which is using prophet model)

    future = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] #yhat is predicted values and yhat_lower and yhat_upper are confidence values
    future = future.iloc[0:len(group_data.index)]
    if segment in group_data.columns: #segment='only'
        future[segment] = group_data[segment].values #passing each value into segment
    else:
        segment="only"
    future['y'] = group_data['y'].values #Adding y values from groupdata into future DF
    future['prophet_anomaly']=0 #initially all are zero
    future.loc[future['y'] > future['yhat_upper'], 'prophet_anomaly'] = 1 #higher than higher confidence means 1
    future.loc[future['y'] < future['yhat_lower'], 'prophet_anomaly'] = -1 #lower than lower confidence means -1
    future['importance']=0
    interval = future['yhat_upper']-future['yhat_lower']
    future.loc[future['prophet_anomaly']==1,'importance']=(future['y']-future['yhat_upper'])/interval
    future.loc[future['prophet_anomaly']==-1,'importance']=(future['yhat_lower']-future['y'])/interval
    future['prophet_anomaly'] = future['prophet_anomaly'].replace(1, -1)
    future['prophet_anomaly'] = future['prophet_anomaly'].replace(0, 1)

    #............isolation forest..........
    #for abnormal points isolation predict will give -1
    iso_data = group_data.drop(['ds'], axis = 1) #having only fraud decl percentage

    clf=IsolationForest(n_estimators=100, max_samples='auto',behaviour='new', contamination='auto', \
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0) #if contamination value is 0.1 then we get 10percent anomalies

    clf.fit(iso_data)
    pred = clf.predict(iso_data)
    iso_data['isolation_anomaly']=pred
    #isolation_outlier=iso_data.loc[iso_data['isolation_anomaly']==-1]
    future['isolation_anomaly'] = iso_data['isolation_anomaly'].values

    #......................................

    #..............SARIMAX...........
    #group_data.to_csv('SarimaxCSV.csv')
    group_data.drop(group_data.columns.difference(['ds','y']), 1, inplace=True)
    group_data['ds'] = pd.to_datetime(group_data['ds'])
    #group_data = group_data.sort_values(by = ['ds'])

    group_data.index = pd.DatetimeIndex(freq='H',start=group_data['ds'].iloc[0],periods=len(group_data.ds))
    group_data.index.name = 'index'
    #y.tail()
    group_data['ds'] = group_data.index

    train = group_data[:-1]
    train_se = pd.Series(data = train['y'], index = train['ds'] )

    test = group_data[-1:]
    test_se = pd.Series(data = test['y'], index = test['ds'] )
    p = range(0,2)
    d = range(0,2)
    q = range(0,2)
    pdq = list(itertools.product(p, d, q))

    seasonal = 1
    seasonal_pdq = [(x[0], x[1], x[2], seasonal) for x in list(itertools.product(p, d, q))]
    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    tmp_model = None
    best_mdl = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                tmp_mdl = sm.tsa.statespace.SARIMAX(train_se,
                                                order = param,
                                                seasonal_order = param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=False)

                res = tmp_mdl.fit()
                #print(param,param_seasonal,res.aic)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_mdl = tmp_mdl

            except:
                #print(param,param_seasonal, "Unexpected error:", sys.exc_info()[1])
                continue

    #print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))

    
    
    
    mdl = sm.tsa.statespace.SARIMAX(train_se,
                                    order=(best_pdq),
                                    seasonal_order=(best_seasonal_pdq),
                                    enforce_stationarity=True,
                                    enforce_invertibility=False)
    res = mdl.fit()

    #print(group_data)
    string =group_data['ds'].iloc[-2]

    forecast = res.get_prediction(start = group_data['ds'].iloc[-2],dynamic = False)

    forecast = res.get_forecast(steps=1)
    ci = forecast.conf_int()

    ci['overall_percent'] = test_se
    ci['anomaly'] = 1
    ci.loc[ci['overall_percent'] > ci['upper y'], 'anomaly'] = -1
    ci.loc[ci['overall_percent'] < ci['lower y'], 'anomaly'] = -1
    #print(ci)
    #print(future)
    future['sa'] = 0
    future['sa'].iloc[-1]=ci['anomaly'].iloc[0]

    future['SUP'] = 0
    future['SLP'] = 0
    future['SUP'].iloc[-1] = ci['upper y'].iloc[0]
    future['SLP'].iloc[-1] = ci['lower y'].iloc[0]

    #-------------------------------------------Ensemble-------------------------------------------

    future['common_points'] = 0
    eub = 0
    elb = 0
    plb = future['yhat_lower'].iloc[-1]
    pub = future['yhat_upper'].iloc[-1]
    slb = ci['lower y'].iloc[-1]
    sub = ci['upper y'].iloc[-1]
    act = future['y'].iloc[-1]

    if(future['prophet_anomaly'].iloc[-1] == future['sa'].iloc[-1]):
        eub = (pub * 0.5) + (sub * 0.5)
        elb = (plb * 0.5) + (slb * 0.5)
        future['common_points'].iloc[-1] = future['prophet_anomaly'].iloc[-1]

    elif(future['prophet_anomaly'].iloc[-1] != future['sa'].iloc[-1]):
        if(future['prophet_anomaly'].iloc[-1] == future['isolation_anomaly'].iloc[-1]):

            if(frequency == 'D'):
                eub = (pub * 0.6) + (sub * 0.4)
                elb = (plb * 0.6) + (slb * 0.4)
            else:
                eub = (pub * 0.4) + (sub * 0.6)
                elb = (plb * 0.4) + (slb * 0.6)

            if((act>eub) or (act<elb)):
                future['common_points'].iloc[-1] = -1
            else:
                future['common_points'].iloc[-1] = 1
        elif(future['sa'].iloc[-1] == future['isolation_anomaly'].iloc[-1]):
            if(frequency == 'D'):
                eub = (pub * 0.4) + (sub * 0.6)
                elb = (plb * 0.4) + (slb * 0.6)
            else:
                eub = (pub * 0.3) + (sub * 0.7)
                elb = (plb * 0.3) + (slb * 0.7)

            if((act>eub) or (act<elb)):
                future['common_points'].iloc[-1] = -1
            else:
                future['common_points'].iloc[-1] = 1

    #-------------------------------------------Severity Importance-------------------------------------------
    if(future['common_points'].iloc[-1]==-1):
        if(act>eub):
            future['importance'].iloc[-1] = abs(((act-eub)/eub)*100)
        elif(act<elb):
            future['importance'].iloc[-1] = abs(((act-elb)/elb)*100)
    future['type2Imp'] = 0
    if(future['common_points'].iloc[-1]==-1):
        if(act>eub):
            future['type2Imp'].iloc[-1] = abs(act-eub)
        elif(act<elb):
            future['type2Imp'].iloc[-1] = abs(act-elb)
    future['EUB'] = 0
    future['ELB'] = 0
    future['EUB'].iloc[-1] = eub
    future['ELB'].iloc[-1] = elb
    return future