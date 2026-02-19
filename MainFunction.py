


import teradata
import getpass
import NextGen.Daily.sparklines as sparklines
from NextGen.Daily.emailFile import send_email
from NextGen.Daily.email_error import error
from NextGen.Daily.email_dataframe import dataframe_mail
from NextGen.Daily.ML_models import anomaly
from NextGen.Daily.RCA_import import percent_rca_ttb, percent_rca_btt, metric_rca_ttb, metric_rca_btt
from NextGen.Daily.LinkReading import read_link
from NextGen.Daily.pattern_identifier import hawk_pattern       ## ADDED
from OutputFile import rda_query
import datetime
import holidays
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
import multiprocessing
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import matplotlib.dates as dates
import matplotlib.dates as mpl_dates
import matplotlib.patheffects as path_effects
from scipy.interpolate import make_interp_spline
import os
import time
import copy
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

@wrap_non_picklable_objects
def main_function(data1, segment_name, date, dictionary, metric_list_percent, metric_list_units, overall_df, date_data, numerator, denominator, country, frequency):
    output = []
    segment = segment_name
    df_list = []
    df_list.append(segment)
    df_list.extend(metric_list_units)
    df_list.append(date)
    data = data1.copy()
    data.drop(data.columns.difference(df_list), 1, inplace=True)
    d = dict(tuple(data.groupby([date])))
    appended_data = []
    for key in d:
        group_data1 = d[key]
        group_data1 = pd.DataFrame(group_data1.groupby(segment)[metric_list_units].apply(lambda x: x.astype(int).sum()))
        group_data1[segment] = group_data1.index
        group_data1['date'] = key
        columns = group_data1.columns.tolist()
        columns = columns[::-1]
        group_data1 = group_data1[columns]
        appended_data.append(group_data1)
    appended_data = pd.concat(appended_data)
    appended_data = pd.merge(date_data, appended_data, on=date, how='left')
    f = pd.Series(appended_data[segment].dropna().unique())
    for i in f:
        k = i
    appended_data[segment] = appended_data[segment].fillna(k)
    group_data = appended_data.copy()

    metric_anomaly_list = []
    RCAremoval = []
    for a in metric_list_percent:
        group_data[a+'_eligible_percent'] = (group_data[numerator[a]]/group_data[denominator[a]])*100
        group_data[a+'_global_percent'] = (group_data[numerator[a]]/overall_df[denominator[a]])*100
        metric_anomaly_list.append(a+'_eligible_percent')
        metric_anomaly_list.append(a+'_global_percent')
        RCAremoval.append(a+'_global_percent')
    for a in metric_list_units:
        group_data[a+'_percent'] = (group_data[a]/overall_df[a])*100
        metric_anomaly_list.append(a+'_percent')
        RCAremoval.append(a+'_percent')
        metric_anomaly_list.append(a)

    group_data = group_data.fillna(0)

    for a in metric_anomaly_list:
        g = group_data.copy()
        g.drop(g.columns.difference([date, segment, a]), 1, inplace=True)
        metric = a
        sub = anomaly(g, segment, metric, date, country, frequency) #Individual anomaly segment wise global
        if(metric in RCAremoval):
            # Filtering RCA for Global percent and Percentage Contr
            if(sub['common_points'].iloc[-1]==1):
                impor = sub['importance'].iloc[-1]
                if(impor < 35):
                    sub['common_points'].iloc[-1]=1
        sub.rename(columns = {'yhat':a+'_yhat'}, inplace = True)
        sub.rename(columns = {'yhat_lower':a+'_yhat_lower'}, inplace = True)
        sub.rename(columns = {'yhat_upper':a+'_yhat_upper'}, inplace = True)
        sub.rename(columns = {'prophet_anomaly':a+'_prophet_anomaly'}, inplace = True)
        sub.rename(columns = {'isolation_anomaly':a+'_isolation_anomaly'}, inplace = True)
        sub.rename(columns = {'sa':a+'_SarimaxAnom'}, inplace = True)
        sub.rename(columns = {'SUP':a+'_SUP'}, inplace = True)
        sub.rename(columns = {'SLP':a+'_SLP'}, inplace = True)
        sub.rename(columns = {'EUB':a+'_EUB'}, inplace = True)
        sub.rename(columns = {'ELB':a+'_ELB'}, inplace = True)
        sub.rename(columns = {'y':'y_metric'}, inplace = True)
        sub.rename(columns = {'ds':'date'}, inplace = True)
        sub.rename(columns = {'common_points':a+'_anomaly'}, inplace = True)
        sub.rename(columns = {'importance':a+'_importance'}, inplace = True)
        sub['segment_name'] = segment

        output.append(sub)
    return output

def start_analysis(data1, date, date_pi, country,metric_list, metric_list_percent, metric_list_units, metric_list_num, numerator, denominator, cc, extra, dataframe_subj, frequency, df_last, overall_df, namingdict, emailInsight, metrictype, product1, favlist, campaignstat, date_date, segment_headings, dictionary, dataframe, dashboardlink, x, highRiskMailDL, MedRiskMailDL, NoAnomalyDL, isDollar, nan_threshold, target_col_list, pruned_path):
    #df_pattern
    discard = ['?', '0', '###', 'NULL', 'null', 'Null', 'unknown', 'Unknown', 'UNKNOWN']
    
    #------for percentage---------
    
    file_list = []
    overall_anomaly = {}
    severitylist = []
    anomaly_flag_dictionary = {}

    def overall_function(data1, metric_list):
        for a in metric_list:
            print('---------------for the metric:', a, '------------------')
            metricdata = data1.copy()
            metricdata.drop(metricdata.columns.difference([a, date]), 1, inplace=True)
            #print(metricdata)
            metric = a
            s = "only"

            #print(metricdata)
            onlymetric = anomaly(metricdata, s, metric, date, country, frequency)
            onlymetric.rename(columns={'y':'overall_'+metric}, inplace=True)
            onlymetric.rename(columns={'ds':'date'}, inplace=True)

            #value1 = (onlymetric['SUP'].iloc[-1] + onlymetric['SLP'].iloc[-1])/2
            #value2 = onlymetric['overall_'+metric].iloc[-1]
            #perchange = ((abs(value1-value2))/((value1+value2)/2))*100

            imp = onlymetric['importance'].iloc[-1]
            print('Actual Value:', onlymetric['overall_'+metric].iloc[-1])
            print('Anomaly or Not:', onlymetric['common_points'].iloc[-1])
            print('Importance:', onlymetric['importance'].iloc[-1])
            if(onlymetric['common_points'].iloc[-1] == 1):
                if ((onlymetric['overall_'+metric].iloc[-1] < 30) and (onlymetric['overall_'+metric].rolling(window=7).mean().shift(1).iloc[-1] < 30)) and (isDollar!='Y'):
                    
                    severity = 'LOW'
                elif (imp >= 60):
                    severity = "HIGHLY CRITICAL"
                elif (imp >= 30) & (imp < 60):
                    severity = "CRITICAL"
                elif (imp >= 10) & (imp < 30):
                    severity = "HIGH"
                elif (imp >= 5) & (imp < 10):
                    severity = "MEDIUM"
                elif (imp < 5):
                    severity = 'LOW'
                anomaly_flag_dictionary[a] = 'True'

            elif(onlymetric['common_points'].iloc[-1] == 1):
                severity = 'No Anomaly'
                anomaly_flag_dictionary[a] = 'False'
            
            onlymetric.rename(columns={'common_points':'overall_'+a+'_anomaly'}, inplace=True)
            onlymetric.rename(columns={'importance':'overall_'+a+'_importance'}, inplace=True)

            filename = "overall_"+a+"_data.csv"
            onlymetric.to_csv(filename)
            file_list.append(filename)

            onlymetric.drop(['prophet_anomaly', 'isolation_anomaly', 'yhat', 'yhat_lower', 'yhat_upper'], axis = 1)
            overall_anomaly[a] = onlymetric
            severitylist.append(severity)

        return overall_anomaly

    overall_anomalyreport = overall_function(overall_df, metric_list)
    severitydict = {metric_list[i]: severitylist[i] for i in range(len(metric_list))}
    print(severitylist)
    t_start = time.time()

    result = {}
    
    totalThreads = 0
    ## -----------------------------CONDITION TO SKIP THE LOW, NO ANOMALY RCA PROCESS (for optimization)----------------------------- ##
    if (set(severitylist) == set(['No Anomaly'])) or (set(severitylist) == set(['LOW'])) or (set(severitylist) == set(['No Anomaly', 'LOW'])):
        dataframe = {}

    #dataframe = {'channel': [dataframefor web, dataframe for mobile]}
    # {'email_tag': [gold dataframe, silver......]}
    for key in dataframe:
        
        combine = []
        group = dataframe[key]
        n = len(pd.unique(data1[key]))
        print("Values in segment:", n)
        cpu = multiprocessing.cpu_count()
        trd = 0
        if(cpu >= n):
            trd = n
        elif(cpu < n):
            if(cpu > 6):
                trd = cpu/2
            else:
                trd = 1
        
        print("Thread:", trd)
        totalThreads += trd
        trd1 = int(trd)
        print("Converted Thread:", trd1)
        segment_results_percent = Parallel(n_jobs=-1)(delayed(main_function)(data1, key, date, dictionary, metric_list_percent, metric_list_units, overall_df, date_data, numerator, denominator, country, frequency) for data1 in group if len(data1.index) >= 2)

        if len(segment_results_percent) == 0:
            continue
        else:
            for i in segment_results_percent:
                j = pd.concat(i, axis=1) #output is list so to make into DF
                j = j.loc[:,~j.columns.duplicated()] #remove repeated columns
                for e in metric_list_units:
                    j[e+'_roll7'] = j[e].rolling(window=7).mean().shift(periods=1)
                    j[e+'_roll30'] = j[e].rolling(window=30).mean().shift(periods=1)
                    j[e+'_change_7'] = ((j[e] - j[e+'_roll7'])/j[e+'_roll7']).multiply(100).round(2)
                    j[e+'_change_30'] = ((j[e] - j[e+'_roll30'])/j[e+'_roll30']).multiply(100).round(2)
                for m in metric_list_percent:
                    j[m+'_roll7'] = j[m+'_eligible_percent'].rolling(window=7).mean().shift(periods=1)
                    j[m+'_roll30'] = j[m+'_eligible_percent'].rolling(window=30).mean().shift(periods=1)
                    j[m+'_change_7'] = ((j[m+'_eligible_percent'] - j[m+'_roll7'])/j[m+'_roll7']).multiply(100).round(2)
                    j[m+'_change_30'] = ((j[m+'_eligible_percent'] - j[m+'_roll30'])/j[m+'_roll30']).multiply(100).round(2)
                combine.append(j)
            
            start = pd.concat(combine)
            start = start.sort_values(by=[date, key])
            result[key] = start
            print("parallel processing time taken: {.3f}s".format(time.time() - t_start)) #Time calculation....

    
    for a in result: #each 'a' is each key in result
        df = result[a]
        for key in dictionary:
            if(key in df.columns):
                df[key] = df[key].astype(int)
                n = key
                group = dictionary[key]
                rename_dict = group.set_index(key).to_dict()['name'] #{0: 'FORCE_SIGNUP', 1: 'GUEST', 2: 'MEMBER'}
                df[n] = df[n].replace(rename_dict) #All labels replaced by names for each key

    
    r = copy.deepcopy(result)
    rmail = copy.deepcopy(result)
    for key in r:
        r[key] = r[key].rename(columns={key: 'segment_value'})
    for key in rmail:
        rmail[key] = rmail[key].rename(columns={key: 'segment_value'})
    
    percent_result = []
    for dk in metric_list:
        maildataoverall = overall_anomalyreport[dk]
        filename = dk+"_data.csv"
        maildataoverall.to_csv(filename)
        file_list.append(filename)
    
    rca_df = pd.DataFrame()
    for a in rmail:
        rcadf = rmail[a].copy()
        filename = a+"_data.csv"
        rcadf.to_csv(filename)
        rca_df = pd.concat([rca_df, rcadf])
        file_list.append(filename)
    
    if(extra == 'Merchant'):
        dataframe_mail(file_list, dataframe_subj, NoAnomalyDL)

   
    
    for a in r:
        rcadf = r[a].copy()
        for dk in metric_list:
            rcadf = pd.merge(rcadf, overall_anomalyreport[dk], on=date)
            print('-------------percent_result-----------------')
            percent_result.append(rcadf)
    
    myImages = []
    mainlist = []
    for name in metric_list_percent:
        internallist = []
        print('-------------------------------------------percent LOOP-------------------------------------------')
        titlepercent = namingdict[name]
        internallist.append(titlepercent)
        label = 'overall_'+name+'_anomaly'
        driver_percent_ttb, driver_percent_ttb_table = percent_rca_ttb(percent_result, date, dictionary, segment_headings, discard, label, name)
        driver_percent_btt, driver_percent_btt_table = percent_rca_btt(percent_result, date, dictionary, segment_headings, discard, label, name)

        if((driver_percent_ttb.empty) or (x not in driver_percent_ttb[date].unique())):
            
            if((driver_percent_btt.empty) or (x not in driver_percent_btt[date].unique())):
                percent_data_table = ""
            
            elif(x in driver_percent_btt[date].unique()):
                print('came')
                x_data_percent = driver_percent_btt[driver_percent_btt[date]==x] #data of last date
                x_data_percent = x_data_percent.loc[~x_data_percent['segment_value'].isin(discard)]

                percent_data_table = driver_percent_btt_table[driver_percent_btt_table[date]==x]

                percent_data_table = percent_data_table.loc[~percent_data_table['segment_value'].isin(discard)] #In null value or unkown we are removing
                percent_data_table.rename(columns={'segment_name': 'Driver'}, inplace=True)
                percent_data_table.rename(columns={'segment_value': 'Value'}, inplace=True)
                percent_data_table.rename(columns={denominator[name]: namingdict[denominator[name]]+'[Dr]'}, inplace=True)
                percent_data_table.rename(columns={name+'_eligible_percent': namingdict[name]}, inplace=True)
                percent_data_table.rename(columns={numerator[name]: namingdict[numerator[name]]+'[Nr]'}, inplace=True)
                percent_data_table.rename(columns={name+'_roll7': 'Rolling 7days'}, inplace=True)
                percent_data_table.rename(columns={name+'_roll30': 'Rolling 30days'}, inplace=True)
                percent_data_table.rename(columns={name+'_change_7': '7Days Delta'}, inplace=True)
                percent_data_table.rename(columns={name+'_change_30': '30Days Delta'}, inplace=True)

                percent_data_table[namingdict[denominator[name]]+'[Dr]'] = percent_data_table[namingdict[denominator[name]]+'[Dr]'].astype(int)
                percent_data_table[namingdict[numerator[name]]+'[Nr]'] = percent_data_table[namingdict[numerator[name]]+'[Nr]'].astype(int)
                percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].round(2)
                percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].round(2)
                percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].map(lambda x: 0 if x==(np.inf) else x)
                percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].map(lambda x: 0 if x==(np.inf) else x)

                ## ----------------------------- NEW CODE LOGIC: RCA optimization ----------------------------- ##
                drop_indices = []
                for i in range(len(percent_data_table)):
                    chg_per_roll30 = abs(percent_data_table[namingdict[numerator[name]]+'[Nr]'].iloc[i] - percent_data_table[numerator[name]+'_roll30'].iloc[i])
                    upper_deviation = percent_data_table[numerator[name]+'_roll30'].iloc[i] + (percent_data_table[numerator[name]+'_roll30'].iloc[i])*0.2
                    lower_deviation = percent_data_table[numerator[name]+'_roll30'].iloc[i] - (percent_data_table[numerator[name]+'_roll30'].iloc[i])*0.2
                    if ((chg_per_roll30 < 20) or (lower_deviation < percent_data_table[namingdict[numerator[name]]+'[Nr]'].iloc[i] < upper_deviation)):
                        drop_indices.append(percent_data_table.iloc[i].name)
                
                if len(percent_data_table)>10:
                    percent_data_table = percent_data_table.drop(labels=drop_indices)
                
                cols = ['Driver', 'Value', namingdict[denominator[name]]+'[Dr]', namingdict[numerator[name]]+'[Nr]', namingdict[name], 'Rolling 7days', '7Days Delta', 'Rolling 30days', '30Days Delta']
                percent_data_table = percent_data_table[cols]
                #--------------------------Spark lines--------------------------
                df_ref = percent_data_table[['Value', 'Driver']]
                df_charts = pd.DataFrame()
                for i in range(len(df_ref)):
                    df_vals = df_ref.iloc[i].tolist()
                    df_trend = rca_df.loc[(rca_df.segment_value==df_vals[0]) & (rca_df.segment_name==df_vals[1])]
                    df_trend = df_trend[[date]+[numerator[name]]].groupby(date).sum().reset_index().T
                    df_trend.columns = df_trend.iloc[0,:]
                    df_trend = df_trend.iloc[1:, -15:]
                    if df_trend.empty:
                        df_charts = pd.concat([df_charts, pd.Series('-')])
                        continue
                    df_trend['sparklines'], _ = sparklines.create(data=df_trend, rca=1)
                    df_charts = pd.concat([df_charts, df_trend['sparklines']])

                df_charts.index = percent_data_table.index
                try:
                    percent_data_table['Trend lines'] = df_charts
                    del df_charts, df_trend
                except ValueError:
                    pass

                #--------------------------Sorting--------------------------
                pdtPos = percent_data_table[percent_data_table['7Days Delta']>=0]
                pdtPos = pdtPos.sort_values(['7Days Delta'], ascending=[False])
                pdtPos['7Days Delta'] = pdtPos['7Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+'%')
                pdtPos['30Days Delta'] = pdtPos['30Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+'%' if x>=0 else '\u25BC'+str(abs(x))+'%')
                print('cam3')
                pdtNeg = percent_data_table[percent_data_table['7Days Delta']<0]
                pdtNeg = pdtNeg.sort_values(['7Days Delta'], ascending=[True])
                pdtNeg['7Days Delta'] = pdtNeg['7Days Delta'].apply(lambda x: '\u25BC'+str(abs(x))+'%')
                pdtNeg['30Days Delta'] = pdtNeg['30Days Delta'].apply(lambda x: '\u25BC'+str(abs(x))+'%' if x<0 else '\u25B2'+str(abs(x))+'%')
                if(favlist[name] == 'u'):
                    percent_data_table = pd.concat([pdtPos, pdtNeg], axis=0)
                else:
                    percent_data_table = pd.concat([pdtNeg, pdtPos], axis=0)
                print('cam4')
                #--------------------------Formatting--------------------------
                percent_data_table[[namingdict[name]]] = percent_data_table[[namingdict[name]]].apply(pd.to_numeric)
                percent_data_table = percent_data_table.round(2)
                percent_data_table[namingdict[name]] = percent_data_table[namingdict[name]].astype(str)
                percent_data_table[namingdict[name]] = percent_data_table[namingdict[name]].apply(lambda x: '&nbsp&nbsp'+x+'%' if len(x)==(percent_data_table[namingdict[name]].map(lambda x: len(x)).max()) else '&nbsp&nbsp'+x+(((percent_data_table[namingdict[name]].map(lambda x: len(x)).max())-len(x))*'&nbsp&nbsp')+'%')
                print('complete1')
        
                percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].apply(lambda x: ' 0.00%' if('e' in x) else x)
                percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].apply(lambda x: ' 0.00%' if('e' in x) else x)

                #--------------------------Equal arrow spacing--------------------------
                percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].apply(lambda x: x if len(x)==(percent_data_table['7Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((percent_data_table['7Days Delta'].map(lambda x: len(x)).max())-len(x))*'&nbsp&nbsp')+x[1:])
                percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].apply(lambda x: x if len(x)==(percent_data_table['30Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((percent_data_table['30Days Delta'].map(lambda x: len(x)).max())-len(x))*'&nbsp&nbsp')+x[1:])

                #--------------------------
                if(favlist[name] == 'u'):
                    percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].map(lambda x: f'<font color="green">'+x if x[0]=='\u25B2' else f'<font color="red">'+x)
                    percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].map(lambda x: f'<font color="green">'+x if x[0]=='\u25B2' else f'<font color="red">'+x)
        
                else:
                    percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].map(lambda x: f'<font color="red">'+x if x[0]=='\u25B2' else f'<font color="green">'+x)
                    percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].map(lambda x: f'<font color="red">'+x if x[0]=='\u25B2' else f'<font color="green">'+x)

                percent_data_table['Rolling 7days'] = percent_data_table['Rolling 7days'].astype(str)
                percent_data_table['Rolling 7days'] = percent_data_table['Rolling 7days'].apply(lambda x: '&nbsp&nbsp'+x+'%' if len(x)==(percent_data_table['Rolling 7days'].map(lambda x: len(x)).max()) else '&nbsp&nbsp'+x+(((percent_data_table['Rolling 7days'].map(lambda x: len(x)).max())-len(x))*'&nbsp&nbsp')+'%')
                percent_data_table['Rolling 30days'] = percent_data_table['Rolling 30days'].astype(str)
                percent_data_table['Rolling 30days'] = percent_data_table['Rolling 30days'].apply(lambda x: '&nbsp&nbsp'+x+'%' if len(x)==(percent_data_table['Rolling 30days'].map(lambda x: len(x)).max()) else '&nbsp&nbsp'+x+(((percent_data_table['Rolling 30days'].map(lambda x: len(x)).max())-len(x))*'&nbsp&nbsp')+'%')
                if(isDollar == 'Y'):
                    percent_data_table[namingdict[denominator[name]]+'[Dr]'] = (percent_data_table[namingdict[denominator[name]]+'[Dr]']/1000).round(2).astype(str)+'K'
                    percent_data_table[namingdict[numerator[name]]+'[Nr]'] = (percent_data_table[namingdict[numerator[name]]+'[Nr]']/1000).round(2).astype(str)+'K'
                else:
                    percent_data_table[namingdict[denominator[name]]+'[Dr]'] = (percent_data_table[namingdict[denominator[name]]+'[Dr]']).astype(int)
                    percent_data_table[namingdict[numerator[name]]+'[Nr]'] = (percent_data_table[namingdict[numerator[name]]+'[Nr]']).astype(int)        
                percent_data_table = percent_data_table.reset_index(drop=True)


        elif(x in driver_percent_ttb[date].unique()):

            x_data_percent = driver_percent_ttb[driver_percent_ttb[date]==x]
            x_data_percent = x_data_percent.loc[~x_data_percent['segment_value'].isin(discard)]

            percent_data_table = driver_percent_ttb_table[driver_percent_ttb_table[date]==x]
            percent_data_table = percent_data_table.loc[~percent_data_table['segment_value'].isin(discard)]

            percent_data_table.rename(columns={'segment_name': 'Driver'}, inplace=True)
            percent_data_table.rename(columns={'segment_value': 'Value'}, inplace=True)
            percent_data_table.rename(columns={denominator[name]: namingdict[denominator[name]]+' [Dr]'}, inplace=True)
            percent_data_table.rename(columns={'eligible_percent': namingdict[name]}, inplace=True)
            percent_data_table.rename(columns={numerator[name]: namingdict[numerator[name]]+' [Nr]'}, inplace=True)
            percent_data_table.rename(columns={'name'+'_roll7': 'Rolling 7days'}, inplace=True)
            percent_data_table.rename(columns={'name'+'_roll30': 'Rolling 30days'}, inplace=True)
            percent_data_table.rename(columns={'name'+'_change_7': '7Days Delta'}, inplace=True)
            percent_data_table.rename(columns={'name'+'_change_30': '30Days Delta'}, inplace=True)

            percent_data_table[namingdict[denominator[name]]+' [Dr]'] = percent_data_table[namingdict[denominator[name]]+' [Dr]'].astype(int)
            percent_data_table[namingdict[numerator[name]]+' [Nr]'] = percent_data_table[namingdict[numerator[name]]+' [Nr]'].astype(int)
            percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].round(2)
            percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].round(2)
            percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].map(lambda x: 0 if x==(np.inf) else x)
            percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].map(lambda x: 0 if x==(np.inf) else x)

            percent_data_table[[namingdict[name]]] = percent_data_table[[namingdict[name]]].apply(pd.to_numeric)
            percent_data_table = percent_data_table.round(2)

            ## -------------------------- NEW CODE LOGIC: RCA optimization --------------------------
            drop_indices = []
            for i in range(len(percent_data_table)):
                chg_per_roll30 = abs(percent_data_table[namingdict[numerator[name]]+' [Nr]'].iloc[i] - percent_data_table[numerator[name]+'_roll30'].iloc[i])
                upper_deviation = percent_data_table[numerator[name]+'_roll30'].iloc[i] + (percent_data_table[numerator[name]+'_roll30'].iloc[i])*0.2
                lower_deviation = percent_data_table[numerator[name]+'_roll30'].iloc[i] - (percent_data_table[numerator[name]+'_roll30'].iloc[i])*0.2
                if ((chg_per_roll30 < 20) or (lower_deviation < percent_data_table[namingdict[numerator[name]]+' [Nr]'].iloc[i] < upper_deviation)):
                    drop_indices.append(percent_data_table.iloc[i].name)
            
            if len(percent_data_table)>10:
                percent_data_table = percent_data_table.drop(labels=drop_indices)

            #--------------------------Spark lines--------------------------
            df_ref = percent_data_table[['Value', 'Driver']]
            df_charts = pd.DataFrame()
            for i in range(len(df_ref)):
                df_vals = df_ref.iloc[i].tolist()
                df_trend = rca_df.loc[(rca_df.segment_value==df_vals[0]) & (rca_df.segment_name==df_vals[1])]  
                df_trend = df_trend[[date]+[numerator[name]]].groupby(date).sum().reset_index().T     
                df_trend.columns = df_trend.iloc[0,:]
                df_trend = df_trend.iloc[1:,-15:]
                if df_trend.empty:
                    df_charts = pd.concat([df_charts, pd.Series('-')])
                    continue
                df_trend['sparklines'], _ = sparklines.create(data=df_trend, rca=1)
                df_charts = pd.concat([df_charts, df_trend['sparklines']])
            
            df_charts.index = percent_data_table.index
            try:
                percent_data_table['Trend lines'] = df_charts
                del df_charts, df_trend
            except ValueError:
                pass

            percent_data_table[namingdict[name]] = percent_data_table[namingdict[name]].astype(str)

            #--------------------------Sorting--------------------------
            pdtPos = percent_data_table[percent_data_table['7Days Delta']>=0]
            pdtPos = pdtPos.sort_values(['7Days Delta'], ascending=[False])
            pdtPos['7Days Delta'] = pdtPos['7Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+'%')
            pdtPos['30Days Delta'] = pdtPos['30Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+'%' if x>=0 else '\u25BC'+str(abs(x))+'%')
            
            pdtNeg = percent_data_table[percent_data_table['7Days Delta']<0]
            pdtNeg = pdtNeg.sort_values(['7Days Delta'], ascending=[True])
            pdtNeg['7Days Delta'] = pdtNeg['7Days Delta'].apply(lambda x: '\u25BC'+str(abs(x))+'%')
            pdtNeg['30Days Delta'] = pdtNeg['30Days Delta'].apply(lambda x: '\u25BC'+str(abs(x))+'%' if x<0 else '\u25B2'+str(abs(x))+'%')
            if(favlist[name] == 'u'):
                percent_data_table = pd.concat([pdtPos, pdtNeg], axis=0)
            else:
                percent_data_table = pd.concat([pdtNeg, pdtPos], axis=0)
            #-------------------------------------------------------------
            percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].apply(lambda x: ' 0.00%' if('e' in x) else x)
            percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].apply(lambda x: ' 0.00%' if('e' in x) else x)
            #--------------------------Equal arrow spacing--------------------------
            
            percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].apply(lambda x: x if len(x)==(percent_data_table['7Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((percent_data_table['7Days Delta'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp')+x[1:])
            percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].apply(lambda x: x if len(x)==(percent_data_table['30Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((percent_data_table['30Days Delta'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp')+x[1:])    

            
            
            if(favlist[name] == 'u'):
                percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].map(lambda x: f'<font color="green">'+x if x[0]=='\u25B2' else f'<font color="red">'+x)
                percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].map(lambda x: f'<font color="green">'+x if x[0]=='\u25B2' else f'<font color="red">'+x)
            
            else:
                percent_data_table['30Days Delta'] = percent_data_table['30Days Delta'].map(lambda x: f'<font color="red">'+x if x[0]=='\u25B2' else f'<font color="green">'+x)
                percent_data_table['7Days Delta'] = percent_data_table['7Days Delta'].map(lambda x: f'<font color="red">'+x if x[0]=='\u25B2' else f'<font color="green">'+x)

            
            cols = ['Driver', 'Value', namingdict[denominator[name]]+'[Dr]', namingdict[numerator[name]]+'[Nr]', namingdict[name], 'Rolling 7days', '7Days Delta', 'Rolling 30days', '30Days Delta']
            if 'Trend lines' in percent_data_table.columns.tolist():
                cols += ['Trend lines']
            percent_data_table = percent_data_table[cols]

            percent_data_table[namingdict[name]] = percent_data_table[namingdict[name]].apply(lambda x: '&nbsp&nbsp'+x+'%' if len(x)==(percent_data_table[namingdict[name]].map(lambda x: len(x)).max()) else '&nbsp&nbsp'+x+(((percent_data_table[namingdict[name]].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp')+'%')
            percent_data_table['Rolling 7days'] = percent_data_table['Rolling 7days'].astype(str)
            percent_data_table['Rolling 7days'] = percent_data_table['Rolling 7days'].apply(lambda x: '&nbsp&nbsp'+x+'%' if len(x)==(percent_data_table['Rolling 7days'].map(lambda x: len(x)).max()) else '&nbsp&nbsp'+x+(((percent_data_table['Rolling 7days'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp')+'%')
            percent_data_table['Rolling 30days'] = percent_data_table['Rolling 30days'].astype(str)
            percent_data_table['Rolling 30days'] = percent_data_table['Rolling 30days'].apply(lambda x: '&nbsp&nbsp'+x+'%' if len(x)==(percent_data_table['Rolling 30days'].map(lambda x: len(x)).max()) else '&nbsp&nbsp'+x+(((percent_data_table['Rolling 30days'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp')+'%')

            if(isDollar == 'Y'):
                percent_data_table[namingdict[denominator[name]]+'[Dr]'] = (percent_data_table[namingdict[denominator[name]]+'[Dr]']/1000).round(2).astype(str)+'K'
                percent_data_table[namingdict[numerator[name]]+'[Nr]'] = (percent_data_table[namingdict[numerator[name]]+'[Nr]']/1000).round(2).astype(str)+'K'
            else:
                percent_data_table[namingdict[denominator[name]]+'[Dr]'] = (percent_data_table[namingdict[denominator[name]]+'[Dr]']).astype(int)
                percent_data_table[namingdict[numerator[name]]+'[Nr]'] = (percent_data_table[namingdict[numerator[name]]+'[Nr]']).astype(int)
            percent_data_table = percent_data_table.reset_index(drop=True)

            print('complete')
            internalist.append(percent_data_table)
            tabletitle = "Detailed Root Cause Analysis summary:-"
            internalist.append(tabletitle)

            #--------------------------GRAPHS--------------------------
            onlymetricreport = overall_anomaly_report[name]
            overall_anomaly = onlymetricreport[onlymetricreport['overall_'+name+'_anomaly']==-1]
            onlymetricreport['anomaly_points'] = onlymetricreport['overall_'+name]
            onlymetricreport.loc[onlymetricreport['overall_'+name+'_anomaly'] == 1, 'anomaly_points'] = ""
            onlymetricreport['anomaly_points'] = pd.to_numeric(onlymetricreport['anomaly_points'])
            onlymetricreport_print = onlymetricreport
            onlymetricreport_print[name+'_roll7'] = onlymetricreport_print['overall_'+name].rolling(7, min_periods=7).mean().shift(1)
            onlymetricreport_print[name+'_roll30'] = onlymetricreport_print['overall_'+name].rolling(30, min_periods=30).mean().shift(1)

            point_metric = onlymetricreport.tail(1)
            fig1 = plt.figure(1,figsize=(10,4))
            ax1 = fig1.add_subplot(111)
            x1 = onlymetricreport_print[date]
            y = onlymetricreport_print['overall_'+name]
            date_num = dates.date2num(x1)
            date_num_smooth = np.linspace(date_num.min(), date_num.max(), 300)
            spl = make_interp_spline(date_num, y, k=3)
            value_np_smooth = spl(date_num_smooth)
            dates1 = date_num_smooth
            ax1 = plt.subplot()

            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["left"].set_visible(False)
            plt.gcf().autofmt_xdate()
            data_format = mpl_dates.DateFormatter('%b, %d') #for more formats see in net
            plt.gca().xaxis.set_major_formatter(data_format)

            plt.plot(onlymetricreport_print[date], onlymetricreport_print['overall_'+name], marker='', markerfacecolor='blue', markersize=12, color='#3d199c', linewidth=1.5, path_effects=[path_effects.SimpleLineShadow(offset=(1,0.8)),path_effects.Normal()])
            ## ---------- Moving Avg. plots ----------#
            plt.plot(onlymetricreport_print[date], onlymetricreport_print[name+'_roll7'], color='peachpuff', ls='-.', lw=.8)
            plt.plot(onlymetricreport_print[date], onlymetricreport_print[name+'_roll30'], color='lightblue', ls='--', lw=.8)
            plt.plot(point_metric[date], point_metric['anomaly_points'], marker='o',markersize=12, linestyle='', color='red', linewidth=2)

            ax = plt.axes()
            fmt = '%.2f%%'
            yticks = mtick.FormatStrFormatter(fmt)
            ax.yaxis.set_major_formatter(yticks)
            ax.yaxis.grid(linestyle='--',color='#dedede')
            plt.legend(['Rate', '7days rolling', '30days rolling'], fontsize=8, loc=2)

            ylabelname = namingdict[name]
            plt.ylabel(ylabelname)

            label = "{:.2f}".format( point_metric['anomaly_points'].iloc[-1])

            plt.annotate(label+'%',
                        (point_metric[date].iloc[-1], point_metric['anomaly_points'].iloc[-1]),
                        textcoords="offset points",
                        xytext=(20,10),
                        ha='center',color = 'red')
            fig1.tight_layout()
            plt.show()
            fig1.savefig(name+cc+'.png', dpi=200)
            internalist.append([name+cc+'.png'])
            internalist.append('')                    ## empty for pattern
            myImages.append(name+cc+'.png')
            mainlist.append(internalist)

    #------------------------------------overall Severity------------------------------------
        pi_flag = 0
        if 'HIGHLY CRITICAL' in severitylist:
            severity = 'HIGHLY CRITICAL'
            pi_flag=1
        elif "CRITICAL" in severitylist:
            severity = 'CRITICAL'
            pi_flag=1
        elif "HIGH" in severitylist:
            severity = 'HIGH'
            pi_flag=1
        elif "MEDIUM" in severitylist:
            severity = 'MEDIUM'
        elif "LOW" in severitylist:
            severity = 'LOW'
        else:
            severity = 'No Anomaly'

        ## -------------------------- PATTERN IDENTIFIER DATA EXTRACTION -------------------------- ##
        date_col=date
        pat_insi_dic = {}
        anomaly_metric = [key for key, value in anomaly_flag_dictionary.items() if value == 'True']
        if pi_flag==1:
            print('DATE ENTERING FOR PI:------------------', date_pi)
            edate_pi = date_pi - datetime.timedelta(days=+7)
            try:
                rda_data = rda_query(date_pi, edate_pi)
            except (NameError, KeyError, ValueError, IOError, Exception):
                subject = "error - " + country + ' ' + product1 + ' on ' + str(date_pi)
                text = "ERROR IN PATTERN IDENTIFIER QUERY ACCESS"
                error(text, subject)
                rda_data = pd.DataFrame(columns=[date])

        df_pattern = rda_data.loc[rda_data[date] == date_pi]

#--------------------------SPARK LINES--------------------------
    df_spark = overall_df[[date] + metric_list].T
    df_spark.columns = df_spark.iloc[0,:]
    n = len(metric_list)+1
    df_spark = df_spark.iloc[1:n, -15:]
    df_spark['Trend lines'], myimagelist = sparklines.create(data=df_spark, anomaly_list=anomaly_metric)
    df_spark = df_spark.reindex(list(namingdict.keys()))
    df_spark.index = emailInsight.index
    emailInsight = pd.concat([emailInsight, df_spark['Trend lines']], axis=1)


#--------------------------UNITS--------------------------
    for name in metric_list_units:
        print('Entered Metric points')
        print('METRIC LIST---------------', name)
        internalist = []

        titlemetric = namingdict[name]
        internalist.append(titlemetric)
        label = 'overall_'+name+'_anomaly'
        driver_metric_ttb, driver_metric_ttb_table = metric_rca_ttb(percent_result,date,dictionary,segment_headings,discard,label,name)
        driver_metric_btt, driver_metric_btt_table = metric_rca_btt(percent_result,date,dictionary,segment_headings,discard,label,name)

        
        if((driver_metric_ttb.empty) or (x not in driver_metric_ttb[date].unique())):

            if((driver_metric_btt.empty) or (x not in driver_metric_btt[date].unique())):
                metric_data_table = ""

            elif(x in driver_metric_btt[date].unique()):
                x_data_percent = driver_metric_btt[driver_metric_btt[date]==x] #data of last date
                x_data_percent = x_data_percent.loc[~x_data_percent['segment_value'].isin(discard)]

                metric_data_table = driver_metric_btt_table[driver_metric_btt_table[date]==x]
                metric_data_table = metric_data_table.loc[~metric_data_table['segment_value'].isin(discard)]
                metric_data_table.rename(columns={'segment_name': 'Driver'}, inplace=True)
                metric_data_table.rename(columns={'segment_value': 'Value'}, inplace=True)
                metric_data_table.rename(columns={name+'_percent': 'Percent Contribution'}, inplace=True)
                metric_data_table.rename(columns={namingdict[name]}, inplace=True)
                metric_data_table.rename(columns={name+'_roll7': 'Rolling 7days'}, inplace=True)
                metric_data_table.rename(columns={name+'_roll30': 'Rolling 30days'}, inplace=True)
                metric_data_table.rename(columns={name+'_change_7': '7Days Delta'}, inplace=True)
                metric_data_table.rename(columns={name+'_change_30': '30Days Delta'}, inplace=True)

                metric_data_table['Rolling 7days'] = metric_data_table['Rolling 7days'].astype(int)
                metric_data_table['Rolling 30days'] = metric_data_table['Rolling 30days'].astype(int)
                # metric_data_table['Percent Contribution'] = metric_data_table['Percent Contribution'].apply(pd.to_numeric)
                metric_data_table['Percent Contribution'] = pd.to_numeric(metric_data_table['Percent Contribution'], errors='coerce')
                metric_data_table = metric_data_table.round(2)
                metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].map(lambda x:0 if x==(np.inf) else x)
                metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].map(lambda x:0 if x==(np.inf) else x)

                ## -------------------------- NEW CODE LOGIC: RCA optimization --------------------------
                drop_indices = []
                for i in range(len(metric_data_table)):
                    chg_per_roll30 = abs(metric_data_table[namingdict[name]].iloc[i] - metric_data_table['Rolling 30days'].iloc[i])
                    upper_deviation = metric_data_table['Rolling 30days'].iloc[i] + (metric_data_table['Rolling 30days'].iloc[i])*0.2
                    lower_deviation = metric_data_table['Rolling 30days'].iloc[i] - (metric_data_table['Rolling 30days'].iloc[i])*0.2

                    if ((chg_per_roll30 < 20) or (lower_deviation < metric_data_table[namingdict[name]].iloc[i] < upper_deviation)):
                        drop_indices.append(metric_data_table.iloc[i].name)

                if len(metric_data_table)>10:
                    metric_data_table = metric_data_table.drop(labels=drop_indices)
                    if len(metric_data_table)>10:
                        metric_data_table = metric_data_table.nlargest(10, 'Percent Contribution')

                metric_data_table['Percent Contribution'] = metric_data_table['Percent Contribution'].astype(str)
                metric_data_table['Percent Contribution'] = metric_data_table['Percent Contribution'].apply(lambda x: '&nbsp&nbsp'+x+'%' if len(x)==(metric_data_table['Percent Contribution'].map(lambda x: len(x)).max()) else '&nbsp&nbsp'+x+(((metric_data_table['Percent Contribution'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp'+'%'))
                cols = ['Driver', 'Value', 'Percent Contribution', namingdict[name], 'Rolling 7days', '7Days Delta', 'Rolling 30days', '30Days Delta']
                metric_data_table = metric_data_table[cols]

                #--------------------------Spark lines--------------------------
                df_ref = metric_data_table[['Value', 'Driver']]
                df_charts = pd.DataFrame()
                for i in range(len(df_ref)):
                    df_vals = df_ref.iloc[i].tolist()
                    df_trend = rca_df.loc[(rca_df.segment_value==df_vals[0]) & (rca_df.segment_name==df_vals[1])]
                    df_trend = df_trend[[date]+[name]].groupby(date).sum().reset_index().T
                    df_trend.columns = df_trend.iloc[0,:]
                    df_trend = df_trend.iloc[1:,-15:]
                    if df_trend.empty:
                        df_charts = pd.concat([df_charts, pd.Series('-')])
                        continue
                    df_trend['sparklines'], _ = sparklines.create(data=df_trend, rca=1)
                    df_charts = pd.concat([df_charts, df_trend['sparklines']])
                
                df_charts.index = metric_data_table.index
                try:
                    metric_data_table['Trend lines'] = df_charts
                    del df_charts, df_trend
                except ValueError:
                    pass

#--------------------------Sorting--------------------------
                mdtPos = metric_data_table[metric_data_table['7Days Delta']>=0]
                print('Done Saving1')
                mdtPos = mdtPos.sort_values(['7Days Delta'], ascending=[False])
                mdtPos['7Days Delta'] = mdtPos['7Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+'%')
                mdtPos['30Days Delta'] = mdtPos['30Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+'%' if x>=0 else '\u25BC'+str(abs(x))+'%')
                print('Done Saving2')
                mdtNeg = metric_data_table[metric_data_table['7Days Delta']<0]
                mdtNeg = mdtNeg.sort_values(['7Days Delta'], ascending=[True])
                mdtNeg['7Days Delta'] = mdtNeg['7Days Delta'].apply(lambda x: '\u25BC'+str(abs(x))+'%')
                mdtNeg['30Days Delta'] = mdtNeg['30Days Delta'].apply(lambda x: '\u25BC'+str(abs(x))+'%' if x<0 else '\u25B2'+str(abs(x))+'%')
                if(favlist[name] == 'u'):
                    metric_data_table = pd.concat([mdtPos,mdtNeg], axis=0)
                else:
                    metric_data_table = pd.concat([mdtNeg,mdtPos], axis=0)
#--------------------------
                metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].apply(lambda x: ' 0.00%' if('e' in x) else x)
                metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].apply(lambda x: ' 0.00%' if('e' in x) else x)
#--------------------------Equal aroow spacing--------------------------

                metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].apply(lambda x: x if len(x)==(metric_data_table['7Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((metric_data_table['7Days Delta'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp')+x[1:])
                metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].apply(lambda x: x if len(x)==(metric_data_table['30Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((metric_data_table['30Days Delta'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp')+x[1:])

#--------------------------

                if(favlist[name] == 'u'):
                    metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].map(lambda x: f'<font color="green">'+x if x[0]=='\u25B2' else f'<font color="red">'+x)
                    metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].map(lambda x: f'<font color="green">'+x if x[0]=='\u25B2' else f'<font color="red">'+x)
                else:
                    metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].map(lambda x: f'<font color="red">'+x if x[0]=='\u25B2' else f'<font color="green">'+x)
                    metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].map(lambda x: f'<font color="red">'+x if x[0]=='\u25B2' else f'<font color="green">'+x)
                
                if(isDollar == 'Y'):
                    metric_data_table[namingdict[name]] = (metric_data_table[namingdict[name]]/1000).round(2).astype(str)+'K'
                    metric_data_table['Rolling 7days'] = (metric_data_table['Rolling 7days']/1000).round(2).astype(str)+'K'
                    metric_data_table['Rolling 30days'] = (metric_data_table['Rolling 30days']/1000).round(2).astype(str)+'K'
                else:
                    metric_data_table[namingdict[name]] = (metric_data_table[namingdict[name]]).astype(int)
                    metric_data_table['Rolling 7days'] = (metric_data_table['Rolling 7days']).astype(int)
                    metric_data_table['Rolling 30days'] = (metric_data_table['Rolling 30days']).astype(int)                
                metric_data_table = metric_data_table.reset_index(drop=True)

        elif(x in driver_metric_ttb[date].unique()):

            x_data_percent = driver_metric_ttb[driver_metric_ttb[date]==x]
            x_data_percent = x_data_percent.loc[~x_data_percent['segment_value'].isin(discard)]

            metric_data_table = driver_metric_ttb_table[driver_metric_ttb_table[date]==x]
            metric_data_table = metric_data_table.loc[~metric_data_table['segment_value'].isin(discard)]
            metric_data_table.rename(columns={'segment_name': 'Driver'}, inplace=True)
            metric_data_table.rename(columns={'segment_value': 'Value'}, inplace=True)
            metric_data_table.rename(columns={'name'+'_percent': 'Percent Contribution'}, inplace=True)
            metric_data_table.rename(columns={namingdict[name]}, inplace=True)
            metric_data_table.rename(columns={'name'+'_roll7': 'Rolling 7days'}, inplace=True)
            metric_data_table.rename(columns={'name'+'_roll30': 'Rolling 30days'}, inplace=True)
            metric_data_table.rename(columns={'name'+'_change_7': '7Days Delta'}, inplace=True)
            metric_data_table.rename(columns={'name'+'_change_30': '30Days Delta'}, inplace=True)

            metric_data_table['Rolling 7days'] = metric_data_table['Rolling 7days'].astype(int)
            metric_data_table['Rolling 30days'] = metric_data_table['Rolling 30days'].astype(int)
            print('-------Done Saving')

            # metric_data_table['Percent Contribution'] = metric_data_table['Percent Contribution'].apply(pd.to_numeric)
            metric_data_table['Percent Contribution'] = pd.to_numeric(metric_data_table['Percent Contribution'], errors='coerce')
            metric_data_table = metric_data_table.round(2)
            metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].map(lambda x:0 if x==(np.inf) else x)
            metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].map(lambda x:0 if x==(np.inf) else x)

            ## -------------------------- NEW CODE LOGIC: RCA optimization --------------------------
            drop_indices = []
            for i in range(len(metric_data_table)):
                chg_per_roll30 = abs(metric_data_table[namingdict[name]].iloc[i] - metric_data_table['Rolling 30days'].iloc[i])
                upper_deviation = metric_data_table['Rolling 30days'].iloc[i] + (metric_data_table['Rolling 30days'].iloc[i])*0.2
                lower_deviation = metric_data_table['Rolling 30days'].iloc[i] - (metric_data_table['Rolling 30days'].iloc[i])*0.2

                if ((chg_per_roll30 < 20) or (lower_deviation < metric_data_table[namingdict[name]].iloc[i] < upper_deviation)):
                    drop_indices.append(metric_data_table.iloc[i].name)

            if len(metric_data_table)>10:
                metric_data_table = metric_data_table.drop(labels=drop_indices)
                if len(metric_data_table)>10:
                    metric_data_table = metric_data_table.nlargest(10, 'Percent Contribution')

            metric_data_table['Percent Contribution'] = metric_data_table['Percent Contribution'].astype(str)
            metric_data_table['Percent Contribution'] = metric_data_table['Percent Contribution'].apply(lambda x: '&nbsp&nbsp'+x+'%' if len(x)==(metric_data_table['Percent Contribution'].map(lambda x: len(x)).max()) else '&nbsp&nbsp'+x+(((metric_data_table['Percent Contribution'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp'+'%'))
            cols = ['Driver', 'Value', 'Percent Contribution', namingdict[name], 'Rolling 7days', '7Days Delta', 'Rolling 30days', '30Days Delta']
            metric_data_table = metric_data_table[cols]

    #-------------------------- NEW CODE: Spark lines --------------------------
            df_ref = metric_data_table[['Value', 'Driver']]
            df_charts = pd.DataFrame()
            for i in range(len(df_ref)):
                df_vals = df_ref.iloc[i].tolist()
                df_trend = rca_df.loc[(rca_df.segment_value==df_vals[0]) & (rca_df.segment_name==df_vals[1])]
                df_trend = df_trend[[date]+[name]].groupby(date).sum().reset_index().T
                df_trend.columns = df_trend.iloc[0,:]
                df_trend = df_trend.iloc[1:,-15:]
                if df_trend.empty:
                    df_charts = pd.concat([df_charts, pd.Series('-')])
                    continue
                df_trend['sparklines'], _ = sparklines.create(data=df_trend, rca=1)
                df_charts = pd.concat([df_charts, df_trend['sparklines']])

            df_charts.index = metric_data_table.index
            try:
                metric_data_table['Trend lines'] = df_charts
                del df_charts, df_trend
            except ValueError:
                pass

    #--------------------------Sorting--------------------------
            mdtPos = metric_data_table[metric_data_table['7Days Delta']>=0]
            mdtPos = mdtPos.sort_values(['7Days Delta'], ascending=[False])
            mdtPos['7Days Delta'] = mdtPos['7Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+'%')
            mdtPos['30Days Delta'] = mdtPos['30Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+'%' if x>=0 else '\u25BC'+str(abs(x))+'%')
            mdtNeg = metric_data_table[metric_data_table['7Days Delta']<0]
            mdtNeg = mdtNeg.sort_values(['7Days Delta'], ascending=[True])
            mdtNeg['7Days Delta'] = mdtNeg['7Days Delta'].apply(lambda x: '\u25BC'+str(abs(x))+'%')
            mdtNeg['30Days Delta'] = mdtNeg['30Days Delta'].apply(lambda x: '\u25BC'+str(abs(x))+'%' if x<0 else '\u25B2'+str(abs(x))+'%')
            if(favlist[name] == 'u'):
                metric_data_table = pd.concat([mdtPos,mdtNeg], axis=0)
            else:
                metric_data_table = pd.concat([mdtNeg,mdtPos], axis=0)
    #--------------------------
            print('Came out')
            metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].apply(lambda x: ' 0.00%' if('e' in x) else x)
            metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].apply(lambda x: ' 0.00%' if('e' in x) else x)
    #--------------------------Equal aroow spacing--------------------------

            metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].apply(lambda x: x if len(x)==(metric_data_table['7Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((metric_data_table['7Days Delta'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp')+x[1:])
            metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].apply(lambda x: x if len(x)==(metric_data_table['30Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((metric_data_table['30Days Delta'].map(lambda x: len(x)).max())-len(x))*f'&nbsp&nbsp')+x[1:])

    #--------------------------

            if(favlist[name] == 'u'):
                metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].map(lambda x: f'<font color="green">'+x if x[0]=='\u25B2' else f'<font color="red">'+x)
                metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].map(lambda x: f'<font color="green">'+x if x[0]=='\u25B2' else f'<font color="red">'+x)
            else:
                metric_data_table['30Days Delta'] = metric_data_table['30Days Delta'].map(lambda x: f'<font color="red">'+x if x[0]=='\u25B2' else f'<font color="green">'+x)
                metric_data_table['7Days Delta'] = metric_data_table['7Days Delta'].map(lambda x: f'<font color="red">'+x if x[0]=='\u25B2' else f'<font color="green">'+x)

            if(isDollar == 'Y'):
                metric_data_table[namingdict[name]] = (metric_data_table[namingdict[name]]/1000).round(2).astype(str)+'K'
                metric_data_table['Rolling 7days'] = (metric_data_table['Rolling 7days']/1000).round(2).astype(str)+'K'
                metric_data_table['Rolling 30days'] = (metric_data_table['Rolling 30days']/1000).round(2).astype(str)+'K'
            else:
                metric_data_table[namingdict[name]] = (metric_data_table[namingdict[name]]).astype(int)
                metric_data_table['Rolling 7days'] = (metric_data_table['Rolling 7days']).astype(int)
                metric_data_table['Rolling 30days'] = (metric_data_table['Rolling 30days']).astype(int)
            
            metric_data_table = metric_data_table.reset_index(drop=True)

            internalist.append(metric_data_table)
            tabletitle = "Detailed Root Cause Analysis summary:-"
            internalist.append(tabletitle)
            #--------------------------GRAPHS--------------------------

            onlymetricreport = overall_anomalyreport[name]
            if(isDollar=='Y'):
                onlymetricreport['overall_'+name] = onlymetricreport['overall_'+name]/1000
            overall_anomaly = onlymetricreport[onlymetricreport['overall_'+name+'_anomaly']==-1]
            onlymetricreport['anomaly_points'] = onlymetricreport['overall_'+name]
            onlymetricreport.loc[onlymetricreport['overall_'+name+'_anomaly'] == 1, 'anomaly_points'] = ""
            onlymetricreport['anomaly_points'] = pd.to_numeric(onlymetricreport['anomaly_points'])
            onlymetricreport_print = onlymetricreport
            onlymetricreport_print[name+'_roll7'] = onlymetricreport_print['overall_'+name].rolling(7, min_periods=7).mean().shift(1)
            onlymetricreport_print[name+'_roll30'] = onlymetricreport_print['overall_'+name].rolling(30, min_periods=30).mean().shift(1)

            point_metric = onlymetricreport.tail(1)
            fig1 = plt.figure(1,figsize=(10,4))
            ax1 = fig1.add_subplot(111)

            x1 = onlymetricreport_print[date]
            y = onlymetricreport_print['overall_'+name]
            date_num = dates.date2num(x1)
            date_num_smooth = np.linspace(date_num.min(), date_num.max(), 300)
            spl = make_interp_spline(date_num, y, k=3)
            value_np_smooth = spl(date_num_smooth)
            dates1 = date_num_smooth
            ax1 = plt.subplot()

            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["left"].set_visible(False)
            plt.gcf().autofmt_xdate()
            data_format = mpl_dates.DateFormatter('%b, %d') #for more formats see in net
            plt.gca().xaxis.set_major_formatter(data_format)

            plt.plot(onlymetricreport_print[date], onlymetricreport_print['overall_'+name], marker='', markerfacecolor='blue', markersize=12, color='#3d199c', linewidth=1.5, path_effects=[path_effects.SimpleLineShadow(offset=(1,0.8)),path_effects.Normal()])
            ## ---------- Moving Avg. plots ----------#
            plt.plot(onlymetricreport_print[date], onlymetricreport_print[name+'_roll7'], color='peachpuff', ls='-.', lw=.8 )
            plt.plot(onlymetricreport_print[date], onlymetricreport_print[name+'_roll30'], color='lightblue', ls='--', lw=.8 )
            plt.plot(point_metric[date], point_metric['anomaly_points'], marker='o',markersize=12, linestyle='', color='red', linewidth=2)

            ax = plt.axes()
            ax.yaxis.grid(linestyle='--',color='#dedede')
            plt.legend(['Apps', '7days rolling', '30days rolling'], fontsize=8, loc=2)
            ylabelname = namingdict[name]
            plt.ylabel(ylabelname)

            label = round(point_metric['anomaly_points'].iloc[-1])
            plt.annotate(label,
                        (point_metric[date].iloc[-1], point_metric['anomaly_points'].iloc[-1]),
                        textcoords="offset points",
                        xytext=(20,10),
                        ha='center',color = 'red')
            fig1.tight_layout()
            plt.show()
            fig1.savefig(name+cc+'.png', dpi=200)
            internalist.append([name+cc+'.png'])
            myImages.append(name+cc+'.png')

            ## PATTERN TO BE PRINTED FROM BELOW FUNCTION.
            if (name=='approved_appl'):
                target_col = target_col_list[0]
            elif (name=='fraud_decline_appl'):
                target_col = target_col_list[1]
            elif (name=='incoming_appl'):
                target_col = target_col_list[2]
                pruned_path=''

            try:
                if (name in anomaly_metric) and (pi_flag==1):
                    pat_insights, common_line, common_pat_dic, pattern = hawk_pattern(df_pattern, date_pi, nan_threshold, favlist[name], date, target_col, pruned_path, target_col_list)
                    pat_insi_dic[name]=pat_insights

                    ########################### Line chart here ###########################
                    rda_chart = rda_data.copy(deep=True)
                    for seg, val in common_pat_dic.items():
                        rda_chart = rda_chart.loc[rda_chart[seg].isin(val)]

                    rda_chart = rda_chart[[date]+target_col_list].groupby([date]).sum()[target_col].astype(int).to_frame().T
                    rda_chart['Trend'] = sparklines.create(rda_chart)[0]
                    print('-----------', target_col, end='\n\n\n')

                    pattern_line = f"""<table style="width:100% ; border: 1px solid #000000 ; text-align:center ; border-spacing: 12px"><font face = "Arial" size=2.5em>
                    <td style="width:auto">{common_line}</td>
                    <td style="text-align:center ; vertical-align: center ; padding:.7em; width: 20%;">{rda_chart.Trend[0]}</td>
                    </font></table>""" + """<style>
                    th, td {
                    border: 1px solid black;
                        padding: 6px;
                        }
                        </style>
                        """                               
                    pattern_lis1, pattern_lis2 = '', ''
                    pat_cnt = 0
                    for i in pattern:
                        if i not in [' ', '', None]:
                            pattern_lis2 += """<td style="text-align:left ; vertical-align: top ; padding:.7em">""" + ' ' + i + '</td>'
                            pat_cnt += 1
                    
                    pattern_lis1 += """
            <table style="width:100% ; border: 1px solid #1e477a ; text-align:left ; border-spacing: 6px">
            <tr>
            """
                    for i in range(pat_cnt):
                        pattern_lis1 += f'<th>SPLIT-{i+1}</th>'
                    
                    pattern_lis1 += '</tr>'
                    pattern_lis = pattern_lis0 + pattern_lis1 + pattern_lis2 + '</table>'
                
                else:
                    pattern_lis, pat_insights, pat_insi_dic[name]='', '', ''

            except (NameError, KeyError, ValueError, IOError, Exception):
                subject = "error - " + country + ' ' + product1 + ' on ' + str(date_pi)
                text = "ERROR IN PATTERN IDENTIFIER"
                error(text, subject)
                pattern_lis, pat_insights, pat_insi_dic[name]='', '', ''
                pass
            internalist.append(pattern_lis)
            print("INSIGHTS: ========", pat_insights)
            mainlist.append(internalist)
            
        
        summary_text = []
        
        holiday_summary=''
        if(severity!='No Anomaly'):
            if(country=='FR'):
                country_holidays = holidays.CountryHoliday('FRA')
            else:
                try:
                    country_holidays = holidays.CountryHoliday(country[:2])
                except:
                    country_holidays = holidays.CountryHoliday('US')

            country_holidays.append({"2023-11-27": "Cyber Monday"})
            country_holidays.append({"2023-11-24": "Black Friday"})
            country_holidays.append({"2024-12-02": "Cyber Monday"})
            country_holidays.append({"2024-11-29": "Black Friday"})
            country_holidays.append({"2025-12-01": "Cyber Monday"})
            country_holidays.append({"2025-11-28": "Black Friday"})
            country_holidays.append({"2026-11-27": "Black Friday"})
            country_holidays.append({"2026-11-30": "Cyber Monday"})
            holiday_summary = country_holidays.get(cc)
            if(holiday_summary != None):
                summary_text.append('Today is '+'<b>'+holiday_summary+'</b>'+' Holiday in '+country)

        anomaly_line = '<b>'+'There is no anomaly.'+'</b>'
        true_anomaly_list = []
        for key, value in anomaly_flag_dictionary.items():
            if value == 'True':
                true_anomaly_list.append(key)
        

        if true_anomaly_list:
            anomaly_line = '<b>'+'Anomaly is detected for - '+'</b>'
            num = 1
            for i in true_anomaly_list:
                if(num<len(true_anomaly_list)):
                    anomaly_line = anomaly_line + '<i>'+ namingdict[i] + '</i>' + ', '
                else:
                    anomaly_line = anomaly_line + '<i>'+ namingdict[i] + '</i>'
                num = num+1

        summary_text.append(anomaly_line)
        new_list = list(true_anomaly_list)

        for key in numerator:          ## appr_rate key
            if(numerator[key] in new_list):  ## APPR_APPL in new_list
                new_list.append(key)        ## appr_rate
                new_list.remove(numerator[key])    ## appr_appl
        for key in denominator:
            if(denominator[key] in new_list):
                new_list.append(key)
                new_list.remove(denominator[key])
        
        res = []
        for i in new_list:
            if i not in res:
                res.append(i)

        for a in res:
            if(a in metric_list_units):
                change = overall_df[a+'_change_7'].iloc[-1]
                change_DoD = overall_df[a+'_DoD'].iloc[-1]

                overall_insight = ''
                if(change>0 and change_DoD<0):
                    if (overall_insight == ''):
                        overall_insight = 'Even though there is a drop of '+str(abs(change_DoD))+'% DoD in '+namingdict[a]
                    else:
                        overall_insight = overall_insight + ' and a drop of '+str(abs(change_DoD))+'% DoD in '+namingdict[a]

                if(change<0 and change_DoD>0):
                    if (overall_insight == ''):
                        overall_insight = 'Even though there is an increase of '+str(abs(change_DoD))+'% DoD in '+namingdict[a]
                    else:
                        overall_insight = overall_insight + ' and an increase of '+str(abs(change_DoD))+'% DoD in '+namingdict[a]
                if (overall_insight == ''):
                    overall_insight = 'Compare to past rolling 7 days '
                else:
                    overall_insight = overall_insight + ', but compare to past rolling 7 days '

                
                if(change>0):
                    overall_insight = overall_insight + 'there is an increase of '+str(abs(change))+'%.'
                else:
                    overall_insight = overall_insight + 'there is a drop '+str(abs(change))+'%.'

                overall_insight = '<b>'+ namingdict[a] + ' : '+'</b>' + overall_insight
                summary_text.append(overall_insight)

            if(a in metric_list_percent):
                change = overall_df[a+'_change_7'].iloc[-1]
                numerator_change = overall_df[numerator[a]+'_change_7'].iloc[-1]
                denominator_change = overall_df[denominator[a]+'_change_7'].iloc[-1]

                change_DoD = overall_df[a+'_DoD'].iloc[-1]
                numerator_change_DoD = overall_df[numerator[a]+'_DoD'].iloc[-1]
                denominator_change_DoD = overall_df[denominator[a]+'_DoD'].iloc[-1]


                overall_insight = ''
                if(change>0 and change_DoD<0):
                    if (overall_insight == ''):
                        overall_insight = 'Even though there is a drop of '+str(abs(change_DoD))+'% DoD in '+namingdict[a]
                    else:
                        overall_insight = overall_insight + ' and a drop of '+str(abs(change_DoD))+'% DoD in '+namingdict[a]

                if(change<0 and change_DoD>0):
                    if (overall_insight == ''):
                        overall_insight = 'Even though there is an increase of '+str(abs(change_DoD))+'% DoD in '+namingdict[a]
                    else:
                        overall_insight = overall_insight + ' and an increase of '+str(abs(change_DoD))+'% DoD in '+namingdict[a]

                if(numerator_change>0 and numerator_change_DoD<0):
                    if (overall_insight == ''):
                        overall_insight = 'Even though there is a drop of '+str(abs(numerator_change_DoD))+'% DoD in '+namingdict[numerator[a]]
                    else:
                        overall_insight = overall_insight + ' and a drop of '+str(abs(numerator_change_DoD))+'% DoD in '+namingdict[numerator[a]]

                if(numerator_change<0 and numerator_change_DoD>0):
                    if (overall_insight == ''):
                        overall_insight = 'Even though there is an increase of '+str(abs(numerator_change_DoD))+'% DoD in '+namingdict[numerator[a]]
                    else:
                        overall_insight = overall_insight + ' and an increase of '+str(abs(numerator_change_DoD))+'% DoD in '+namingdict[numerator[a]]

                if(denominator_change>0 and denominator_change_DoD<0):
                    if (overall_insight == ''):
                        overall_insight = 'Even though there is a drop of '+str(abs(denominator_change_DoD))+'% DoD in '+namingdict[denominator[a]]
                    else:
                        overall_insight = overall_insight + ' and a drop of '+str(abs(denominator_change_DoD))+'% DoD in '+namingdict[denominator[a]]

                if(denominator_change<0 and denominator_change_DoD>0):
                    if (overall_insight == ''):
                        overall_insight = 'Even though there is an increase of '+str(abs(denominator_change_DoD))+'% DoD in '+namingdict[denominator[a]]
                    else:
                        overall_insight = overall_insight + ' and an increase of '+str(abs(denominator_change_DoD))+'% DoD in '+namingdict[denominator[a]]
                if (overall_insight == ''):
                    overall_insight = 'Compare to past rolling 7 days '
                else:
                    overall_insight = overall_insight + ', but compare to past rolling 7 days '

                if(change>0):
                    if(numerator_change>0 and denominator_change<0):
                        overall_insight = overall_insight + 'there is an increase of '+str(abs(change))+'%.' + ' This is due to increase in '+namingdict[numerator[a]]+' by '+str(abs(numerator_change))+'% and drop in ' +namingdict[denominator[a]] + ' by ' + str(abs(denominator_change))+'%.'
                    elif(numerator_change<0 and denominator_change<0):
                        overall_insight = overall_insight + 'there is an increase of '+str(abs(change))+'%.' + ' This is because the rate at which '+namingdict[denominator[a]] + ' dropped ['+str(abs(denominator_change))+'%] is greater than the rate at which '+namingdict[numerator[a]] + ' dropped ['+str(abs(numerator_change))+'%].'
                    elif(numerator_change>0 and denominator_change>0):
                        overall_insight = overall_insight + 'there is an increase of '+str(abs(change))+'%.' + ' This is because the rate at which '+namingdict[numerator[a]] + ' increased ['+str(abs(numerator_change))+'%] is greater than the rate at which '+'<b>'+namingdict[denominator[a]]+'</b>' + ' increased ['+str(abs(denominator_change))+'%].'
                    elif(numerator_change==0 and denominator_change<0):
                        overall_insight = overall_insight + 'there is an increase of '+str(abs(change))+'%.' + ' This is due to drop in '+'<b>'+namingdict[denominator[a]]+'</b>' + ' by '+str(abs(denominator_change))+'% while '+'<b>'+namingdict[numerator[a]]+'</b>' + ' remains stable.'
                    elif(numerator_change>0 and denominator_change==0):
                        overall_insight = overall_insight + 'there is an increase of '+str(abs(change))+'%.' + ' This is due to increase in '+'<b>'+namingdict[numerator[a]]+'</b>' + ' by '+str(abs(numerator_change))+'% while '+'<b>'+namingdict[denominator[a]]+'</b>' + ' remains stable.'

                    if(a == 'decline_rate'):
                        if(favlist[a] == 'u'):
                            overall_insight += '<b><u>' + 'This is due to better quality of '+namingdict[denominator[a]]+'.' + '</u></b>'
                        else:
                            if ('fraud_decline_appl' in anomaly_metric) and (pat_insi_dic['fraud_decline_appl'] != ''):
                                overall_insight += '<b><u>' + 'There is higher decline from ' + pat_insi_dic['fraud_decline_appl'] + ' risky segments. ' + 'This is due to higher bad '+namingdict[denominator[a]]+'.'+'</u></b>'

                            elif ('incoming_appl' in anomaly_metric) and (pat_insi_dic['incoming_appl'] != ''):
                                overall_insight += '<b><u>' + 'There is higher incoming from ' + pat_insi_dic['incoming_appl'] + ' risky segments. ' + 'This is due to higher bad '+namingdict[denominator[a]]+'.'+'</u></b>'

                            else:
                                overall_insight += '<b><u>' + 'This is due to higher bad '+namingdict[denominator[a]]+'.'+'</u></b>'

                
                elif(change<0):
                    if(numerator_change<0 and denominator_change>0):
                        overall_insight = overall_insight + 'there is a drop of '+str(abs(change))+'%.' + ' This is due to increase in '+'<b>'+namingdict[denominator[a]]+'</b>' + ' by '+str(abs(denominator_change))+'% and drop in '+'<b>'+namingdict[numerator[a]]+'</b>' + ' by '+str(abs(numerator_change))+'%.'
                    elif(numerator_change<0 and denominator_change<0):
                        overall_insight = overall_insight + 'there is a drop of '+str(abs(change))+'%.' + ' This is because the rate at which '+'<b>'+namingdict[numerator[a]]+'</b>' + ' dropped ('+str(abs(numerator_change))+'%) is greater than the rate at which '+'<b>'+namingdict[denominator[a]]+'</b>' + ' dropped '+'('+str(abs(denominator_change))+'%).'
                    elif(numerator_change>0 and denominator_change>0):
                        overall_insight = overall_insight + 'there is a drop of '+str(abs(change))+'%.' + ' This is because the rate at which '+'<b>'+namingdict[denominator[a]]+'</b>' + ' increased ('+str(abs(denominator_change))+'%) is greater than the rate at which '+'<b>'+namingdict[numerator[a]]+'</b>' + ' increased '+'('+str(abs(numerator_change))+'%).'
                    elif(numerator_change==0 and denominator_change>0):    
                        overall_insight = overall_insight + 'there is a drop of '+str(abs(change))+'%.' + ' This is due to increase in '+'<b>'+namingdict[denominator[a]]+'</b>' + ' by '+str(abs(denominator_change))+'% while '+'<b>'+namingdict[numerator[a]]+'</b>' + ' remains stable.'
                    elif(numerator_change<0 and denominator_change==0):
                        overall_insight = overall_insight + 'there is a drop of '+str(abs(change))+'%.' + ' This is due to drop in '+'<b>'+namingdict[numerator[a]]+'</b>' + ' by '+str(abs(numerator_change))+'% while '+'<b>'+namingdict[denominator[a]]+'</b>' + ' remains stable.'

                    if(a == 'decline_rate'):
                        ## ----------------------------------------- CHANGES ----------------------------------------- ##
                        appr_rate_chg = overall_df['appr_rate_change_7'].iloc[-1]

                        if ('fraud_decline_appl' in anomaly_metric) and (pat_insi_dic['fraud_decline_appl'] != ''):
                            overall_insight += ' There is higher decline from ' + '<b><u>' + pat_insi_dic['fraud_decline_appl'] + '</u></b>' + ' risky segments. '

                        elif ('incoming_appl' in anomaly_metric) and (pat_insi_dic['incoming_appl'] != ''):
                            overall_insight += ' There is higher incoming from ' + '<b><u>' + pat_insi_dic['incoming_appl'] + '</u></b>' + ' risky segments. '

                        elif (favlist[a] == 'u'):
                            overall_insight += '<b><u>' + 'This is due to higher bad '+namingdict[denominator[a]]+'.' + '</u></b>'

                        elif (denominator_change >= -40) and (appr_rate_chg > -5) and (appr_rate_chg != 0):
                            overall_insight += '<b><u>' + 'This is due to better quality of '+namingdict[denominator[a]]+'.' + '</u></b>'

                    elif(a == 'appr_rate'):
                        if(favlist[a] == 'd') and ('approved_appl' in anomaly_metric) and (pat_insi_dic['approved_appl'] != ''):
                            overall_insight += '<b><u>' + 'There is approvals from ' + pat_insi_dic['approved_appl'] + ' risky segments. ' + '</u></b>'

                overall_insight = '<b>'+ namingdict[a] + ' : '+'</b>' + overall_insight
                summary_text.append(overall_insight)

        
        if(campaignstat!='' and holiday_summary == None):
            summary_text.append('<b>'+campaignstat+'</b>')

        mailpoints = """<style>
                        ul li { margin-bottom: 1em; }
                        </style> <ul>"""
        for e in summary_text:
            mailpoints+='<li>'+e+'</li>'

        mailpoints += '</ul>'
        read_link(severity)

    send_email([["",mainlist]],mailpoints,country,severity,emailInsight,metrictype,product1,cc,dashboardLink,extra,highRiskMailDL,MedRiskMailDL,NoAnomalyDL,isDollar,pi_flag)

    for img in myImages:
        os.remove(img)
    for fil in file_list:
        os.remove(fil)
    print('returning')
    return severitylist,severity,str(summary_text)   
                                            