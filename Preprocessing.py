import teradata
import getpass
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
import joblib
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
import multiprocessing
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import os
import time
import copy
import datetime
from NextGen.Daily.emailfile import send_email
from NextGen.Daily.mainfunctionFile import start_analysis
from datetime import date
from NextGen.Daily.email_error import error
from dateutil.relativedelta import *
from OutputFile import tera, rda_query
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
from scipy.stats import trim_mean

def execute(data1, metric_list_num, metric_list_denom, metric_list_units, metric_list_percent, metric_list, indexInsight, favourability, Date, country, product1, metrictype, extra, frequency, campaignlist, dashboardlink, cutt_off_conditions, highRiskMailDL, MedRiskMailDL, NoAnomalyDL, isDollar, nan_threshold, target_col_list, pruned_path): #rda_df
    diff = 10
    newdata = data1
    while(diff > 0):
        data1 = newdata
        print('data completion')
        
        namingdict = {metric_list[i]: indexInsight[i] for i in range(len(metric_list))}
        print(namingdict)
        favlist = {metric_list[i]: favourability[i] for i in range(len(metric_list))}
        numerator = {}
        for x in range(len(metric_list_percent)):
            numerator[metric_list_percent[x]] = metric_list_num[x]
            
        denominator = {}
        for i in range(len(metric_list_percent)):
            denominator[metric_list_percent[i]] = metric_list_denom[i]
            
        df = pd.DataFrame(data1.groupby(Date)[metric_list_units].apply(lambda x: x.astype(float).sum()))
        df.insert(0, Date, df.index)
        df.reset_index(inplace=True, drop=True)
        df[Date] = pd.to_datetime(df[Date])
        df.sort_values(by=[Date], inplace=True)
        print(df.tail(3))
        last_row = df.tail(1)
        
        last_date = last_row[Date].iloc[-1]
        last_date_hawk_alert = last_date.strftime("%Y-%m-%d")
        date_file = pd.read_csv('date_File.csv')
        date_file['dates'] = pd.to_datetime(date_file['dates'])
        last_file_date = date_file['dates'].iloc[-1]
        last_file_date_hawk_alert = last_file_date.strftime("%Y-%m-%d")
        file_day = last_file_date.day
        file_month = last_file_date.month
        file_year = last_file_date.year
        last_day = last_date.day
        last_month = last_date.month
        last_year = last_date.year
        last_date = date(last_year, last_month, last_day)
        file_date = date(file_year, file_month, file_day)
        cc1 = file_date + datetime.timedelta(days=+1)
        delta = last_date - cc1
        diff = delta.days
        
        print(diff)
        print(cc1)
        cc = cc1.strftime("%d %B %Y")
        print(cc)
        
        df = df[df[Date] <= pd.Timestamp(cc1)]
        data1 = data1[data1[Date] <= cc1]
        
        for i in range(len(metric_list_percent)):
            df[metric_list_percent[i]] = (df[metric_list_num[i]] / df[metric_list_denom[i]]) * 100


        campaignli = []
        campaignstat = ''
        if(campaignlist[0] != '' and campaignlist[1] != ''):
            for a in campaignlist:
                df[a + '_percentile90'] = df[a].rolling(window=7, center=False).apply(lambda x: pd.Series(x).quantile(0.9), raw=True).shift(periods=1)
                df[a + '_per90cutoff'] = ((df[a] - df[a + '_percentile90']) / df[a + '_percentile90']).multiply(100).round(2)
                df[a + '_campaign'] = 0
                df.loc[df[a + '_per90cutoff'] >= 10, a + '_campaign'] = 1
                campaignli.append(df[a + '_campaign'].iloc[-1])
            if(campaignli[0] == 1 and campaignli[1] == 1):
                campaignstat = 'Today could be a campaign event'
                
        ## ---------------- CAMPAIGN EVENT FLAG IN PAST DAYS ---------------- ##
        campaignli_1 = []
        if campaignstat == '':
            for i in range(2, 6):
                if(campaignlist[0] != '' and campaignlist[1] != ''):
                    for a in campaignlist:
                        df[a + '_percentile90'] = df[a].rolling(window=7, center=False).apply(lambda x: pd.Series(x).quantile(0.9), raw=True).shift(periods=1)
                        df[a + '_per90cutoff1'] = ((df[a] - df[a + '_percentile90']) / df[a + '_percentile90']).multiply(100).round(2)
                        df[a + '_campaign_prev'] = 0
                        df.loc[df[a + '_per90cutoff1'] >= 10, a + '_campaign_prev'] = 1
                        campaignli_1.append(df[a + '_campaign_prev'].iloc[-(i-1)])
                    if(campaignli_1[i-2] == 1 and campaignli_1[i-1] == 1):
                        campaignstat = 'There could be campaign event in past 4 days.'
                        break
        
        listinsightUpfav = []
        listinsightDownfav = []
        indexFavUp = []
        indexFavDown = []
        
        for a, b, c in zip(metric_list, favourability, indexInsight):
            df[a + '_DoD'] = df[a].pct_change().multiply(100).round(2)
            if(a in metric_list_percent):
                inlist = [str(df[a].iloc[-1].round(2)) + '%']
            else:
                if(isDollar == 'Y'):
                    inlist = [str(round((df[a].iloc[-1]) / 1000)) + 'K']
                else:
                    inlist = [round(df[a].iloc[-1])]
            
            df[a + '_roll_7_mean'] = df[a].rolling(window=7).mean().shift(periods=1)
            if(a in metric_list_percent):
                inlist.append(str(df[a + '_roll_7_mean'].iloc[-1].round(2)) + '%')
            elif(a in metric_list_units):
                tempname = namingdict[a]
                if(isDollar == 'Y'):
                    inlist.append(str(round((df[a + '_roll_7_mean'].iloc[-1]) / 1000)) + 'K')
                else:
                    inlist.append(round(df[a + '_roll_7_mean'].iloc[-1]))
            
            df[a + '_change_7'] = ((df[a] - df[a + '_roll_7_mean']) / df[a + '_roll_7_mean']).multiply(100).round(2)
            inlist.append(df[a + '_change_7'].iloc[-1])
            
            df[a + '_roll_30_mean'] = df[a].rolling(window=30).mean().shift(periods=1)
            if(a in metric_list_percent):
                inlist.append(str(df[a + '_roll_30_mean'].iloc[-1].round(2)) + '%')
            elif(a in metric_list_units):
                tempname = namingdict[a]
                if(isDollar == 'Y'):
                    inlist.append(str(round((df[a + '_roll_30_mean'].iloc[-1]) / 1000)) + 'K')
                else:
                    inlist.append(round(df[a + '_roll_30_mean'].iloc[-1]))
            
            df[a + '_change_30'] = ((df[a] - df[a + '_roll_30_mean']) / df[a + '_roll_30_mean']).multiply(100).round(2)
            inlist.append(df[a + '_change_30'].iloc[-1])
            
            if(b == 'd'):
                listinsightDownfav.append(inlist)
                indexFavDown.append(c)
            else:
                listinsightUpfav.append(inlist)
                indexFavUp.append(c)
            
            df[a + '_trim_mean_roll_9'] = df[a].rolling(9).apply(lambda x: trim_mean(x, 0.2), raw=True).shift(periods=1)
            
            #-----numerator_cutoff-----#
            df[a + '_cutoff'] = ((df[a] - df[a + '_trim_mean_roll_9']) / df[a + '_trim_mean_roll_9']).multiply(100).round(2)
            df[a + '_cutoff_flag'] = 0
            df.loc[(df[a + '_cutoff'] < -15) | (df[a + '_cutoff'] > 15), a + '_cutoff_flag'] = 1
            

        insightTable1 = pd.DataFrame(listinsightDownfav, columns=['Actuals', 'Rolling 7Days', '7Days Delta', 'Rolling 30Days', '30Days Delta'])
        insightTable2 = pd.DataFrame(listinsightUpfav, columns=['Actuals', 'Rolling 7Days', '7Days Delta', 'Rolling 30Days', '30Days Delta'])
        
        insightTable1['Metric'] = indexFavDown
        insightTable2['Metric'] = indexFavUp
        
        insightTable1['7Days Delta'] = insightTable1['7Days Delta'].map(lambda x: 0 if x==(np.inf) else x)
        insightTable1['30Days Delta'] = insightTable1['30Days Delta'].map(lambda x: 0 if x==(np.inf) else x)
        insightTable2['7Days Delta'] = insightTable2['7Days Delta'].map(lambda x: 0 if x==(np.inf) else x)
        insightTable2['30Days Delta'] = insightTable2['30Days Delta'].map(lambda x: 0 if x==(np.inf) else x)
        
        #directions
        insightTable1['30Days Delta'] = insightTable1['30Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+' %' if x>=0 else '\u25BC'+str(abs(x))+' %')
        insightTable1['7Days Delta'] = insightTable1['7Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+' %' if x>=0 else '\u25BC'+str(abs(x))+' %')
        insightTable2['30Days Delta'] = insightTable2['30Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+' %' if x>=0 else '\u25BC'+str(abs(x))+' %')
        insightTable2['7Days Delta'] = insightTable2['7Days Delta'].apply(lambda x: '\u25B2'+str(abs(x))+' %' if x>=0 else '\u25BC'+str(abs(x))+' %')
        #spacing
        insightForSpace = pd.concat([insightTable1, insightTable2], axis=0)
        insightTable1['7Days Delta'] = insightTable1['7Days Delta'].apply(lambda x: x if len(x)==(insightForSpace['7Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((insightForSpace['7Days Delta'].map(lambda x: len(x)).max())-len(x))*(2*f'&nbsp'))+x[1:])
        insightTable1['30Days Delta'] = insightTable1['30Days Delta'].apply(lambda x: x if len(x)==(insightForSpace['30Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((insightForSpace['30Days Delta'].map(lambda x: len(x)).max())-len(x))*(2*f'&nbsp'))+x[1:])
        
        insightTable2['7Days Delta'] = insightTable2['7Days Delta'].apply(lambda x: x if len(x)==(insightForSpace['7Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((insightForSpace['7Days Delta'].map(lambda x: len(x)).max())-len(x))*(2*f'&nbsp'))+x[1:])
        insightTable2['30Days Delta'] = insightTable2['30Days Delta'].apply(lambda x: x if len(x)==(insightForSpace['30Days Delta'].map(lambda x: len(x)).max()) else x[0]+(((insightForSpace['30Days Delta'].map(lambda x: len(x)).max())-len(x))*(2*f'&nbsp'))+x[1:])
        
        #coloring
        insightTable1['7Days Delta'] = insightTable1['7Days Delta'].map(lambda x: f"<font color='red'>"+x if x[0] == "\u25B2" else f"<font color='green'>"+x)
        insightTable1['30Days Delta'] = insightTable1['30Days Delta'].map(lambda x: f"<font color='red'>"+x if x[0] == "\u25B2" else f"<font color='green'>"+x)
        insightTable2['7Days Delta'] = insightTable2['7Days Delta'].map(lambda x: f"<font color='green'>"+x if x[0] == "\u25B2" else f"<font color='red'>"+x)
        insightTable2['30Days Delta'] = insightTable2['30Days Delta'].map(lambda x: f"<font color='green'>"+x if x[0] == "\u25B2" else f"<font color='red'>"+x)
        
        insightTable = pd.concat([insightTable1, insightTable2], axis=0)
        names = insightTable['Metric'].tolist()
        insightTable.index = names
        insightTable.drop(['Metric'], axis=1, inplace=True)
        emailInsight = insightTable
        emailInsight = emailInsight.reindex(indexInsight)
        print(emailInsight)
        
        df_last = df[df[Date] == pd.Timestamp(cc1)]
        df_last = df_last.round(2)
        dfe = df_last
        
        #--------------------------Data Mature check Conditions--------------------------
        if((cutt_off_conditions==[]) or ('' in cutt_off_conditions)):
            cutt_off_conditions = []
            for i in range(len(metric_list_units)):
                cutt_off_conditions.append(40)
                
        if(diff==0):
            print('entering datacondition')
            last_value_u=[]
            avg_u = []
            for datadenom in metric_list_units:
                last_value_u.append(df[datadenom].iloc[-1])
                avg_u.append(df[datadenom+'_roll_30_mean'].iloc[-1])
            print(last_value_u)
            print(avg_u)
            lower_u = []
            for i,j in zip(avg_u,cutt_off_conditions):
                lower_u.append(i-((j/100)*i))
            print(lower_u)
            flag = []
            for i,j in zip(last_value_u,lower_u):
                if(i>=j):
                    flag.append(0)
                else:
                    flag.append(1)
            print(flag)
        else:
            flag = [0]
        #--------------------------------------------------------------------------------
        
        product = country+extra
        dataframe_subj = "[HAWK] data - "+country+extra+' '+product1+' '+metrictype+" on "+str(cc)+" dataframe"
        if(1 in flag):
            text = "Data issue. Data not Mature."+" The count of "+ str(metric_list_units) +" is : "+ str(last_value_u)+'.'
            print("data issue")
            subject = "[HAWK] - "+country+' '+extra+' '+product1+' '+metrictype+" on "+str(cc)
            error(text,subject)
            status = "error - data not mature"
            summary_point = 'NULL'
            severity = ["" for i in range(len(metric_list))]
            highsev = 'NULL'
            
        elif(diff>=0):
            print("proceed")
            os.remove('date_File.csv')
            myData = [["dates"],[cc1]]
            myFile = open('date_File.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(myData)
            print("Writing complete")
            
            overall_df = df
            
            data1[Date]= pd.to_datetime(data1[Date])
            data1 = data1.sort_values(by=[Date])
            
            data1.fillna("###", inplace = True)
            x = np.datetime64(data1[Date].iloc[-1])
            date_data = data1.copy()
            
            date_data.drop(date_data.columns.difference([Date]), 1, inplace=True)
            date_data = date_data.drop_duplicates(keep='first')
            data1.dtypes
            data1.columns
            segment_headings = []
            for i in data1.columns:
                if(i not in metric_list and i not in Date):
                    segment_headings.append(i)
                    
            data2 = data1[data1[Date] == cc1]
            null_cols = data2.columns[data2.isin(['###']).any()].tolist()
            
            for null_col in null_cols:
                msn_pct = data2.loc[data2[null_col] == '###'].count()[0]/len(data2)
                if msn_pct >= .5:
                    segment_headings.remove(null_col)
                    data1.drop(null_col, axis=1, inplace=True)
                    
            print (segment_headings)
            dictionary = {}
            for name in segment_headings:
                data1[name] = data1[name].astype(str)
                rename = data1[name].unique()
                rename = sorted(rename)
                rename = pd.DataFrame(rename,columns=['name'])
                rename.insert(0,name,rename.index)
                rename.reset_index(inplace=True, drop=True)
                dictionary[name] = rename
                
            for segment in segment_headings:
                number = LabelEncoder()
                data1[segment]=number.fit_transform(data1[segment].astype('str'))
            data1.dtypes
            
            dataframe = {}
            for name in segment_headings:
                dataf = []
                data1[name] = data1[name].astype(str)
                rename = data1[name].unique()
                for a in rename:
                    df = data1[data1[name]==a]
                    dataf.append(df)
                dataframe[name] = dataf
                
            severity,highsev,summary_point = start_analysis(data1,Date,cc1,country,metric_list,metric_list_percent,metric_list_units,metric_list_num,numerator,denominator,cc,extra,dataframe_subj,frequency,df_last,overall_df,namingdict,emailInsight,metrictype,product1,favlist,campaignstat,date_data,segment_headings,dictionary,dataframe,dashboardlink,x,highRiskMailDL,MedRiskMailDL,NoAnomalyDL,isDollar,nan_threshold,target_col_list,pruned_path) #df_pattern
            status = "success"
            
            if(isDollar == 'Y'):
                subject = "[HAWK] "+highsev+" - "+country+' '+product1+' '+metrictype+" supress view ("+extra+")"+" on "+str(cc)
            else:
                subject = "[HAWK] "+highsev+" - "+country+' '+product1+' '+metrictype+" "+extra+" on "+str(cc)
                
        elif(diff==0):
            text = "Data not updated"
            print(text)
            subject = "[HAWK] - "+country+extra+' '+product1+' '+metrictype+" on "+str(cc)
            error(text,subject)
            status = "error - data not updated"
            severity = ["" for i in range(len(metric_list))]
            summary_point = 'NULL'
            highsev = 'NULL'
            
        elif(diff<0):
            text = 'Date is in past'
            print(text)
            subject = "[HAWK] - "+country+' '+extra+' '+product1+' '+metrictype+" on "+str(cc)
            error(text,subject)
            summary_point = 'NULL'
            status = "error - data not mature"
            highsev = 'NULL'
            severity = ["" for i in range(len(metric_list))]
            
        #--------------------------sql insert--------------------------
        summary_point = summary_point.replace("<b>", "")
        summary_point = summary_point.replace("</b>", "")
        summary_point = summary_point.replace("<i>", "")
        summary_point = summary_point.replace("</i>", "")
        summary_point = summary_point.replace("<u>", "")
        summary_point = summary_point.replace("</u>", "")
        summary_point = summary_point.replace("\"", "")
        summary_point = summary_point.replace("'", "")
        summary_point = summary_point.replace("[", "")
        summary_point = summary_point.replace("]", "")
        print(summary_point)
        tera(country,product1,metrictype,'overall',highsev,status,last_file_date_hawk_alert,cc1,last_date_hawk_alert,subject,'D',extra,summary_point)
        for i,j in zip(metric_list,severity):
            tera(country,product1,metrictype,i,j,status,last_file_date_hawk_alert,cc1,last_date_hawk_alert,subject,'D',extra,summary_point)
            
    return 1