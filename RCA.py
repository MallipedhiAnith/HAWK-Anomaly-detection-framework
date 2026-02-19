import time
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os


#{email_tAG : DF ,FLOW_TAX : df}
#top to bottom
#percent_rca_ttb(percent_result,date,dictionary,segment_headings,discard)
def percent_rca_ttb(percent_result,date,dictionary,segment_headings,discard,label,name):

    metric = name
    append_list = []
    append_table = []
    for l in range(len(percent_result)):
        df_merge_col = percent_result[l]
        del_date = df_merge_col[date].iloc[0]
        df_merge_col = df_merge_col[df_merge_col[date]!=del_date]

        ttb = []
        table = []
        df_merge_col2 = df_merge_col[df_merge_col[label]==-1] #filtered dataframe with anomalies on particular metric
        for d in df_merge_col2[date].unique(): #for every unique date
            df = df_merge_col2[df_merge_col2[date]==d] #individual date wise
            df = df.loc[~df['segment_value'].isin(discard)] #...extra removing records with having discard values in segment_value
            df1 = df.copy()
            df['weightage']=int(0)

            #decline_rate_global_percent_anomaly
            if -1 in df[metric+'_global_percent_anomaly'].unique():
                df = df[df[metric+'_global_percent_anomaly']==-1] #Taking df were global_anomaly is -1
                if -1 in df[metric+'_eligible_percent_anomaly'].unique():
                    df = df[df[metric+'_eligible_percent_anomaly']==-1] #Taking df were eligible_anomaly is -1
                    df['weightage']=(df[metric+'_global_percent']*0.6)+(df[metric+'_eligible_percent']*0.4) #giving weightage
                    ans = df[df.weightage == df.weightage.max()]

                    ttb.append(ans)

                else:
                    df['weightage']=(df[metric+'_global_percent']*0.6)+(df[metric+'_eligible_percent']*0.4)
                    ans = df[df.weightage == df.weightage.max()]

                    ttb.append(ans)
                    #table.append(df)
            elif -1 in df[metric+'_eligible_percent_anomaly'].unique():
                df = df[df[metric+'_eligible_percent_anomaly']==-1]
                df['weightage']=(df[metric+'_global_percent']*0.6)+(df[metric+'_eligible_percent']*0.4)
                ans = df[df.weightage == df.weightage.max()]
                ttb.append(ans)

            if -1 in df1[metric+'_global_percent_anomaly'].unique():
                df = df1[df1[metric+'_global_percent_anomaly']==-1]
                table.append(df)
            if -1 in df1[metric+'_eligible_percent_anomaly'].unique():
                df = df1[df1[metric+'_eligible_percent_anomaly']==-1]
                table.append(df)

        #btt.append(ans)
        if ttb:
            ttb = pd.concat(ttb)
            ttb = ttb.drop_duplicates(keep='first')
        if table:
            table = pd.concat(table)
            table = table.drop_duplicates(keep='first')

        append_list.append(ttb)
        append_table.append(table)

    if append_list:
        append_list = pd.concat(append_list)
        append_table = pd.concat(append_table)
        append_list = append_list.sort_values([date,'segment_name','segment_value'], ascending=[True, True, True])
        append_table = append_table.sort_values([date,'segment_name','segment_value'], ascending=[True, True, True])

    append_list = pd.DataFrame(append_list)
    append_table = pd.DataFrame(append_table)
    return append_list,append_table


#bottom to top
def percent_rca_btt(percent_result,date,dictionary,segment_headings,discard,label,name):

    metric = name
    append_list = []
    append_table = []
    for l in range(len(percent_result)):
        df_merge_col = percent_result[l]
        del_date = df_merge_col[date].iloc[0]
        df_merge_col = df_merge_col[df_merge_col[date]!=del_date]

        btt = []
        table = []
        df_merge_col2 = df_merge_col[df_merge_col[label]!=-1]
        for d in df_merge_col2[date].unique():
            df = df_merge_col2[df_merge_col2[date]==d]
            df = df.loc[~df['segment_value'].isin(discard)] #...extra
            df1 = df.copy()
            df['weightage'] = int(0)
            if -1 in df[metric+'_global_percent_anomaly'].unique():
                df = df[df[metric+'_global_percent_anomaly']==-1]
                if -1 in df[metric+'_eligible_percent_anomaly'].unique():
                    df = df[df[metric+'_eligible_percent_anomaly']==-1]
                    df['weightage']=(df[metric+'_global_percent']*0.6)+(df[metric+'_eligible_percent']*0.4)
                    ans = df[df.weightage == df.weightage.max()]
                    btt.append(ans)
                    #table.append(df)
                else:
                    df['weightage']=(df[metric+'_global_percent']*0.6)+(df[metric+'_eligible_percent']*0.4)
                    ans = df[df.weightage == df.weightage.max()]
                    btt.append(ans)
                    #table.append(df)
            elif -1 in df[metric+'_eligible_percent_anomaly'].unique():
                df = df[df[metric+'_eligible_percent_anomaly']==-1]
                df['weightage']=(df[metric+'_global_percent']*0.6)+(df[metric+'_eligible_percent']*0.4)
                ans = df[df.weightage == df.weightage.max()]
                btt.append(ans)
                #table.append(df)

            if -1 in df1[metric+'_global_percent_anomaly'].unique():
                df = df1[df1[metric+'_global_percent_anomaly']==-1]
                table.append(df)
            if -1 in df1[metric+'_eligible_percent_anomaly'].unique():
                df = df1[df1[metric+'_eligible_percent_anomaly']==-1]
                table.append(df)


        #btt.append(ans)
        if btt:
            btt = pd.concat(btt)
            btt = btt.drop_duplicates(keep='first')
        if table:
            table = pd.concat(table)
            table = table.drop_duplicates(keep='first')

            append_list.append(btt)
            append_table.append(table)
    if append_list:
        append_list = pd.concat(append_list)
        append_table = pd.concat(append_table)
        append_list = append_list.sort_values([date,'segment_name','segment_value'], ascending=[True, True, True])
        append_table = append_table.sort_values([date,'segment_name','segment_value'], ascending=[True, True, True])

    append_list = pd.DataFrame(append_list)
    append_table = pd.DataFrame(append_table)
    return append_list,append_table

#...............................................unit...................................................

#top to bottom
def metric_rca_ttb(unit_result,date,dictionary,segment_headings,discard,label,name):
    metric = name

    append_list = []
    append_table = []
    for l in range(len(unit_result)):
        df_merge_col = unit_result[l]
        del_date = df_merge_col[date].iloc[0]
        df_merge_col = df_merge_col[df_merge_col[date]!=del_date]

        ttb = []
        table = []
        df_merge_col1 = df_merge_col[df_merge_col[label]==-1]
        
        for d in df_merge_col1[date].unique():

            df = df_merge_col1[df_merge_col1[date]==d]
            df = df.loc[~df['segment_value'].isin(discard)] #...extra
            df1 = df.copy()
            df['weightage'] = int(0)

            if -1 in df[metric+'_anomaly'].unique():
                df = df[df[metric+'_anomaly']==-1]
                if -1 in df[metric+'_percent_anomaly'].unique():
                    df = df[df[metric+'_percent_anomaly']==-1]
                    df['weightage']=(df[metric]*0.6)+(df[metric+'_percent']*0.4)
                    ans = df[df.weightage == df.weightage.max()]
                    ttb.append(ans)

                else:
                    df['weightage']=(df[metric]*0.6)+(df[metric+'_percent']*0.4)
                    ans = df[df.weightage == df.weightage.max()]
                    ttb.append(ans)

            elif -1 in df[metric+'_percent_anomaly'].unique():
                df = df[df[metric+'_percent_anomaly']==-1]
                df['weightage']=(df[metric]*0.6)+(df[metric+'_percent']*0.4)
                ans = df[df.weightage == df.weightage.max()]
                ttb.append(ans)

            if -1 in df1[metric+'_anomaly'].unique():
                df = df1[df1[metric+'_anomaly']==-1]
                table.append(df)
            if -1 in df1[metric+'_percent_anomaly'].unique():
                df = df1[df1[metric+'_percent_anomaly']==-1]
                table.append(df)


        if ttb:
            ttb = pd.concat(ttb)
            ttb = ttb.drop_duplicates(keep='first')
        if table:
            table = pd.concat(table)
            table = table.drop_duplicates(keep='first')

            append_list.append(ttb)
            append_table.append(table)

    if append_list:
        append_list = pd.concat(append_list)
        append_table = pd.concat(append_table)
        append_list = append_list.sort_values([date,'segment_name','segment_value'], ascending=[True, True, True])
        append_table = append_table.sort_values([date,'segment_name','segment_value'], ascending=[True, True, True])

    append_list = pd.DataFrame(append_list)
    append_table = pd.DataFrame(append_table)
    return append_list,append_table

#bottom to top
def metric_rca_btt(unit_result,date,dictionary,segment_headings,discard,label,name):

    metric = name
    append_list = []
    append_table = []
    for l in range(len(unit_result)):
        df_merge_col = unit_result[l]
        del_date = df_merge_col[date].iloc[0]
        df_merge_col = df_merge_col[df_merge_col[date]!=del_date]

        btt = []
        table = []
        df_merge_col1 = df_merge_col[df_merge_col[label]!=-1]
        for d in df_merge_col1[date].unique():
            df = df_merge_col1[df_merge_col1[date]==d]
            df = df.loc[~df['segment_value'].isin(discard)] #...extra
            df1 = df.copy()
            df['weightage'] = int(0)
            #print('---------------------------RCA DF====================================')
            if -1 in df[metric+'_anomaly'].unique():
                df = df[df[metric+'_anomaly']==-1]
                if -1 in df[metric+'_percent_anomaly'].unique():
                    df = df[df[metric+'_percent_anomaly']==-1]
                    df['weightage']=(df[metric]*0.6)+(df[metric+'_percent']*0.4)
                    ans = df[df.weightage == df.weightage.max()]
                    btt.append(ans)

                else:
                    df['weightage']=(df[metric]*0.6)+(df[metric+'_percent']*0.4)
                    ans = df[df.weightage == df.weightage.max()]
                    btt.append(ans)

            elif -1 in df[metric+'_percent_anomaly'].unique():
                df = df[df[metric+'_percent_anomaly']==-1]
                df['weightage']=(df[metric]*0.6)+(df[metric+'_percent']*0.4)
                ans = df[df.weightage == df.weightage.max()]
                btt.append(ans)

            if -1 in df1[metric+'_anomaly'].unique():
                df = df1[df1[metric+'_anomaly']==-1]
                table.append(df)
            if -1 in df1[metric+'_percent_anomaly'].unique():
                df = df1[df1[metric+'_percent_anomaly']==-1]
                table.append(df)

        if btt:
            btt = pd.concat(btt)
            btt = btt.drop_duplicates(keep='first')
        if table:
            table = pd.concat(table)
            table = table.drop_duplicates(keep='first')

            append_list.append(btt)
            append_table.append(table)

    if append_list:
        append_list = pd.concat(append_list)
        append_table = pd.concat(append_table)
        append_list = append_list.sort_values([date,'segment_name','segment_value'], ascending=[True, True, True])
        append_table = append_table.sort_values([date,'segment_name','segment_value'], ascending=[True, True, True])

    append_list = pd.DataFrame(append_list)
    append_table = pd.DataFrame(append_table)
    return append_list,append_table