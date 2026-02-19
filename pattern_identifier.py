from sklearn import tree
from sklearn.tree import _tree, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import decimal
import multiprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
import timeit
import datetime
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', -1)


"""
---------------------------------------------------------------------------
Here our pre-processing of data starts
---------------------------------------------------------------------------
"""

def prep_data(df, indicator_cols, na_threshold=95):
    ## differentiating columns with most unique values
    lists = []
    cols = df.columns
    [lists.append(i) if (df[i].nunique())>=len(df)*0.5 else '' for i in cols]    
    ## na values count analysis
    miss_val_per = 100 * df.isnull().sum()/len(df)
    df_f=pd.DataFrame(miss_val_per).nlargest(100, columns=0)
    na_list = list(df_f.loc[df_f[0]>na_threshold].index)
    
    ## id and dt colums, we can ignore them for feature ext.
    ty = [w for w in list(df.columns) if w.endswith('time')]
    url = [w for w in list(cols) if w.endswith('url')]
    dt = [w for w in list(cols) if w.endswith('dt')]
    ids = [w for w in list(cols) if w.endswith('id')]
    
    ## drop list
    drp_list = list(set(lists + na_list + url + dt + ids))
    df1 = df.drop(drp_list, axis=1)
    print('Dropped pre-filtering: ', len(drp_list), drp_list)
    
    ## capturing datatypes of each column
    dtype = [type(df1[i].value_counts().idxmax()) for i in df1.columns]
    
    def list_duplicates(seq):
        tally = defaultdict(list)
        for i,item in enumerate(seq):
            tally[item].append(i)
        return [(key,locs) for key,locs in tally.items() 
                           if len(locs)>0]
                           
    dic = {dup: keys for dup, keys in list_duplicates(dtype)}
    
    ## typecasting the str columns just for safety
    str_lis = []
    for COL in dic[type('abc')]:
        if df1.iloc[:, COL].nunique() < 2:             ## this cond'n skips the columns with one entry eg. ['Y', None]. No use of these columns
            continue
        str_lis.append(df1.iloc[:, COL].name)
        
    ## typecasting the deci columns just for safety
    num_lis = []
    if type(decimal.Decimal('1.1')) in dic:
        for COL in dic[type(decimal.Decimal('1.1'))]:
            if df1.iloc[:, COL].nunique() >= 2:       ## if any values only like -9, -99 are present, then dropping the col
                num_lis.append(df1.iloc[:, COL].name)
                
    final_col_lis = str_lis + num_lis
    if (indicator_cols is not None) and (indicator_cols in final_col_lis):
        final_col_lis.remove(indicator_cols)
        
    return final_col_lis, str_lis, num_lis

"""
---------------------------------------------------------------------------
Here the decoding and patterns extraction function for the tree for standalone pkg
---------------------------------------------------------------------------
"""
def get_rules_pkg(tree, feature_names, class_names, df1):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            dec = lis_enc[name]
            vals1 = (dec.loc[dec[target]<=threshold][name].tolist(), dec.loc[dec[target]<=threshold][target].tolist())
            p1 += [{name: vals1}]
            recurse(tree_.children_left[node], p1, paths)
            dec = lis_enc[name]
            vals2 = (dec.loc[dec[target]>threshold][name].tolist(), dec.loc[dec[target]>threshold][target].tolist())
            p2 += [{name: vals2}]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)
    
    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    for j in range(len(paths)):
        df2=df1.copy(deep=True)
        for idx, i in enumerate(paths[j]):
            if type(i)==tuple:
                continue
            
            elif list(i.values())[0][0] in [ [[]], '', 'nan', np.nan, [], '#', None ]:
                del paths[j][idx]
                
            else:
                colmn=list(i.keys())[0]
                seg=list(i.values())[0][0]
                df3=pd.DataFrame()
                if len(seg)>1:
                    for k in seg:
                        df3 = pd.concat([df2.loc[df2[colmn]==k], df3], axis=0)
                    df2=df3.copy()
                else:
                    df2 = df2.loc[df2[colmn]==seg[0]]
        paths[j].append(df2)
        
    rules, res = [], []
    missed_cols, pat_cols, col_len = [], [], []
    for path in paths:
        rule = "=> "
        col=[]
        for p in path[:-2]:
            colmn = list(p.keys())[0]
            if colmn in col:
                #p[colmn]=list(set(list(path[:-2][col.index(colmn)].values())[0][0]).intersection(set(list(p.values())[0][0])))
                continue
                
            elif df1[colmn].nunique() == len(list(p.values())[0][0]):
                continue
                
            elif rule != "=> " :
                rule += " ; "
            col.append(colmn)
            rule += str(list(p.keys())[0]) + ' -> ' + str([f'{var} ({int(contrib*100)}%)' for var, contrib in zip(list(p.values())[0][0][:3], list(p.values())[0][1][:3])])
            
        not_in_path = set(least_feat)-set(col)
        for i in not_in_path:
            population = df1.groupby(i)[target].sum()
            if (path[-1][i].nunique() == 1) and (path[-1][i].unique()[0] not in [None, 'None', '']): # and (population.sort_values(ascending=False)[0]>=sum(population)*.7):
                col.append(i)
                contrib_ = int((sum(df1[df1[i]==path[-1][i].unique()[0]][target])/sum(df1[target]))*100)
                rule += " ; " + str(f"{i} -> ['{path[-1][i].unique()[0]} ({contrib_}%)']")
                
        if class_names is None:
            missed = df1.columns.difference(col).tolist()
            missed.remove(target)
            missed_cols.append(missed)
            pat_cols.append(col)
            col_len.append(len(col))
            res.append(np.round(path[-2][0][0][0],3))
        rules += [rule]
        common_line = ''
    
    return common_line, rules, res, missed_cols, pat_cols, col_len

"""
---------------------------------------------------------------------------
Here the decoding and patterns extraction function for the tree for HAWK
---------------------------------------------------------------------------
"""

def get_rules_hawk(tree, feature_names, class_names, df1):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            dec = lis_enc[name]
            vals1 = (dec.loc[dec[target]<=threshold] [name].tolist(), dec.loc[dec[target]<=threshold] [target].tolist())
            p1 += [{name: vals1}]
            recurse(tree_.children_left[node], p1, paths)
            dec = lis_enc[name]
            vals2 = (dec.loc[dec[target]>threshold] [name].tolist(), dec.loc[dec[target]>threshold] [target].tolist())
            p2 += [{name: vals2}]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)
    
    # sort by samples count
    samples_count = [p[-1][0][0][0] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)][:3]
    
    for j in range(len(paths)):
        df2=df1.copy(deep=True)
        col_1=set()
        for idx, i in enumerate(paths[j]):
            if type(i)==tuple:
                continue
            
            elif list(i.values())[0][0] in [ [[]], '', 'nan', 'None', [], '#', None ]:
                del paths[j][idx]
                
            else:
                colmn=list(i.keys())[0]
                col_1.add(colmn)
                seg=list(i.values())[0][0]
                df3=pd.DataFrame()
                if len(seg)>1:
                    for k in seg:
                        df3 = pd.concat([df2.loc[df2[colmn]==k], df3], axis=0)
                    df2=df3.copy()
                else:
                    df2 = df2.loc[df2[colmn]==seg[0]]
                    
        not_in_path = set(least_feat)-col_1
        for i in not_in_path:
            k=df2[i].unique()
            #print('k- here: -----', k)
            contrib = float(((df1.groupby(by=i)[target].sum().sort_values(ascending=False)/sum(df1.groupby(by=i)[target].sum()))[k[0]]))
            if (df2[i].nunique() == 1) and (k[0] not in [None, 'None', '', '#']):
                paths[j].insert(len(paths[j])-1, ({i:([k[0]], [contrib])}))
                
    ## if the variable repeats in all three patterns, we'll color code it.
    rep_cnt=2
    color_code_col, color_cols = [], {}
    col_dict = {}
    for idx, path in enumerate(paths):
        col_rep=[]
        for i in path[:-1]:
            segment = list(i.keys())[0]
            col_rep.append(segment)
            segment_val = [j for j in list(i.values())[0][0] if j not in [None, 'None', '', []]][:3]
            if (segment in color_cols.keys()) and (segment_val==color_cols[segment]):
                color_code_col.append(segment)
            elif col_rep.count(segment)==1:
                color_cols[segment]=segment_val
        col_dict[idx]=col_rep
        
    print("These are color coded cols: ", set(color_code_col))
    
    rules, res, n_samples = [], [], []
    missed_cols, pat_cols, col_len = [], [], []
    common_patt = {}
    data_dic = pd.read_csv('fav_dic.tsv', sep='\t', index_col=0)        ##read the data_dic for color coding
    
    for path in paths:
        rule = " "
        pat_insights = {}
        col=[]
        
        for p in path[:-1]:
            colmn = list(p.keys())[0]
            vals = [i for i in list(p.values())[0][0] if i not in [None, 'None', '', []]][:3]
            
            if colmn in col:
                continue
                
            elif (df1[colmn].nunique() == len(list(p.values())[0][0])) or (vals in [None, '', 'None', []]):
                continue
                
            elif color_code_col.count(colmn)>=rep_cnt:
                common_patt[colmn] = vals
                continue
                
            col.append(colmn)
            rule += str(f'<p> {colmn.upper()}') + ' &#8594 '
            
            if colmn in data_dic.index.tolist():
                ## Risky segments color code for better rep.
                severity={risk for risk in ['Favour', 'Unfavour'] for val in vals if data_dic.loc[data_dic.index==colmn][risk].str.contains(val).values[0] == True}
                
                rule_severity=[]
                rule_vals = []
                if 'Favour' in severity:
                    severity_lis = data_dic.loc[data_dic.index==colmn].Favour.str.split(',').values[0]
                    rule_severity += [f'<span style="color:green"><b>{var}</b></span> ' for var in vals if var in severity_lis]
                    rule_vals += [var for var in vals if var in severity_lis]
                    
                if 'Unfavour' in severity:
                    severity_lis = data_dic.loc[data_dic.index==colmn].Unfavour.str.split(',').values[0]
                    rule_severity += [f'<span style="color:red"><b>{var}</b></span> ' for var in vals if var in severity_lis]
                    rule_vals += [var for var in vals if var in severity_lis]
                    try:
                        pat_insights[colmn.upper()] += [var for var in vals if var in severity_lis]
                    except KeyError:
                        pat_insights[colmn.upper()] = [var for var in vals if var in severity_lis]
                        
                else:
                    rule_severity += [f'<b>{var}</b> ' for var in vals if var not in rule_vals+[None, 'None', '#', '']]
                    
                rule += str(rule_severity) + '</p>'
                
            else:
                rule += str([f'<b>{var}</b> ' for var, contrib in zip(vals, list(p.values())[0][1][:3]) if var not in ['None', 'None', '#', '']]) + ' </p>'
        if class_names is None:
            missed = df1.columns.difference(col).tolist()
            missed.remove(target)
            missed_cols.append(missed)
            pat_cols.append(col)
            col_len.append(len(col))
            res.append(np.round(path[-1][0][0][0],3))
            n_samples.append(path[-1][1])
        rules += [rule]
        
    print('Repeated cols: ', common_patt, end='\n\n')
    common_line='<p>'
    for key, val in common_patt.items():
        common_line += str(f'{key.upper()}') + ' &#8594 '
        severity={risk for risk in ['Favour', 'Unfavour'] if data_dic.loc[data_dic.index==key][risk].str.contains('|'.join(val)).values == True}
        severity.append('')
        if (severity[0] in ['Favour', 'Unfavour']):
            rule_severity=[]
            rule_vals = []
            severity.sort()
            if severity[1] == 'Favour':
                severity_lis = data_dic.loc[data_dic.index==key].Favour.str.split(',').values[0]
                rule_severity += [f'<span style="color:green"><b>{var}</b></span> ' for var in val if var in severity_lis]
                rule_vals += [var for var in val if var in severity_lis]
            if severity[-1] == 'Unfavour':
                severity_lis = data_dic.loc[data_dic.index==key].Unfavour.str.split(',').values[0]
                rule_severity += [f'<span style="color:red"><b>{var}</b></span> ' for var in val if var in severity_lis]
                rule_vals += [var for var in val if var in severity_lis]
                try:
                    pat_insights[key.upper()] += [var for var in val if var in severity_lis]
                except KeyError:
                    pat_insights[key.upper()] = [var for var in val if var in severity_lis]
            else:
                rule_severity += [f'<b>{var}</b> ' for var in val if var not in rule_vals+[None, 'None', '#', '']]
                
            common_line += str(rule_severity) + '<span style="color:#2997DB">&nbsp &#10095&#10095 &nbsp </span>'
            
        else:
            common_line += str([f'<b>{var}</b> ' for var in val]) + '<span style="color:#2997DB">&nbsp &#10095&#10095 &nbsp </span>'
            
    pat_insight = ''
    for k, v in pat_insights.items():
        pat_insight += str(k) + ' {' + ",".join(v) + '}, '
        
    if pat_insight != '':
        pat_insight = '<b>' + pat_insight[:-2] + '</b>'
        
    common_line = common_line[:-29] + '</p>'
    return pat_insight, common_line, rules, res, missed_cols, pat_cols, col_len, common_patt


"""
---------------------------------------------------------------------------
Here's our customized tree algorithm with encoding, pruning, CV, etc.,
---------------------------------------------------------------------------
"""

def tree_algo(df1, plot_tree=None):
    ## ENCODING FOR DT REGR.
    global lis_enc
    global least_feat
    lis_enc = {}
    def mean_encoding(df, column, target_):
        means_list=(df.groupby(by=column)[target_].sum().sort_values(ascending=False)/sum(df.groupby(by=column)[target_].sum())).to_dict()
        means_list['None']=0
        label = pd.DataFrame(means_list.items(), columns=[column, target_])
        lis_enc[column] = label
        return df[column].map(means_list)

    X = df1.drop(target, axis=1)
    y = df1[target]
    for x in X.columns:
        X[x] = mean_encoding(df1, x, target)

    ## CROSS_VAL
    def dtree_grid_search(X,y,nfolds):
        #create a dictionary of all values we want to test
        val = len(X.columns)
        param_grid = {'max_depth': np.arange(round(val*0.4), val)}
        # decision tree model
        dtree_model=DecisionTreeRegressor()
        cpu = multiprocessing.cpu_count()
        #use gridsearch to test all values
        n = multiprocessing.cpu_count()
        dtree_gscv = GridSearchCV(dtree_model, param_grid, scoring='r2', cv=nfolds, n_jobs=round(n/2))
        #fit model to data
        dtree_gscv.fit(X, y)
        return dtree_gscv.best_params_

    ## DT IMPLEMENTATION
    grid = dtree_grid_search(X, y, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
    regr = DecisionTreeRegressor(**grid, random_state=0)
    regr.fit(X_train.values, y_train.values)

    ## PRUNING TREE
    path = regr.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    model = DecisionTreeRegressor(**grid, random_state=0, ccp_alpha=max(ccp_alphas)*0.2)
    model.fit(X_train,y_train)

    # PLOT TREE WITH EXPECTED FLOW
    if plot_tree == 'Y':
        export_graphviz(regr, out_file=f'{target[:-3]}tree.dot',filled=True, feature_names=X.columns.tolist(), proportion=True, impurity=True, rounded=True, node_ids=True)
        plt.figure(figsize=(30,25)) #, dpi=250)
        tree_plt(model,filled=True, feature_names=X.columns.tolist(), fontsize=10, proportion=True, impurity=True, rounded=True, node_ids=True)
        plt.show()
    else:
        pass

    ## FEATURE IMPORTANCE
    feats = {}
    for feature, importance in zip(X.columns, regr.feature_importances_):
        feats[feature] = importance #add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'mse'})
    least_feat = importances.nsmallest(7, 'mse').index.tolist()

    ## PRINT RULES IN ONE LINER
    if pkg==1:
        pat_insights, common_line, rules, response, missed_cols, pat_cols, col_len = get_rules_pkg(model, X.columns, None, df1)
        common_patt=None
    else:
        pat_insights, common_line, rules, response, missed_cols, pat_cols, col_len, common_patt = get_rules_hawk(model, X.columns, None, df1)

    r=pd.DataFrame([rules, response, missed_cols, pat_cols, col_len]).T.rename(columns={0:'Rule', 1:'Response', 2:'Missed_cols', 3:'Pattern_cols', 4:'Pattern_len'})
    r[['Response', 'Pattern_len']] = r[['Response', 'Pattern_len']].astype(float)
    r.nlargest(3, 'Response')

    return pat_insights, common_line, common_patt, r.sort_values(by='Response', ascending=False)[:5]

"""
---------------------------------------------------------------------------
Now comes the final part main function to run all the above functions and insert the data into table
---------------------------------------------------------------------------
"""

def main_func(df, s_date, e_date, na_threshold, plot_tree, date_col, include_cols, indicator_cols, target_col, target_val, appl_count_list):
#   start = timeit.default_timer()
    global target
    df = df.loc[(df[date_col] >= s_date) & (df[date_col] <= e_date)]
    ## replacing the '#' values in the df
    for i in df.columns:
        df[i].replace(to_replace=['?','#','###','NULL','null','Null','unknown','Unknown','UNKNOWN','None'], value=None, regex=True, inplace=True)

    if target_val is not None:
        cleaned_cols, str_cols, numerical_cols = prep_data(df, indicator_cols, na_threshold)

        ## ADVANCE FILTERING
        final_columns = []
        drp_cols = []
        for i in cleaned_cols:
            if (df[i].nunique() <= 15):
                final_columns.append(i)
            elif i in numerical_cols:
                final_columns.append(i)
            else:
                drp_cols.append(i)

        if include_cols != None:
            df12 = df[list(set(final_columns + [target_col] + include_cols))]
        else:
            df12 = df[list(set(final_columns + [target_col]))]

        try:
            df12.drop(indicator_cols, axis=1, errors='ignore', inplace=True)
        except ValueError:
            pass
        # df12.dropna(how='all', axis=1, inplace=True)
        # df12.dropna(how='all', axis=0, inplace=True)
        colmn = df12.columns.tolist()
        print('Dropped in filtering: ', len(drp_cols), drp_cols, '\n')
        print('Final cols: ', len(colmn), colmn)

        if target_val=='':
            print(df12.shape, '\n')
            df = df12.drop(appl_count_list, axis=1, errors='ignore')
            cols = df.columns.tolist()
            target='incoming_appls_cnt'
            df[target]=1
            df = df.fillna('None').groupby(cols)[target].sum().reset_index()

        elif (target_val !='') and (target_col in colmn):
            df = df12.loc[df12[target_col]==target_val]
            print(df.shape, '\n')
            df.drop(appl_count_list, axis=1, errors='ignore', inplace=True)
            cols = df.columns.tolist()
            target=target_col+'_cnt'
            df[target]=1
            df = df.fillna('None').groupby(cols)[target].sum().reset_index()

        else:
            print(df12.shape, '\n')
            target=target_col
            cols = df12.columns.tolist()
            cols.remove(target)
            df = df12.fillna('None').groupby(cols)[target].sum().reset_index()
            df = df[~df.index.duplicated()]

    else:
        target=target_col
        target_cols = appl_count_list.copy()
        try:
            target_cols.remove(target)
        except ValueError:
            pass
        drp_cols = target_cols + [w for w in df.columns.tolist() if w.endswith(('dt', 'ts'))]
        df.drop(drp_cols, axis=1, inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        cols = df.columns.tolist()
        try:
            cols.remove(target)
        except ValueError:
            pass
        df[target] = df[target].astype(float)
        df = df.fillna('None').groupby(cols)[target].sum().reset_index()
        df = df.loc[df[target]!=0]

    pat_insights, common_line, common_patt, output = tree_algo(df, plot_tree)
    return pat_insights, common_line, common_patt, output

"""
The below method is to print the patterns in all the target metrics for STANDALONE pkg
"""


def pprint(df, s_date, e_date, na_threshold, plot_tree, date_col, include_cols, indicator_cols, target_val, appl_count_list):
    global pkg
    pkg = 1
    s_date, e_date = datetime.datetime.strptime(s_date, '%Y-%m-%d').date(), datetime.datetime.strptime(e_date, '%Y-%m-%d').date()
    diff = (e_date - s_date).days
    print('Pattern Identifier running for: ', diff+1, ' days.')
    for i, segment in enumerate(appl_count_list):
        if (i==0) and (target_val!=None) and (len(appl_count_list)>1) :
            _, common_line, _0, df_pattern = main_func(df, s_date, e_date, na_threshold, plot_tree, date_col, include_cols, indicator_cols, segment, '', appl_count_list)
            print('********************************** TOP-5 PATTERN FROM ANALYSIS FOR: Incoming appl **********************************', end='\n')
            [print(i, end='\n\n') for i in df_pattern.Rule.values]
            print('\n\n')
        _, common_line, _0, df_pattern = main_func(df, s_date, e_date, na_threshold, plot_tree, date_col, include_cols, indicator_cols, segment, target_val, appl_count_list)
        print('********************************** TOP-5 PATTERN FROM ANALYSIS FOR: ', segment, ' **********************************')
        [print(i, end='\n\n') for i in df_pattern.Rule.values]
        print('\n\n')

"""
The below method is to print the patterns in all the target metrics for HAWK
"""
def hawk_pattern(df, date, na_threshold, favour, date_col, segment, target_val, appl_count_list):
    global metric_fav, pkg
    pkg = 0
    metric_fav=favour
    s_date=e_date=date #datetime.date(2022,2,1) #date
    plot_tree, include_cols, indicator_cols = None, None, None
    pat_insights, common_line, common_patt, df_pattern = main_func(df, s_date, e_date, na_threshold, plot_tree, date_col, include_cols, indicator_cols, segment, target_val, appl_count_list)
    return pat_insights, common_line, common_patt, [i for i in df_pattern.Rule.values[:3]]