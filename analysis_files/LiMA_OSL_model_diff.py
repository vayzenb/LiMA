# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:47:42 2022

@author: vayze
"""

import pandas as pd
import numpy as np
import pdb
import itertools
from scipy import stats

curr_dir = 'C:/Users/vayze/Desktop/GitHub_Repos/LiMA/'
#curr_dir = '/user_data/vayzenbe/GitHub_Repos/LiMA'

model_type= ['skel','pixel1','CorNet_S',"SayCam", "ResNet_IN", "ResNet_SN", "flownet"]
actual_name = ['Skeleton', 'Pixel','CorNet-S', 'ResNext-SAY', 'ResNet-IN','ResNet-SIN', 'FlowNet']

skel = [['23', '31', '26'], ['31_0', '31_50']]
SF = ['Skel', 'Bulge']



model_cols = ['Model', 'Skel1', 'SF1', 'hab_trials', 'hab_start', 'hab_end', 'skel2','sf2', 'skel_cat', "sf_cat", 'error', 'diff']
summary_cols = ['Model', 'Condition', 'hab_num', 'hab_start', 'hab_end', 'Acc', 'CI_Low', 'CI_High']

alpha = .05
iter = 1000


exp = ['Exp1', 'Exp2']

def extract_model_data(exp, comps):
    object_summary = pd.DataFrame(columns = ['exp','model','skel', 'sf', 'score'])
    for mm in model_type:
        print(mm)
        curr_model = pd.DataFrame(columns = model_cols)
        for sk in skel[exn]:
            for sf in SF:
                #load csv
                temp_df = pd.read_csv(f'{curr_dir}/Results/AE/{ee}_{mm}_Figure_{sk}_{sf}_Result.csv', 
                                      header = None, names = model_cols)
                
                #subtract habituation
                temp_df['error'] = temp_df['error'] - temp_df['hab_end']
                temp_df['error'][temp_df['error'] <=0] = 0
                
                diff_score =temp_df['error'][(temp_df['skel_cat'] == 'diff') & (temp_df['sf_cat'] == 'diff')].mean()
                same_score = temp_df['error'][(temp_df['skel_cat'] == comps[0]) & (temp_df['sf_cat'] == comps[1])].mean()

                #
                try:
                    cat_score = diff_score/ (diff_score + same_score)
                except:
                    cat_score = .5
                    
                temp_summary = pd.Series([ee, mm, sk, sf, cat_score], index = object_summary.columns)
                
                object_summary = object_summary.append(temp_summary, ignore_index = True )
    
    return object_summary
'''
def extract_model_data(exp, comps):
    model_summary = pd.DataFrame(columns = ['exp','model','skel', 'sf', 'score'])
    all_models = pd.DataFrame(columns = model_cols)
    for mm in model_type:
        print(mm)
        
        for sk in skel[exn]:
            for sf in SF:
                #load csv
                temp_df = pd.read_csv(f'{curr_dir}/Results/AE/{ee}_{mm}_Figure_{sk}_{sf}_Result.csv', 
                                      header = None, names = model_cols)
                
                #subtract habituation
                temp_df['diff'] = temp_df['error'] - temp_df['hab_end']
                temp_df['diff'][temp_df['diff'] <=0] = 0
                
                all_models = all_models.append(temp_df)
                
                
    return all_models
'''
def extract_infant_data(exp):
    """
    Organize infant data like models
    """
    infant_data = pd.read_csv(f'{curr_dir}/Infant_Data/{exp}_Infant_Data.csv')
    infant_data['skel1'][infant_data['skel1'] == 266] = 26
   
    #diff_score = infant_data['Novel']- infant_data['HabEnd']
    infant_data['diff'] = infant_data['Novel']- infant_data['HabEnd']
    
    infant_data['diff'][infant_data['diff'] <0] = 0
    infant_data['same']= infant_data['Familiar']- infant_data['HabEnd']
    infant_data['same'][infant_data['same'] <0] = 0
    
    infant_data['score'] = infant_data['diff']
    
    for ii in range(0, len(infant_data)):
        if (infant_data['diff'].iloc[ii] + infant_data['same'].iloc[ii]) <=0 :
            infant_data['score'].iloc[ii] = .5
        else:
            infant_data['score'].iloc[ii] = infant_data['score'].iloc[ii]/(infant_data['diff'].iloc[ii] + infant_data['same'].iloc[ii])
        
    
    #pdb.set_trace()
    #infant_data['diff'] = diff_score
    #infant_data['diff'][infant_data['diff'] <=0] = 0
    
    # same_score = infant_data['Familiar']- infant_data['HabEnd']
    # cat_score = diff_score.mean()/(diff_score.mean() + same_score.mean())
    # ci_low, ci_high = calc_bootstrap_ci(infant_data, ee, 'Infants')
    
    # infant_summary = pd.Series(['Infants', cat_score, ci_low,ci_high],index = model_summary.columns)
    
    return infant_data



for exn, ee in enumerate(exp):
    infant_df= extract_infant_data(ee)
    comps = ['same', 'diff'] #set what trial comparisons to pull out for the 'familiar' trial
    model_df= extract_model_data(ee,comps)
    
    #def calc_diff_CI(infant_df, model_df):
    diff_summary = pd.DataFrame(columns = ['model', 'score','CI_low', 'CI_high'])
    cat_diff=[]
    for ii in range(0,iter):
        
        temp_infant = infant_df.sample(len(infant_df), replace = True)
        
        #Calculate infant score
        diff_score = temp_infant['Novel'].mean()- temp_infant['HabEnd'].mean()
        if diff_score <0: diff_score = 0
        
        same_score = temp_infant['Familiar'].mean()- temp_infant['HabEnd'].mean()
        if same_score <0: same_score = 0
        
        try: #this is to catch div 0 errors
            infant_score = diff_score/(diff_score + same_score)
        except:
            #if divide by 0, retry and repeat loops
            ii = ii-1
            continue
        
        model_diff = []
        for mm in model_type:
            temp_model = model_df[model_df['model'] == mm]
            temp_model = temp_model.sample(len(temp_model), replace = True)
            model_score = temp_model['score'].mean()
            
            '''
            #Calculate model score
            diff_score =temp_model['error'][(temp_model['skel_cat'] == 'diff') & (temp_model['sf_cat'] == 'diff')].mean()
            same_score = temp_model['error'][(temp_model['skel_cat'] == comps[0]) & (temp_model['sf_cat'] == comps[1])].mean()
    
            #
            try:
                model_score = diff_score/ (diff_score + same_score)
                
                
                
            except:
                #if divide by zero, break out of loops
                model_score = .5
                
            '''
            model_diff.append(infant_score - model_score)
        cat_diff.append(model_diff)        
    
    
    
    cat_diff = np.array(cat_diff)
    
    for mm in enumerate(model_type):
        
        diff_score = infant_df['Novel']- infant_df['HabEnd']
        same_score = infant_df['Familiar']- infant_df['HabEnd']
        infant_score = diff_score.mean()/(diff_score.mean() + same_score.mean())
        
        model_score = model_df[model_df['model'] == mm[1]]['score'].mean()
        
        score = infant_score - model_score
        ci_low = np.percentile(cat_diff[:,mm[0]], alpha*100)
        ci_high= np.percentile(cat_diff[:,mm[0]], 100-alpha*100)
        
        print(mm[1], infant_score, model_score,score )
        
        diff_summary = diff_summary.append(pd.Series([mm[1], score,ci_low, ci_high], index = diff_summary.columns), ignore_index = True)
        
    pdb.set_trace()    
    
            
