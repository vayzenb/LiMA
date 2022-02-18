# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:57:17 2021
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



model_cols = ['Model', 'Skel1', 'SF1', 'hab_trials', 'hab_start', 'hab_end', 'skel2','sf2', 'skel_cat', "sf_cat", 'error']
summary_cols = ['Model', 'Condition', 'hab_num', 'hab_start', 'hab_end', 'Acc', 'CI_Low', 'CI_High']

alpha = .05
iter = 5000


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
    
    
def calc_bootstrap_ci(df, exp, model):
    """
    Calculate bootstrap confidene intervals for all models
    Parameters
    ----------
    exp : which experiment to load
        DESCRIPTION.
    object_data : a dataframe with object data
        DESCRIPTION.
    Returns
    -------
    CIs for models
    """
    
    
    #ci_low = []
    #ci_high = []
    
    
    boot_vals = []
    
    for ii in range(0,iter):              
        temp_df = df.sample(len(df), replace = True)
        
        if model != 'Infants':
            boot_vals.append(temp_df['score'].mean())
        else:
            diff_score = temp_df['Novel'].mean()- temp_df['HabEnd'].mean()
            if diff_score <0: diff_score = 0
            same_score = temp_df['Familiar'].mean()- temp_df['HabEnd'].mean()
            if same_score <0: same_score = 0
            
            try: #this is to catch div 0 errors
                cat_score = diff_score/(diff_score + same_score)
                boot_vals.append(cat_score)
            except:
                ii = ii-1
            
    
    
    #ci_low = np.percentile(boot_vals, alpha/2*100)
    #ci_high= np.percentile(boot_vals, 100-alpha/2*100)

    ci_low = np.percentile(boot_vals, alpha*100)
    ci_high= np.percentile(boot_vals, 100-alpha*100)
    #model_summary['CI_low'] = pd.Series(ci_low)
    #model_summary['CI_high'] = pd.Series(ci_high)
    
    #model_summary.to_csv(f'{curr_dir}/Infant_Data/{ee}_AE_model_results.csv', index=False)
    
    return ci_low, ci_high
    

def extract_infant_data(exp):
    """
    Organize infant data like models
    """
    infant_data = pd.read_csv(f'{curr_dir}/Infant_Data/{exp}_Infant_Data.csv')
    infant_data['skel1'][infant_data['skel1'] == 266] = 26
   
    diff_score = infant_data['Novel']- infant_data['HabEnd']
    same_score = infant_data['Familiar']- infant_data['HabEnd']
    cat_score = diff_score.mean()/(diff_score.mean() + same_score.mean())
    ci_low, ci_high = calc_bootstrap_ci(infant_data, ee, 'Infants')
    
    infant_summary = pd.Series(['Infants', cat_score, ci_low,ci_high],index = model_summary.columns)
    
    return infant_summary


for exn, ee in enumerate(exp):
    print(ee)
    
    '''
    Do one-shot categorization across surface form differences
    '''
    comps = ['same', 'diff'] #set what trial comparisons to pull out for the 'familiar' trial
    object_data = extract_model_data(ee,comps)
    model_summary = pd.DataFrame(object_data.groupby(['model'],as_index = False, sort = False).mean())
    model_summary['CI_low'] = np.nan
    model_summary['CI_high'] = np.nan
    
    for mn, mm in enumerate(model_type):
        print(mm)
        model_summary['model'][model_summary['model'] == mm] = actual_name[mn]
        curr_model = object_data[object_data['model'] == mm]
        model_summary['CI_low'][mn], model_summary['CI_high'][mn]= calc_bootstrap_ci(curr_model, ee, mm)
         
    
    #pdb.set_trace()

    infant_summary = extract_infant_data(ee)
    
    model_summary = model_summary.append(infant_summary, ignore_index = True)
    
    
    
    model_summary.to_csv(f'{curr_dir}/Infant_Data/{ee}_skel_cat.csv', index = False)
    
    '''
    Do one-shot categorization across skeleton differences
    '''
    comps = ['diff', 'same'] #set what trial comparisons to pull out for the 'familiar' trial
    object_data = extract_model_data(ee,comps)
    model_summary = pd.DataFrame(object_data.groupby(['model'],as_index = False, sort = False).mean())
    model_summary['CI_low'] = np.nan
    model_summary['CI_high'] = np.nan
    
    for mn, mm in enumerate(model_type):
        print(mm)
        model_summary['model'][model_summary['model'] == mm] = actual_name[mn]
        curr_model = object_data[object_data['model'] == mm]
        model_summary['CI_low'][mn], model_summary['CI_high'][mn]= calc_bootstrap_ci(curr_model, ee, mm)
    
    
    
    model_summary.to_csv(f'{curr_dir}/Infant_Data/{ee}_sf_cat.csv', index = False)
    
    
    '''
    temp_df = infant_data[(infant_data['skel1'].astype(str) == sk) & (infant_data['sf1'] == sf)]
    
    diff_score = temp_df['Novel'].mean()  - temp_df['HabEnd'].mean()
    #diff_score[diff_score<0] = temp_df['Novel'][diff_score<0]
    
    same_score = temp_df['Familiar'].mean()-temp_df['HabEnd'].mean()
    #same_score[same_score<0] = temp_df['Novel'][same_score<0]
    
    cat_score = diff_score / (diff_score + same_score)
    #cat_score = cat_score.mean()
    
    temp_summary = pd.Series([ee, 'infant', sk, sf, cat_score], index = object_summary.columns)
    object_summary = object_summary.append(temp_summary, ignore_index = True )
        
    
    pdb.set_trace()
    '''      
            

    
    
    
    #pdb.set_trace()
            
            
                
    
        

        #pdb.set_trace()