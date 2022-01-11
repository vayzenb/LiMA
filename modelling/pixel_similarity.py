import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.image as img
from matplotlib import pyplot as plt
from glob import glob as glob
import time
import pdb

exp = ['Exp1', 'Exp2']
skel = [['23','31', '26'],['31_0', '31_50']]
SF = ['Skel', 'Bulge']

frame_dir = '/user_data/vayzenbe/GitHub_Repos/LiMA/Frames'

def load_frames(frame_files):
    all_frames = []
    for ff in frame_files:
        all_frames.append(np.array(Image.open(ff).convert('L')).astype(float))

    return all_frames

def compare_frames(frame_ims1, frame_ims2):
    all_dists =[]
    for ff in enumerate(frame_ims1):
        all_dists.append(np.linalg.norm(frame_ims1[ff[0]] - frame_ims2[ff[0]]))
            
    return all_dists

def compare_frames_pw(frame_ims1, frame_ims2):
    all_dists =[]
    for ff1 in frame_ims1:
        temp_dist = []
        for ff2 in frame_ims2:
            temp_dist.append(np.linalg.norm(ff1 - ff2))
        
        all_dists.append(np.mean(temp_dist))
            
    return all_dists

for ee in enumerate(exp):
    pair_summary = pd.DataFrame()
    pair_summary_pw = pd.DataFrame()

    object_summary = pd.DataFrame(columns = ['skel1','sf1','skel2','sf2','skel_cond','sf_cond','dist'])
    object_summary_pw = pd.DataFrame(columns = ['skel1','sf1','skel2','sf2','skel_cond','sf_cond','dist'])

    dist_mat = np.zeros((len(skel[ee[0]])*2,len(skel[ee[0]])*2))
    dist_mat_pw = np.zeros((len(skel[ee[0]])*2,len(skel[ee[0]])*2))
    nn1 = 0
    for sk1 in skel[ee[0]]:
        for sf1 in SF:
            frame_files1 = glob(f'{frame_dir}/Figure_{sk1}_{sf1}/*.jpg')

            frame_ims1 = load_frames(frame_files1)
            nn2 = 0
            for sk2 in skel[ee[0]]:
                for sf2 in SF:
                    frame_files2 = glob(f'{frame_dir}/Figure_{sk2}_{sf2}/*.jpg')
                    frame_ims2 = load_frames(frame_files2)

                    #compare videos to matched frames (e.g., frame1 to frame1)
                    all_dists = compare_frames(frame_ims1, frame_ims2)
                    pair_summary[f'{sk1}{sf1}_{sk2}{sf2}'] =pd.Series(all_dists)
                    dist_mat[nn1,nn2] = np.mean(all_dists)
                    
                    if sk1 == sk2: 
                        skel_cond ='same'
                    else:
                        skel_cond ='diff'
                    
                    if sf1 == sf2:
                        sf_cond = 'same'
                    else:
                        sf_cond = 'diff'

                    curr_data = pd.Series([sk1, sf1, sk2,sf2,skel_cond, sf_cond, np.mean(all_dists)], index= object_summary.columns)
                    object_summary = object_summary.append(curr_data, ignore_index = True)

                    #compare video frames in a pairwise (pw) way (e.g., frame1 to frame1-300, frame2 to frame1-300 etc)
                    all_dists_pw = compare_frames_pw(frame_ims1, frame_ims2)
                    pair_summary_pw[f'{sk1}{sf1}_{sk2}{sf2}'] =pd.Series(all_dists_pw)
                    dist_mat_pw[nn1,nn2] = np.mean(all_dists_pw)

                    curr_data = pd.Series([sk1, sf1, sk2,sf2,skel_cond, sf_cond, np.mean(all_dists_pw)], index= object_summary_pw.columns)
                    object_summary_pw = object_summary_pw.append(curr_data, ignore_index = True)
                    
                    nn2 +=1
            nn1 +=1
    
    #pdb.set_trace()
    np.savetxt(f'/user_data/vayzenbe/GitHub_Repos/LiMA/modelling/pixel_data/{ee[1]}_pixel_dist_mat.csv',dist_mat,delimiter= ',', fmt='%1.3f')
    np.savetxt(f'/user_data/vayzenbe/GitHub_Repos/LiMA/modelling/pixel_data/{ee[1]}_pixel_dist_mat_pw.csv',dist_mat_pw,delimiter= ',',fmt='%1.3f')
    
    pair_summary.to_csv(f'/user_data/vayzenbe/GitHub_Repos/LiMA/modelling/pixel_data/{ee[1]}_pixel_dist_frames.csv', index=False)
    pair_summary_pw.to_csv(f'/user_data/vayzenbe/GitHub_Repos/LiMA/modelling/pixel_data/{ee[1]}_pixel_dist_frames_pw.csv', index=False)

    object_summary.to_csv(f'/user_data/vayzenbe/GitHub_Repos/LiMA/modelling/pixel_data/{ee[1]}_pixel_dist_object.csv', index=False)
    object_summary_pw.to_csv(f'/user_data/vayzenbe/GitHub_Repos/LiMA/modelling/pixel_data/{ee[1]}_pixel_dist_object_pw.csv', index=False)







                



            


