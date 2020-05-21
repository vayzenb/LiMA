rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(boot)

setwd('C:/Users/vayze/Desktop/GitHub_Repos/LiMA')

exp = c('Exp1', 'Exp2')
cond = c('', '_Size20')
models= c('GBJ', 'GIST', 'FF_IN', 'R_IN', 'FF_SN', 'R_SN')

ModelCols = c('Exp', 'Model', 'Obj1', 'Obj2', 'Skel', 'SF', 'trAcc_ocs', 'tsAcc_ocs','trAcc_isof', 'tsAcc_isof')
sz = 20

iter = 10000

alpha = .05
for (ee in 1:length(exp)){
 #Load infant data
  df.infant = read.table(paste("Infant_Data/Experiment_",ee,"_Infant_Data.csv", sep=""),header = TRUE, sep=",")
  df.mSF = read.table(paste("Results/LiMA_Exp", ee,"_allModels_AllClassifiers.csv", sep=""),header = FALSE, sep=",")
  colnames(df.mSF) = ModelCols
  df.mSize = read.table(paste("Results/LiMA_Exp", ee,"_allModels_AllClassifiers_Size", sz,".csv", sep=""),header = FALSE, sep=",")
  colnames(df.mSize) = ModelCols
  
  #Create coded score for different skeleton objects (e.g., a 0 cat accuracy is actually 100% correct)
  df.mSF$OCS = df.mSF$tsAcc_ocs
  df.mSF$OCS[df.mSF$Skel == 'Diff'] = 1-df.mSF$OCS[df.mSF$Skel == 'Diff']
  df.mSF$ISOF = df.mSF$tsAcc_isof
  df.mSF$ISOF[df.mSF$Skel == 'Diff'] = 1-df.mSF$ISOF[df.mSF$Skel == 'Diff']
  
  df.mSize$OCS = df.mSize$tsAcc_ocs
  df.mSize$OCS[df.mSize$Skel == 'Diff'] = 1-df.mSize$OCS[df.mSize$Skel == 'Diff']
  df.mSize$ISOF = df.mSize$tsAcc_isof
  df.mSize$ISOF[df.mSize$Skel == 'Diff'] = 1-df.mSize$ISOF[df.mSize$Skel == 'Diff']
  
  
  bootMat.infant = matrix(0,1,iter)
  
  #the number of rows is multiplied by 2 because of the two classifiers
  bootMat.view = matrix(0,length(models)*2,iter)
  bootMat.SF = matrix(0,length(models)*2,iter)
  bootMat.size = matrix(0,length(models)*2,iter)
  
  
  
  #Start boot test
  for (ii in 1:iter){
    
  #Sample with replacement
  tempInfant = sample_n(df.infant,nrow(df.infant), replace = TRUE)
  
  novelDiff = mean(tempInfant$Novel) - mean(tempInfant$HabEnd)
  famDiff = mean(tempInfant$Familiar) - mean(tempInfant$HabEnd)
  if(novelDiff < 0){novelDiff = 0}
  if(famDiff < 0){famDiff = 0}
  bootMat.infant[1,ii] =  novelDiff / (novelDiff + famDiff)

  
  #loop through models and resample for each model
  for (mm in 1:length(models)){
    #Pull out values from each model and each condition
    tempView = df.mSF[df.mSF$Model == models[mm] & df.mSF$SF == "Same",]
    tempSF = df.mSF[df.mSF$Model == models[mm] & df.mSF$SF == "Diff",]
    tempSize = df.mSize[df.mSize$Model == models[mm] & df.mSize$SF == "Same",]
    
    #sample with replacement
    tempView = sample_n(tempView,nrow(tempView), replace = TRUE)
    tempSF = sample_n(tempSF,nrow(tempSF), replace = TRUE)
    tempSize = sample_n(tempSize,nrow(tempSize), replace = TRUE)
    
    #Store new average ; first line is OSC value, second is ISOF value
    bootMat.view[mm, ii] = (mean(tempView$OCS[tempView$Skel=="Same"], na.rm = TRUE) +
      mean(tempView$OCS[tempView$Skel=="Diff"], na.rm = TRUE))/2
      
    bootMat.view[mm +length(models), ii] = (mean(tempView$ISOF[tempView$Skel=="Same"], na.rm = TRUE) +
      mean(tempView$ISOF[tempView$Skel=="Diff"], na.rm = TRUE))/2
      
    
    bootMat.SF[mm, ii] = (mean(tempSF$OCS[tempSF$Skel=="Same"], na.rm = TRUE) + 
      mean(tempSF$OCS[tempSF$Skel=="Diff"], na.rm = TRUE))/2
    bootMat.SF[mm +length(models), ii] = (mean(tempSF$ISOF[tempSF$Skel=="Same"], na.rm = TRUE) + 
      mean(tempSF$ISOF[tempSF$Skel=="Diff"], na.rm = TRUE))/2
    
    bootMat.size[mm, ii] = (mean(tempSize$OCS[tempSize$Skel=="Same"], na.rm = TRUE) +
      mean(tempSize$OCS[tempSize$Skel=="Diff"], na.rm = TRUE))/2
    bootMat.size[mm +length(models), ii] = (mean(tempSize$ISOF[tempSize$Skel=="Same"], na.rm = TRUE) +
      mean(tempSize$OCS[tempSize$Skel=="Diff"], na.rm = TRUE))/2
    
  }
  
  
  
  }
  #Create summary matrix
  ModelSummary.SF = matrix(0,nrow(bootMat.SF)+1,6)
  ModelSummary.View = matrix(0,nrow(bootMat.SF),6)
  ModelSummary.Size = matrix(0,nrow(bootMat.SF),6)
  
  #Calcualte infant data
  novelDiff = mean(df.infant$Novel) - mean(df.infant$HabEnd)
  famDiff = mean(df.infant$Familiar) - mean(df.infant$HabEnd)
  CIs = quantile(bootMat.infant, probs = c(alpha/2, 1-alpha/2));
  ModelSummary.SF[1,] = c("Infant", "Infant", "SF", novelDiff/(novelDiff+famDiff),CIs[1],CIs[2]) 
  
  ModelSummary.View[,1] = c(models, models)
  ModelSummary.View[,2] = c(rep("OCS", length(models)), rep("ISOF", length(models)))
  ModelSummary.View[,3] = "View"
  
  ModelSummary.SF[,1] = c(models, models)
  ModelSummary.SF[,2] = c(rep("OCS", length(models)), rep("ISOF", length(models)))
  ModelSummary.SF[,3] = "SF"
  
  
  ModelSummary.Size[,1] = c(models, models)
  ModelSummary.Size[,2] = c(rep("OCS", length(models)), rep("ISOF", length(models)))
  ModelSummary.Size[,3] = "Size"
  

  
  for (mm in 1:(nrow(bootMat.view))){
    ModelSummary.View[mm,4] = mean(df.mSF[df.mSF$Model == models[mm] & df.mSF$SF == "Same",])
    
      tempView = df.mSF[df.mSF$Model == models[mm] & df.mSF$SF == "Same",]
    tempSF = df.mSF[df.mSF$Model == models[mm] & df.mSF$SF == "Diff",]
    tempSize = df.mSize[df.mSize$Model == models[mm] & df.mSize$SF == "Same",]
    #Calculate CIs
    CIs.view = quantile(bootMat.view[mm,], probs = c(alpha/2, 1-alpha/2));
    CIs.SF = list(quantile(bootMat.SF[mm,], probs = c(alpha/2, 1-alpha/2)));
    CIs.size = list(quantile(bootMat.size[mm,], probs = c(alpha/2, 1-alpha/2)));
    
    ModelSummary.View[mm,5:6] =CIs.view
    ModelSummary.SF[mm,5:6] =CIs.view
    ModelSummary.Size[mm,5:6] =CIs.view
    
    
  }
  #Add all Values to Summary
  
}