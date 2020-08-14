rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(boot)

setwd('C:/Users/vayze/Desktop/GitHub_Repos/LiMA')

exp = c('Exp1', 'Exp2')
cond = c('View', 'SF')
ModelType= c('GBJ', 'GIST', 'FF_IN', 'R_IN', 'FF_SN', 'R_SN')
classifier = c("OCS", "ISOF")

ModelCols = c('Exp', 'Model', 'Obj1', 'Obj2', 'Skel', 'SF', 'trAcc_ocs', 'tsAcc_ocs','trAcc_isof', 'tsAcc_isof', "Cond")

sz = 20

iter = 10000

alpha = .05
for (ee in 1:length(exp)){
 #Load infant data
  df.infant = read.table(paste("Infant_Data/Experiment_",ee,"_Infant_Data.csv", sep=""),header = TRUE, sep=",")
  df.sf = read.table(paste("Results/LiMA_Exp", ee,"_allModels_OSL.csv", sep=""),header = FALSE, sep=",")
  df.sf$Cond = "SF"
  
  df = df.sf
  colnames(df) = ModelCols
  
  #Create coded score for different skeleton objects (e.g., a 0 cat accuracy is actually 100% correct)
  df$OCS = df$tsAcc_ocs
  df$OCS[df$Skel == 'Diff'] = 1-df$OCS[df$Skel == 'Diff']
  
  df$ISOF = df$tsAcc_isof
  df$ISOF[df$Skel == 'Diff'] = 1-df$ISOF[df$Skel == 'Diff']
  
  

  #set up empty matrices for each condition (2) and classifier(2)  (the number of rows in matrix corresponds to the model)
  bootMat.infant = matrix(0,1,iter)
  bootMat.model = list(matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter),
                       matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter))
  
  #Start boot test
  for (ii in 1:iter){
    #Sample with replacement for infants
    tempInfant = sample_n(df.infant,nrow(df.infant), replace = TRUE)
    
    novelDiff = mean(tempInfant$Novel) - mean(tempInfant$HabEnd)
    famDiff = mean(tempInfant$Familiar) - mean(tempInfant$HabEnd)
    if(novelDiff < 0){novelDiff = 0}
    if(famDiff < 0){famDiff = 0}
    bootMat.infant[1,ii] =  novelDiff / (novelDiff + famDiff)
    
    n = 1
   for (cl in classifier){
    for (cc in cond){
     for (mm in 1:length(ModelType)){
      
       #Select appropriate data depending on condition
      if(cc == 'View' ) {
       tempMAT = df[df$Model == ModelType[mm] & df$SF == "Same" & df$Cond == "SF",]
       
      }else if (cc == 'SF' )  {
        tempMAT = df[df$Model == ModelType[mm] & df$SF == "Diff" & df$Cond == "SF",]
       
      }else if (cc == "Size") {
        tempMAT = df[df$Model == ModelType[mm] & df$SF == "Same" & df$Cond == "Size",]
      }
       
       #Sample with replacement
       tempMAT = sample_n(tempMAT,nrow(tempMAT), replace = TRUE)
       #Add to appropriate matrix
       bootMat.model[[n]][mm,ii] = (mean(tempMAT[[cl]][tempMAT$Skel=="Same"], na.rm = TRUE) +
                                      mean(tempMAT[[cl]][tempMAT$Skel=="Diff"], na.rm = TRUE))/2
       
     }
      n= n +1
    }
   }
    
  }
  
  
  #Write to file
  ModelSummary = matrix(0,length(bootMat.model)*length(ModelType) +1,6)
  
  #This section comiles the summary data
  
  novelDiff = mean(df.infant$Novel) - mean(df.infant$HabEnd)
  famDiff = mean(df.infant$Familiar) - mean(df.infant$HabEnd)
  CI = quantile(bootMat.infant, probs = c(alpha/2, 1-alpha/2));
  ModelSummary[1,] = c("Infant", "Infant", "SF", novelDiff/(novelDiff + famDiff),CI[1],CI[2])
  
  ms = 2
  n=1
  for (cl in classifier){
    for (cc in cond){
      for (mm in 1:length(ModelType)){
        
        #Select appropriate data depending on condition
        if(cc == 'View' ) {
          tempMAT = df[df$Model == ModelType[mm] & df$SF == "Same" & df$Cond == "SF",]
          
        }else if (cc == 'SF' )  {
          tempMAT = df[df$Model == ModelType[mm] & df$SF == "Diff" & df$Cond == "SF",]
          
        }else if (cc == "Size") {
          tempMAT = df[df$Model == ModelType[mm] & df$SF == "Same" & df$Cond == "Size",]
        }
        
        tempMean = (mean(tempMAT[[cl]][tempMAT$Skel=="Same"], na.rm = TRUE) +
                      mean(tempMAT[[cl]][tempMAT$Skel=="Diff"], na.rm = TRUE))/2
        
        CI = quantile(bootMat.model[[n]][mm,], probs = c(alpha/2, 1-alpha/2));
        ModelSummary[ms,] = c(ModelType[mm], cl, cc, tempMean,CI[1],CI[2])
        ms= ms +1  

        
      }
      n= n + 1
    }
  }
  
  colnames(ModelSummary) = c("Model", "Classifier", "Condition", "Acc", "CI_Low", "CI_High")
  
  assign(paste(exp[ee], '.Models', sep=""), ModelSummary)
  
}

save(Exp1.Models, Exp2.Models, file="Infant_Data/LiMA_Model_Data.RData")
   


    
