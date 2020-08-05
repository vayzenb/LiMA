rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(boot)

setwd('C:/Users/vayze/Desktop/GitHub_Repos/LiMA')

exp = c('Exp1', 'Exp2')
ModelType= c('GBJ', 'GIST', 'FF_IN', 'R_IN', 'FF_SN', 'R_SN')
classifier = c("OCS", "ISOF")

ModelCols = c('Exp', 'Model', 'Obj1', 'Obj2', 'Skel', 'SF', 'trAcc_ocs', 'tsAcc_ocs','trAcc_isof', 'tsAcc_isof')

iter = 10000

alpha = .05
for (ee in 1:length(exp)){
  #Load infant data
  df = read.table(paste("Results/LiMA_Exp", ee,"_allModels_multiClass.csv", sep=""),header = FALSE, sep=",")
  
  
  
  colnames(df) = ModelCols
  
  #Create coded score for different skeleton objects (e.g., a 0 cat accuracy is actually 100% correct)
  df$OCS = df$tsAcc_ocs
  df$OCS[df$Skel == 'Diff'] = 1-df$OCS[df$Skel == 'Diff']
  
  df$ISOF = df$tsAcc_isof
  df$ISOF[df$Skel == 'Diff'] = 1-df$ISOF[df$Skel == 'Diff']
  
  
  
  #set up empty matrices for each condition (2) and classifier(2)  (the number of rows in matrix corresponds to the model)
  bootMat.model = list(matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter))
  
  #Start boot test
  for (ii in 1:iter){

    n = 1
    for (cl in classifier){
        for (mm in 1:length(ModelType)){
          tempMAT = df[df$Model == ModelType[mm],]
          
          #Sample with replacement
          tempMAT = sample_n(tempMAT,nrow(tempMAT), replace = TRUE)
          #Add to appropriate matrix
          bootMat.model[[n]][mm,ii] = (mean(tempMAT[[cl]][tempMAT$Skel=="Same"], na.rm = TRUE) +
                                         mean(tempMAT[[cl]][tempMAT$Skel=="Diff"], na.rm = TRUE))/2
          
        }
        n= n +1
      }
  }
  
  
  #Write to file
  ModelSummary = matrix(0,length(bootMat.model)*length(ModelType),5)
  
  
  ms = 1
  n=1
  for (cl in classifier){
      for (mm in 1:length(ModelType)){

        
        tempMAT = df[df$Model == ModelType[mm],]
        
        tempMean = (mean(tempMAT[[cl]][tempMAT$Skel=="Same"], na.rm = TRUE) +
                      mean(tempMAT[[cl]][tempMAT$Skel=="Diff"], na.rm = TRUE))/2
        
        CI = quantile(bootMat.model[[ms]][mm,], probs = c(alpha/2, 1-alpha/2));
        ModelSummary[n,] = c(ModelType[mm], cl, tempMean,CI[1],CI[2])
        n= n + 1
        
        
      }
   ms = ms +1
    }
  
  colnames(ModelSummary) = c("Model", "Classifier", "Acc", "CI_Low", "CI_High")
  
  assign(paste(exp[ee], '.Models_MSL', sep=""), ModelSummary)
  
}

save(Exp1.Models_MSL, Exp2.Models_MSL, file="Infant_Data/LiMA_Model_Data_MultiItem.RData")




