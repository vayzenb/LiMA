rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(boot)

setwd('C:/Users/vayze/Desktop/GitHub_Repos/LiMA')

exp = c('Exp1', 'Exp2')
ModelType= c('GBJ', 'GIST', 'FF_IN', 'R_IN', 'FF_SN', 'R_SN')
classifier = c("OCS", "ISOF")

ModelCols = c('Exp', 'Model', 'Obj1', 'Obj2', 'Skel', 'SF', 'trAcc_ocs', 'tsAcc_ocs','trAcc_isof', 'tsAcc_isof', "Cond")

imSize = c(10, 20, 30, 40, 50)

iter = 5000

alpha = .05
for (ee in 1:length(exp)){
  
  #set up empty matrices for each condition and classifier (the number of rows corresponds to the model
  bootMat.model = list(matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter),
                       matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter),matrix(0,length(ModelType),iter),
                       matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter),matrix(0,length(ModelType),iter),
                       matrix(0,length(ModelType),iter))
  
  ModelSummary = matrix(0,length(bootMat.model)*length(ModelType),6)
  n= 1
  ms = 1
  for (sz in imSize){
    print(sz)
  
    df = read.table(paste("Results/LiMA_Exp", ee,"_allModels_AllClassifiers_Size", sz,".csv", sep=""),header = FALSE, sep=",")
    df$Cond = sz
    
    colnames(df) = ModelCols
    df[df == "None"] = NA
    df = na.omit(df)
    #Create coded score for different skeleton objects (e.g., a 0 cat accuracy is actually 100% correct)
    df$OCS = as.numeric(as.character(df$tsAcc_ocs))
    df$OCS[df$Skel == 'Diff'] = 1-df$OCS[df$Skel == 'Diff']
    
    df$ISOF = as.numeric(as.character(df$tsAcc_isof))
    df$ISOF[df$Skel == 'Diff'] = 1-df$ISOF[df$Skel == 'Diff']
    
    #Start boot test
    for (ii in 1:iter){
       for (mm in 1:length(ModelType)){
    
            tempMAT = df[df$Model == ModelType[mm] & df$SF == "Same",]
        
           #Sample with replacement
           tempMAT = sample_n(tempMAT,nrow(tempMAT), replace = TRUE)
           
           clNum = 0
           for (cl in classifier){
           #Add to appropriate matrix
           bootMat.model[[n + clNum]][mm,ii] = (mean(tempMAT[[cl]][tempMAT$Skel=="Same"], na.rm = TRUE) +
                                          mean(tempMAT[[cl]][tempMAT$Skel=="Diff"], na.rm = TRUE))/2
           clNum = clNum + length(imSize)
           
         }
        
      }
      
     }
    
    
    for (mm in 1:length(ModelType)){
      #Select appropriate data depending on condition
      tempMAT = df[df$Model == ModelType[mm] & df$SF == "Same",]
      
 
      clNum = 0
      for (cl in classifier){
        tempMean = (mean(tempMAT[[cl]][tempMAT$Skel=="Same"], na.rm = TRUE) +
                      mean(tempMAT[[cl]][tempMAT$Skel=="Diff"], na.rm = TRUE))/2
        
        CI = quantile(bootMat.model[[n+ clNum]][mm,], probs = c(alpha/2, 1-alpha/2));
        ModelSummary[ms,] = c(ModelType[mm], cl, sz, tempMean,CI[1],CI[2])
        ms= ms +1  
        clNum = clNum + length(imSize)
      }
      
    }
    
    n= n +1
  }
  
  
  #Write to file
  
  
  colnames(ModelSummary) = c("Model", "Classifier", "Condition", "Acc", "CI_Low", "CI_High")
  
  assign(paste(exp[ee], '.Models_Size', sep=""), ModelSummary)
}

save(Exp1.Models_Size, Exp2.Models_Size, file="Infant_Data/LiMA_Model_Data_Size.RData")
   


    
