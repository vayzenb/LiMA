rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(boot)

setwd('B:/home/vayzenbe/GitHub_Repos/LiMA')

exp = c('Exp1', 'Exp2')
cond = c('View', 'SF', 'Skel')
ModelType= c( 'CorNet_Z', 'CorNet_S',"ResNext-TC-SAY", "ResNet_IN", "ResNet_SN")
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
  
  #This codes objects with same skel at 1 and objects with diff skels as 0
  df$OCS = df$tsAcc_ocs
  df$OCS[df$Skel == 'Diff'] = 1-df$OCS[df$Skel == 'Diff']
  
  df$ISOF = df$tsAcc_isof
  df$ISOF[df$Skel == 'Diff'] = 1-df$ISOF[df$Skel == 'Diff']
  
  #This codes objects with same SF at 1 and objects with diff SFs as 0
  df$OCS_SF = df$tsAcc_ocs
  df$OCS_SF[df$SF == 'Diff'] = 1-df$OCS_SF[df$SF == 'Diff']
  
  df$ISOF_SF = df$tsAcc_isof
  df$ISOF_SF[df$SF == 'Diff'] = 1-df$ISOF_SF[df$SF == 'Diff']
  
  

  #set up empty matrices for each condition (3) and classifier(2)  (the number of rows in matrix corresponds to the model)
  bootMat.infant = matrix(0,1,iter)
  bootMat.model = list(matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter),matrix(0,length(ModelType),iter),
                       matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter))
  
  #Start boot test
  for (ii in 1:iter){
    #Sample with replacement for infants
    tempInfant = sample_n(df.infant,nrow(df.infant), replace = TRUE)
    
    novelDiff = mean(tempInfant$Novel) - mean(tempInfant$HabEnd)
    famDiff = mean(tempInfant$Familiar) - mean(tempInfant$HabEnd)
    if(novelDiff < 0){novelDiff = 0}
    if(famDiff < 0){famDiff = 0}
    if(is.nan(novelDiff / (novelDiff + famDiff)) == TRUE){
      bootMat.infant[1,ii] = 0
    }else {
      bootMat.infant[1,ii] =  novelDiff / (novelDiff + famDiff)  
    }
    
    
    n = 1
   for (cl in classifier){
    for (cc in cond){
     for (mm in 1:length(ModelType)){
      
       #Select appropriate data depending on condition
       # if(cc == 'Ident' ) { #same object
       #   temp_fam = df[df$Model == ModelType[mm] & df$skel_cat == "same" & df$sf_cat == "same",]
       #   temp_nov = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "same",]
       #   
       # }else if (cc == 'SF' )  { #generalize across SF
       #   temp_fam = df[df$Model == ModelType[mm] & df$skel_cat == "same" & df$sf_cat == "diff",]
       #   temp_nov = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "diff",]
       #   
       # }else if (cc == "Skel") { #generalize across skel
       #   temp_fam = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "same",]
       #   temp_nov = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "diff",]
       # }
       # 
       # 
       # #Sample familiar and novel errors with replacement
       # temp_fam = sample_n(temp_fam,nrow(temp_fam), replace = TRUE) 
       # temp_nov = sample_n(temp_nov,nrow(temp_fam), replace = TRUE)  #note, there are more rows in the novel array, so it is subsampled to match familiar
       # 
       # fam_error = mean(temp_fam$error, na.rm = TRUE)
       # nov_error = mean(temp_nov$error, na.rm = TRUE)
       # bootMat.model[[n]][mm,ii] = nov_error/(fam_error + nov_error)
       
       #Select appropriate data depending on condition
      if(cc == 'View' ) {
        temp_fam = df[df$Model == ModelType[mm] & df$Skel == "Same" & df$SF == "Same",]
        temp_nov = df[df$Model == ModelType[mm] & df$Skel == "Diff" & df$SF == "Same",]
       
      }else if (cc == 'SF' )  {
        temp_fam = df[df$Model == ModelType[mm] & df$Skel == "Same" & df$SF == "Diff",]
        temp_nov = df[df$Model == ModelType[mm] & df$Skel == "Diff" & df$SF == "Diff",]
       
      }else if (cc == "Skel") {
        temp_fam = df[df$Model == ModelType[mm] & df$Skel == "Diff" & df$SF == "Same",]
        temp_nov = df[df$Model == ModelType[mm] & df$Skel == "Diff" & df$SF == "Diff",]
      }
       
       #Sample familiar and novel errors with replacement
       temp_fam = sample_n(temp_fam,nrow(temp_fam), replace = TRUE) 
       temp_nov = sample_n(temp_nov,nrow(temp_fam), replace = TRUE)  #note, there are more rows in the novel array, so it is subsampled to match familiar
       

       if (cc == "SF" | cc == "View"){
         fam_error = mean(temp_fam[[cl]], na.rm = TRUE)
         nov_error = mean(temp_nov[[cl]], na.rm = TRUE)
        
         
       }else{
         fam_error = mean(temp_fam[[paste(cl, "_SF",sep="")]], na.rm = TRUE)
         nov_error = mean(temp_nov[[paste(cl, "_SF",sep="")]], na.rm = TRUE)
         
       }
       
       bootMat.model[[n]][mm,ii] = (fam_error + nov_error)/2
       #Add to appropriate matrix
       # if (cc == 'SF' | cc == 'View')  {
       # bootMat.model[[n]][mm,ii] = (mean(tempMAT[[cl]][tempMAT$Skel=="Same"], na.rm = TRUE) +
       #                                mean(tempMAT[[cl]][tempMAT$Skel=="Diff"], na.rm = TRUE))/2
       # }else if (cc == "Skel") {
       #   bootMat.model[[n]][mm,ii] = (mean(tempMAT[[paste(cl, "_SF",sep="")]][tempMAT$SF=="Same"], na.rm = TRUE) +
       #                                  mean(tempMAT[[paste(cl, "_SF",sep="")]][tempMAT$SF=="Diff"], na.rm = TRUE))/2
       # }
       
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
          
        }else if (cc == "Skel") {
          tempMAT = df[df$Model == ModelType[mm] & df$Skel == "Diff" & df$Cond == "SF",]
        }
        
        if (cc == 'SF' | cc == 'View')  {
          tempMean = (mean(tempMAT[[cl]][tempMAT$Skel=="Same"], na.rm = TRUE) +
                                         mean(tempMAT[[cl]][tempMAT$Skel=="Diff"], na.rm = TRUE))/2
        }else if (cc == "Skel") {
          tempMean = (mean(tempMAT[[paste(cl, "_SF",sep="")]][tempMAT$SF=="Same"], na.rm = TRUE) +
                                         mean(tempMAT[[paste(cl, "_SF",sep="")]][tempMAT$SF=="Diff"], na.rm = TRUE))/2
        }
          
        #tempMean = (mean(tempMAT[[cl]][tempMAT$Skel=="Same"], na.rm = TRUE) +
         #             mean(tempMAT[[cl]][tempMAT$Skel=="Diff"], na.rm = TRUE))/2
        
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
   


    
