rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(boot)

setwd('B:/home/vayzenbe/GitHub_Repos/LiMA')

exp = c('Exp1', 'Exp2')

#ident[ical] is whether it can discriminate between objes with the same SFs, diff skels; tested with habituated object
#Sf is whether it generalized across an SF chnage (one-shot by skeleton)
#skel is whether igeneralied  across a skel change (one-shot by )
cond = c('Ident', 'SF', 'Skel') 
ModelType= c( 'CorNet_Z', 'CorNet_S',"SayCam", "ResNet_IN", "ResNet_SN")
#ModelType= c("SayCam")
skel = list(list('23', '31', '26'), list('31_0', '31_50'))
SF = list('Skel', 'Bulge')

ModelCols = c('Model', 'Skel1', 'SF1', 'hab_trials', 'hab_start', 'hab_end', 'skel2','sf2', 'skel_cat', "sf_cat", 'error')
SummaryCols = c('Model', 'Condition', 'hab_num', 'hab_start', 'hab_end', 'Acc', 'CI_Low', 'CI_High')
alpha = .05
iter = 10000




for (ee in 1:length(exp)){
  #combine all model data
  df = NULL

  for(mm in ModelType){
    
    for (sk in skel[[ee]]){
      for (sf in SF){
        temp_df = read.table(paste("Results/AE/Exp",ee,'_',mm,'_Figure_',sk,'_',sf,'_Result.csv', sep=""),header = FALSE, sep=",")
       
        df =rbind(df, temp_df)
      }
      
    }
    
  }
  colnames(df) = ModelCols
      
  #set up empty matrices for each condition (3) (the number of rows in matrix corresponds to the model)
  bootMat.model = list(matrix(0,length(ModelType),iter), matrix(0,length(ModelType),iter),matrix(0,length(ModelType),iter))
  
  #Start boot test
  for (ii in 1:iter){
    
    n = 1
      for (cc in cond){
        for (mm in 1:length(ModelType)){
          

          #Select appropriate data depending on condition
          if(cc == 'Ident' ) { #same object
            temp_fam = df[df$Model == ModelType[mm] & df$skel_cat == "same" & df$sf_cat == "same",]
            temp_nov = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "same",]
            
          }else if (cc == 'SF' )  { #generalize across SF
            temp_fam = df[df$Model == ModelType[mm] & df$skel_cat == "same" & df$sf_cat == "diff",]
            temp_nov = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "diff",]
            
          }else if (cc == "Skel") { #generalize across skel
            temp_fam = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "same",]
            temp_nov = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "diff",]
          }
          

          #Sample familiar and novel errors with replacement
          temp_fam = sample_n(temp_fam,nrow(temp_fam), replace = TRUE) 
          temp_nov = sample_n(temp_nov,nrow(temp_fam), replace = TRUE)  #note, there are more rows in the novel array, so it is subsampled to match familiar
          
          fam_error = mean(temp_fam$error, na.rm = TRUE)
          nov_error = mean(temp_nov$error, na.rm = TRUE)
          bootMat.model[[n]][mm,ii] = nov_error/(fam_error + nov_error)
          
          
        }
        n= n +1
      }
    }
    
  ModelSummary = NULL
  n = 1    
  for (cc in cond){
        
        for (mm in 1:length(ModelType)){

          #Select appropriate data depending on condition
          if(cc == 'Ident' ) { #same object
            temp_fam = df[df$Model == ModelType[mm] & df$skel_cat == "same" & df$sf_cat == "same",]
            temp_nov = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "same",]

            
          }else if (cc == 'SF' )  { #generalize across SF
            temp_fam = df[df$Model == ModelType[mm] & df$skel_cat == "same" & df$sf_cat == "diff",]
            temp_nov = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "diff",]
            
          }else if (cc == "Skel") { #generalize across skel
            temp_fam = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "same",]
            temp_nov = df[df$Model == ModelType[mm] & df$skel_cat == "diff" & df$sf_cat == "diff",]
          }
          hab_trials = mean(temp_fam$hab_trials, na.rm = TRUE)
          hab_start = mean(temp_fam$hab_start, na.rm = TRUE)
          hab_end = mean(temp_fam$hab_end, na.rm = TRUE)
          
          fam_error = mean(temp_fam$error, na.rm = TRUE)
          nov_error = mean(temp_nov$error, na.rm = TRUE)
          
          acc = nov_error/(fam_error + nov_error)
          CI = quantile(bootMat.model[[n]][mm,], probs = c(alpha/2, 1-alpha/2));
          
          
          result = c(ModelType[mm], cc,
                     as.numeric(hab_trials), as.numeric(hab_start), as.numeric(hab_end),
                     as.numeric(acc),as.numeric(CI[1]),as.numeric(CI[2]))
          #colnames(result) = SummaryCols
          #print(result)
          #ModelSummary = as.data.frame(Model = c(ModelSummary$Model, , Condition = as.character(), 
          #hab_num = as.numeric(), hab_start = as.numeric(), hab_end = as.numeric(),
          #Acc = as.numeric(), CI_Low = as.numeric(), CI_High = as.numeric())
          
          ModelSummary = rbind(ModelSummary, result)
        }
        n = n + 1
      }
      
  
  colnames(ModelSummary) = SummaryCols
  assign(paste(exp[ee], '.Models_AE', sep=""), ModelSummary)
  
  
}

save(Exp1.Models, Exp2.Models, file="Infant_Data/LiMA_AE_Data.RData")
