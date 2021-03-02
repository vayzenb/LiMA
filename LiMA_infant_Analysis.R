rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(boot)

setwd('B:/home/vayzenbe/GitHub_Repos/LiMA')

exp = c('Exp1', 'Exp2')
#cond = c('View', 'SF', 'Skel')
#ModelType= c( 'CorNet_Z', 'CorNet_S',"ResNext-TC-SAY", "ResNet_IN", "ResNet_SN")
#classifier = c("OCS", "ISOF")

#ModelCols = c('Exp', 'Model', 'Obj1', 'Obj2', 'Skel', 'SF', 'trAcc_ocs', 'tsAcc_ocs','trAcc_isof', 'tsAcc_isof', "Cond")

skel = list(list("23", "31", "266"), list("31_0", "31_50"))
SF= c("Skel", "Bulge")

sz = 20

iter = 10000
alpha = .05


for (ee in 1:length(exp)){
  df = read.table(paste("Infant_Data/Experiment_",ee,"_Infant_Data.csv", sep=""),header = TRUE, sep=",")
  df$skel1 = as.character(df$skel1)
  df$skel2 = as.character(df$skel2)
  
  
  
  df.sum = matrix(0, 30,7)
  n = 1
  for (sk1 in skel[[ee]]){
    for (sf1 in SF){
      for (sk2 in skel[[ee]]){
        for (sf2 in SF){
          #print(paste(sk1, sf1,sk2,sf2, sep = " "))
          #Extract items with specific hab/test combo
          tempInfant = df[df$skel1 == sk1 & df$sf1 == sf1 & df$skel2 == sk2 & df$sf2 == sf2,]
          
          
          if(nrow(tempInfant) != 0){
            
            print(paste(sk1, sf1,sk2,sf2, sep = " "))
            #Normalize to end of hab
            novelDiff = mean(tempInfant$Novel, na.rm = TRUE) - mean(tempInfant$HabEnd, na.rm = TRUE)
            famDiff = mean(tempInfant$Familiar, na.rm = TRUE) - mean(tempInfant$HabEnd, na.rm = TRUE)
            if(novelDiff < 0){novelDiff = 0}
            if(famDiff < 0){famDiff = 0}
            cat_score =  novelDiff / (novelDiff + famDiff)
            
            #data_row = as.data.frame(tempInfant[1,5:8], cat_score)
            df.sum[n,] =c(sk1, sf1, sk2, sf2, cat_score, 0, 0)
             n = n +1
          }

          
        }
      }
    }
    
  }
  df.sum = df.sum[df.sum[,5] != 0,]
  
  bootMat.infant = matrix(0,1,iter)
  
  
  #Start boot test
  for (ii in 1:iter){
    #Sample with replacement for infants
    tempInfant = sample_n(df.infant,nrow(df.infant), replace = TRUE)
    
    novelDiff = mean(tempInfant$Novel) - mean(tempInfant$HabEnd)
    famDiff = mean(tempInfant$Familiar) - mean(tempInfant$HabEnd)
    
    
  }
}
