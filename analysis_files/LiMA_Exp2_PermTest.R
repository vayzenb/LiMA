rm(list = ls())

library(ggplot2)
library(reshape2)
library(dplyr)
library(boot)

setwd('C:/Users/vayze/Desktop/GitHub_Repos/LiMA/')

load('Infant_Data/LiMA_data_all.RData')

iter = 10000
#Exp1.perm = sample_n(AllSubs.Exp1,nrow(subs)/2, replace = FALSE)
alpha = .05

AllSubs.Exp2$Familiar1 = as.numeric(as.character(AllSubs.Exp2$Familiar1))
AllSubs.Exp2$Novel1 = as.numeric(as.character(AllSubs.Exp2$Novel1))

bootMat_comp = matrix(0,1,iter)
bootMat_fam= matrix(0,1,iter)
bootMat_novel= matrix(0,1,iter)
bootMat_first_comp =matrix(0,1,iter)
bootMat_first_fam =matrix(0,1,iter)
bootMat_first_nov =matrix(0,1,iter)

for (ii in 1:iter){
  Exp2.boot = sample_n(AllSubs.Exp2,34, replace = TRUE)
  
  #novel vs. familiar 
  t_comp = t.test(Exp2.boot$Novel, Exp2.boot$Familiar, paired=TRUE)
  bootMat_comp[ii] = t_comp$statistic/sqrt(nrow(Exp2.boot))
  
  #familiar vs. habend
  t_fam = t.test(Exp2.boot$Familiar, Exp2.boot$HabEnd, paired=TRUE)
  bootMat_fam[ii] = t_fam$statistic/sqrt(nrow(Exp2.boot))
  
  #Novel vs. habend
  t_novel = t.test(Exp2.boot$Novel, Exp2.boot$HabEnd, paired=TRUE)
  bootMat_novel[ii] = t_novel$statistic/sqrt(nrow(Exp2.boot))
  
  #Same analyses but for only the first test trial
  first_novel = Exp2.boot[Exp2.boot$FirstTest == "Novel",]
  first_fam = Exp2.boot[Exp2.boot$FirstTest == "Familiar",]
  
  t_first_fam = t.test(first_fam$Familiar1, first_fam$HabEnd, paired=TRUE)
  t_first_nov = t.test(first_novel$Novel1, first_novel$HabEnd, paired=TRUE)
  bootMat_first_nov[ii] = t_first_nov$statistic/sqrt(nrow(Exp2.boot))
  bootMat_first_fam[ii] = t_first_fam$statistic/sqrt(nrow(Exp2.boot))
  

  t_first_comp = t.test(first_novel$Novel1, first_fam$Familiar1, paired=FALSE)
  bootMat_first_comp[ii] = t_first_comp$statistic/sqrt(nrow(Exp2.boot))
  
  
}

CIs_comp = list(conf = quantile(bootMat_comp, probs = c(alpha/2, 1-alpha/2)))
CIs_fam = list(conf = quantile(bootMat_fam, probs = c(alpha/2, 1-alpha/2)))
CIs_novel = list(conf = quantile(bootMat_novel, probs = c(alpha/2, 1-alpha/2)))
CIs_first_nov = list(conf = quantile(bootMat_first_nov, probs = c(alpha/2, 1-alpha/2)))
CIs_first_fam = list(conf = quantile(bootMat_first_fam, probs = c(alpha/2, 1-alpha/2)))
CIS_first_comp = list(conf = quantile(bootMat_first_comp, probs = c(alpha/2, 1-alpha/2)))
