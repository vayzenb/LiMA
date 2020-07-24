rm(list = ls())

library(ggplot2)
library(reshape2)

setwd('C:/Users/vayze/Desktop/GitHub_Repos/LiMA')

load('Infant_Data/LiMA_data_all.RData')
load('Infant_Data/LiMA_Model_Data.RData')


ModelType= c( 'FF_IN', 'R_IN', 'FF_SN', 'R_SN')
ActualName= c( 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SN', 'ResNet-SN')

for (mm in 1:length(ModelType)){
  Exp1.Models[,1][Exp1.Models[,1] == ModelType[mm]] = ActualName[mm]
  Exp2.Models[,1][Exp2.Models[,1] == ModelType[mm]] = ActualName[mm]
  
}

sLine = .7
sAx = 8
sTitle = 10
sPlot = 2.5




#INFANT PLOTS

ggplot(Exp1.summary, aes(x = Condition, y = Fixation)) + geom_col(color = "black", fill = "#32759b", width = .5, size = sLine) + 
  geom_linerange(ymin = Exp1.summary$Fixation - Exp1.summary$SE, ymax =Exp1.summary$Fixation + Exp1.summary$SE, size = sLine) +
  xlab("Trial Type") + ylab("Mean Looking Time (s)") + scale_y_continuous(breaks = seq(0, 12, by = 2), limits=c(0,12), expand = c(0,0)) +
  scale_x_discrete(breaks=c("First 4","Last 4","Familiar", "Novel"), labels=c("First 4","Last 4","Same", "Different")) +
  theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                          axis.text.x = element_text(size=sAx, color = "black"),  
                          axis.title.x = element_blank(), 
                          axis.title.y = element_text(size=sTitle),
                          axis.line = element_line(size = sLine),
                          axis.ticks= element_line(size = sLine, color = 'black'),
                          axis.ticks.length = unit(.09, "cm"))


ggsave(filename =  'Infant_Data/Figures/LiMA - Exp1.png', plot = last_plot(), dpi = 300,width =2.2, height = 2.75)


Exp1.ratio = data.frame(Subj = AllSubs.Exp1$Subj, Ratio = AllSubs.Exp1$Ratio-.5)

ggplot(Exp1.ratio, aes(x = reorder(Subj, Ratio), y = Ratio)) +  geom_col(color = "black", fill = "Gray85",width = 1, size =sLine)+
  theme_classic() + coord_flip() +
  theme(axis.line.y = element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
        axis.text.x = element_text(size=sAx, color = "black"), 
        axis.title.x = element_blank(),
        axis.line = element_line(size = sLine)) +
  ylab("Difference from chance") + xlab(" ")

ggsave(filename =  'P:/Manuscripts/LiMA/Figures/LiMA - Exp1_Hist.png', plot = last_plot(), dpi = 300,width =2.2, height = 2.75)

ggplot(Exp2.summary, aes(x = Condition, y = Fixation)) + geom_col(color = "black", fill = "#32759b", width = .5, size = sLine) + 
  geom_linerange(ymin = Exp2.summary$Fixation - Exp2.summary$SE, ymax =Exp2.summary$Fixation + Exp2.summary$SE, size = sLine) +
  xlab("Trial Type") + ylab("Mean Looking Time (s)") + scale_y_continuous(breaks = seq(0, 12, by = 2), limits=c(0,12), expand = c(0,0)) +
  scale_x_discrete(breaks=c("First 4","Last 4","Familiar", "Novel"), labels=c("First 4","Last 4","Same", "Different")) +
  theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                          axis.text.x = element_text(size=sAx, color = "black"),  
                          axis.title.x = element_blank(), 
                          axis.title.y = element_text(size=sTitle),
                          axis.line = element_line(size = sLine),
                          axis.ticks= element_line(size = sLine, color = 'black'),
                          axis.ticks.length = unit(.09, "cm"))


ggsave(filename =  'Infant_Data/Figures/LiMA - Exp2.png', plot = last_plot(), dpi = 300,width =2.2, height = 2.75)


Exp2.ratio = data.frame(Subj = AllSubs.Exp2$Subj, Ratio = AllSubs.Exp2$Ratio-.5)

ggplot(Exp2.ratio, aes(x = reorder(Subj, Ratio), y = Ratio)) +  geom_col(color = "black", fill = "Gray85",width = 1, size =sLine)+
  theme_classic() + coord_flip() +
  theme(axis.line.y = element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
        axis.text.x = element_text(size=sAx, color = "black"), 
        axis.title.x = element_blank(),
        axis.line = element_line(size = sLine)) +
  ylab("Difference from chance") + xlab(" ")

ggsave(filename =  'P:/Manuscripts/LiMA/Figures/LiMA - Exp2_Hist.png', plot = last_plot(), dpi = 300,width =2.2, height = 2.75)
