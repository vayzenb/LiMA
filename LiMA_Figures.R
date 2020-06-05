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


ModelCols = c('#39a055', '#8ccf8a', '#c81b1d', '#ec3f2f', '#fa7051', '#fc9e80')

#Exp 1 MODEL PLOTS
Exp1.Models = as.data.frame(Exp1.Models)
Exp1.Models$Model = factor(Exp1.Models$Model, levels = c( 'Infant', 'GBJ', 'GIST', 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SN', 'ResNet-SN'))
Exp1.Models$Condition = factor(Exp1.Models$Condition, levels = c('SF','View', 'Size'))
Exp1.Models$Acc = as.numeric(as.character(Exp1.Models$Acc))
Exp1.Models$CI_Low = as.numeric(as.character(Exp1.Models$CI_Low))
Exp1.Models$CI_High = as.numeric(as.character(Exp1.Models$CI_High))
#Exp1.Models$roworder = allCondNums
#attach(Exp1.Models)
#Exp1.Models = Exp1.Models[order(roworder),]



#Exp1.Control$roworder = colNums
#attach(Exp1.Control)
#Exp1.Control = Exp1.Control[order(roworder),]

#For Exp1 SF condition; ISOF classifier
Exp1.SF = Exp1.Models[Exp1.Models$Condition == 'SF' & Exp1.Models$Classifier == "ISOF" | Exp1.Models$Classifier == "Infant",]

ggplot(Exp1.SF, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=c('#32759b', ModelCols)) +
  geom_linerange(aes(ymin =Exp1.SF$CI_Low, ymax=Exp1.SF$CI_High, x = Model), size = sLine) +
  scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  xlab("Models") + ylab("Categorization Accuracy") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                             axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                             axis.title.x = element_blank(), 
                                                                             axis.title.y = element_text(size=sTitle),
                                                                             axis.line = element_line(size = sLine),
                                                                             axis.ticks= element_line(size = sLine, color = 'black'),
                                                                             axis.ticks.length = unit(.09, "cm"),
                                                                             legend.position ="none")


ggsave(filename =  'Infant_Data/Figures/Exp1_models_SF.png', plot = last_plot(), dpi = 300,width =3, height = 3)

#Control graph
Exp1.Control = Exp1.Models[Exp1.Models$Condition != 'SF' & Exp1.Models$Classifier == "ISOF",]

#Exp1 Control conditions
ggplot(Exp1.Control, aes(x = Model, y= Acc, fill = Model)) + geom_bar(stat="identity", color = "black", width = .5, size = sLine, position=position_dodge()) + 
  facet_grid(. ~ Condition) + 
  geom_linerange(stat="identity", aes(x = Model, ymin = Exp1.Control$CI_Low, ymax=Exp1.Control$CI_High), size = sLine) +
  scale_fill_manual(values=ModelCols) +
  scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  xlab("Models") + ylab("Categorization Accuracy") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                             axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                             axis.title.x = element_blank(), 
                                                                             axis.title.y = element_text(size=sTitle),
                                                                             axis.line = element_line(size = sLine),
                                                                             axis.ticks= element_line(size = sLine, color = 'black'),
                                                                             axis.ticks.length = unit(.09, "cm"),
                                                                             legend.position ="none", 
                                                                             strip.background = element_blank(),
                                                                             strip.text.x = element_blank())



ggsave(filename =  'Infant_Data/Figures/Exp1_models_Controls.png', plot = last_plot(), dpi = 300,width =6, height = 3)


#Experiment 2 figures

Exp2.Models = as.data.frame(Exp2.Models)
Exp2.Models$Model = factor(Exp2.Models$Model, levels = c( 'Infant', 'GBJ', 'GIST', 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SN', 'ResNet-SN'))
Exp2.Models$Condition = factor(Exp2.Models$Condition, levels = c('SF','View', 'Size'))
Exp2.Models$Acc = as.numeric(as.character(Exp2.Models$Acc))
Exp2.Models$CI_Low = as.numeric(as.character(Exp2.Models$CI_Low))
Exp2.Models$CI_High = as.numeric(as.character(Exp2.Models$CI_High))

#For Exp1 SF condition; ISOF classifier
Exp2.SF = Exp2.Models[Exp2.Models$Condition == 'SF' & Exp2.Models$Classifier == "ISOF" | Exp2.Models$Classifier == "Infant",]

ggplot(Exp2.SF, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=c('#32759b', ModelCols)) +
  geom_linerange(aes(ymin =Exp2.SF$CI_Low, ymax=Exp2.SF$CI_High, x = Model), size = sLine) +
  scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  xlab("Models") + ylab("Categorization Accuracy") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                             axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                             axis.title.x = element_blank(), 
                                                                             axis.title.y = element_text(size=sTitle),
                                                                             axis.line = element_line(size = sLine),
                                                                             axis.ticks= element_line(size = sLine, color = 'black'),
                                                                             axis.ticks.length = unit(.09, "cm"),
                                                                             legend.position ="none")


ggsave(filename =  'Infant_Data/Figures/Exp2_models_SF.png', plot = last_plot(), dpi = 300,width =3, height = 3)

#Control graph
Exp2.Control = Exp2.Models[Exp2.Models$Condition != 'SF' & Exp2.Models$Classifier == "ISOF",]

#Exp2 Control conditions
ggplot(Exp2.Control, aes(x = Model, y= Acc, fill = Model)) + geom_bar(stat="identity", color = "black", width = .5, size = sLine, position=position_dodge()) + 
  facet_grid(. ~ Condition) + 
  geom_linerange(stat="identity", aes(x = Model, ymin = Exp2.Control$CI_Low, ymax=Exp2.Control$CI_High), size = sLine) +
  scale_fill_manual(values=ModelCols) +
  scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  xlab("Models") + ylab("Categorization Accuracy") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                             axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                             axis.title.x = element_blank(), 
                                                                             axis.title.y = element_text(size=sTitle),
                                                                             axis.line = element_line(size = sLine),
                                                                             axis.ticks= element_line(size = sLine, color = 'black'),
                                                                             axis.ticks.length = unit(.09, "cm"),
                                                                             legend.position ="none", 
                                                                             strip.background = element_blank(),
                                                                             strip.text.x = element_blank())



ggsave(filename =  'Infant_Data/Figures/Exp2_models_Controls.png', plot = last_plot(), dpi = 300,width =6, height = 3)








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
