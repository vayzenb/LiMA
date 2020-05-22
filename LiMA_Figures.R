rm(list = ls())

library(ggplot2)
library(reshape2)

setwd('R:/LourencoLab/Adult Studies/Shapes (All Experiments)/LiMA/Analysis Files')

load('LiMA_data_all.RData')
load('LiMA_OneClass_Models.RData')


sLine = .7
sAx = 8
sTitle = 10
sPlot = 2.5

allCondNums = c(8, 2, 14, 9, 3, 15, 10, 4, 16, 11, 5, 17, 12, 6, 18, 13, 7, 19, 1)
colNums = c(1,7, 2, 8, 3, 9,4,10,5,11,6,12) 


ModelCols = c('#39a055', '#8ccf8a', '#c81b1d', '#ec3f2f', '#fa7051', '#fc9e80')

#Exp 1 MODEL PLOTS
Exp1.Models$Model = factor(Exp1.Models$Model, levels = c( 'Infant', 'GBJ', 'GIST', 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SN', 'ResNet-SN'))
Exp1.Models$Condition = factor(Exp1.Models$Condition, levels = c('Diff_SF','Same_SF', 'Diff_Size'))
Exp1.Models$Acc = as.numeric(as.character(Exp1.Models$Acc))
Exp1.Models$SE = as.numeric(as.character(Exp1.Models$SE))
Exp1.Models$roworder = allCondNums
attach(Exp1.Models)
Exp1.Models = Exp1.Models[order(roworder),]

Exp2.Models = as.data.frame(Exp2.Models)
Exp2.Models$Model = factor(Exp2.Models$Model, levels = c( 'Infant', 'GBJ', 'GIST', 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SN', 'ResNet-SN'))
Exp2.Models$Condition = factor(Exp2.Models$Condition, levels = c( 'Diff_SF', 'Same_SF', 'Diff_Size'))
Exp2.Models$Acc = as.numeric(as.character(Exp2.Models$Acc))
Exp2.Models$SE = as.numeric(as.character(Exp2.Models$SE))
Exp2.Models$roworder = allCondNums
attach(Exp2.Models)
Exp2.Models = Exp2.Models[order(roworder),]


Exp1.SF = Exp1.Models[Exp1.Models$Condition == 'Diff_SF',]
Exp1.Control = Exp1.Models[Exp1.Models$Condition != 'Diff_SF',]
Exp1.Control$roworder = colNums
attach(Exp1.Control)
Exp1.Control = Exp1.Control[order(roworder),]

Exp2.SF = Exp2.Models[Exp2.Models$Condition == 'Diff_SF',]
Exp2.Control = Exp2.Models[Exp2.Models$Condition != 'Diff_SF',]
Exp2.Control$roworder = colNums
attach(Exp2.Control)
Exp2.Control = Exp2.Control[order(roworder),]




#For Dissertation
DissCol = c("gray","gray", "#2F5597",  "#2F5597", "#B90000","#B90000","#B90000","#B90000")
Exp.Diss = rbind(Exp1.SF, Exp2.Models[Exp2.Models$Model == 'Infant',])
Exp.Diss$Model = as.character(Exp.Diss$Model)
Exp.Diss$Acc = as.numeric(Exp.Diss$Acc)
Exp.Diss$SE = as.numeric(Exp.Diss$SE)
Exp.Diss[1,1] = 'Experiment 1'
Exp.Diss[8,1] = 'Experiment 2'
Exp.Diss$Model = factor(Exp.Diss$Model, levels = c('Experiment 1','Experiment 2', 'GBJ', 'GIST', 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SN', 'ResNet-SN'))
                   
ggplot(Exp.Diss, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) +scale_fill_manual(values=DissCol) +
  geom_linerange(aes(ymin = Exp.Diss$Acc - Exp.Diss$SE, ymax=Exp.Diss$Acc + Exp.Diss$SE, x = Model), size = sLine) +
  scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  xlab("Models") + ylab("Categorization Accuracy") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                               axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                               axis.title.x = element_blank(), 
                                                                               axis.title.y = element_text(size=sTitle),
                                                                               axis.line = element_line(size = sLine),
                                                                               axis.ticks= element_line(size = sLine, color = 'black'),
                                                                               axis.ticks.length = unit(.09, "cm"),
                                                                              legend.position ="none")

 
  
ggsave(filename =  'Figures/Diss_models.png', plot = last_plot(), dpi = 300,width =4.4, height = 3)

#Exp 1 and 2 combine conditions
ggplot(Exp1.Models, aes(x = Model, y= Acc, fill = Model)) + geom_bar(stat="identity", color = "black", width = .5, size = sLine, position=position_dodge()) + 
  facet_grid(. ~ Condition) + 
  geom_linerange(stat="identity", aes(x = Model, ymin = Exp1.Models$Acc - Exp1.Models$SE, ymax=Exp1.Models$Acc + Exp1.Models$SE), size = sLine) +
  scale_fill_manual(values=c('#32759b', ModelCols)) +
  scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
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

ggsave(filename =  'Figures/Exp1_models_allConds.png', plot = last_plot(), dpi = 300,width =6.6, height = 3)

#Exp2
ggplot(Exp2.Models, aes(x = Model, y= Acc, fill = Model)) + geom_bar(stat="identity", color = "black", width = .5, size = sLine, position=position_dodge()) + 
  facet_grid(. ~ Condition) + 
  geom_linerange(stat="identity", aes(x = Model, ymin = Exp2.Models$Acc - Exp2.Models$SE, ymax=Exp2.Models$Acc + Exp2.Models$SE), size = sLine) +
  scale_fill_manual(values=c('#32759b', ModelCols)) +
  scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
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

ggsave(filename =  'Figures/Exp2_models_allConds.png', plot = last_plot(), dpi = 300,width =6.6, height = 3)


#For Exp1 SF condition

ggplot(Exp1.SF, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) +scale_fill_manual(values=c('#32759b', ModelCols)) +
  geom_linerange(aes(ymin = Exp1.SF$Acc - Exp1.SF$SE, ymax=Exp1.SF$Acc + Exp1.SF$SE, x = Model), size = sLine) +
  scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  xlab("Models") + ylab("Categorization Accuracy") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                             axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                             axis.title.x = element_blank(), 
                                                                             axis.title.y = element_text(size=sTitle),
                                                                             axis.line = element_line(size = sLine),
                                                                             axis.ticks= element_line(size = sLine, color = 'black'),
                                                                             axis.ticks.length = unit(.09, "cm"),
                                                                             legend.position ="none")

ggsave(filename =  'Figures/Exp1_models_SF.png', plot = last_plot(), dpi = 300,width =3.85, height = 3)

#Exp1 Control conditions
ggplot(Exp1.Control, aes(x = Model, y= Acc, fill = Model)) + geom_bar(stat="identity", color = "black", width = .5, size = sLine, position=position_dodge()) + 
  facet_grid(. ~ Condition) + 
  geom_linerange(stat="identity", aes(x = Model, ymin = Exp1.Control$Acc - Exp1.Control$SE, ymax=Exp1.Control$Acc + Exp1.Control$SE), size = sLine) +
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

ggsave(filename =  'Figures/Exp1_models_Controls.png', plot = last_plot(), dpi = 300,width =3.85, height = 3)


#For Exp2 SF condition

ggplot(Exp2.SF, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) +scale_fill_manual(values=c('#32759b', ModelCols)) +
  geom_linerange(aes(ymin = Exp2.SF$Acc - Exp2.SF$SE, ymax=Exp2.SF$Acc + Exp2.SF$SE, x = Model), size = sLine) +
  scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  xlab("Models") + ylab("Categorization Accuracy") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                             axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                             axis.title.x = element_blank(), 
                                                                             axis.title.y = element_text(size=sTitle),
                                                                             axis.line = element_line(size = sLine),
                                                                             axis.ticks= element_line(size = sLine, color = 'black'),
                                                                             axis.ticks.length = unit(.09, "cm"),
                                                                             legend.position ="none")

ggsave(filename =  'Figures/Exp2_models_SF.png', plot = last_plot(), dpi = 300,width =3.85, height = 3)

#Exp2 Control conditions
ggplot(Exp2.Control, aes(x = Model, y= Acc, fill = Model)) + geom_bar(stat="identity", color = "black", width = .5, size = sLine, position=position_dodge()) + 
  facet_grid(. ~ Condition) + 
  geom_linerange(stat="identity", aes(x = Model, ymin = Exp2.Control$Acc - Exp2.Control$SE, ymax=Exp2.Control$Acc + Exp2.Control$SE), size = sLine) +
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

ggsave(filename =  'Figures/Exp2_models_Controls.png', plot = last_plot(), dpi = 300,width =3.85, height = 3)





#INFANT PLOTS

ggplot(Exp1.summary, aes(x = Condition, y = Fixation)) + geom_col(color = "black", fill = "Gray85", width = .5, size = sLine) + 
  geom_linerange(ymin = Exp1.summary$Fixation - Exp1.summary$SE, ymax =Exp1.summary$Fixation + Exp1.summary$SE, size = sLine) +
  xlab("Trial Type") + ylab("Mean Looking Time (s)") + scale_y_continuous(breaks = seq(0, 12, by = 2), limits=c(0,12), expand = c(0,0)) +
  theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                          axis.text.x = element_text(size=sAx, color = "black"),  
                          axis.title.x = element_blank(), 
                          axis.title.y = element_text(size=sTitle),
                          axis.line = element_line(size = sLine),
                          axis.ticks= element_line(size = sLine, color = 'black'),
                          axis.ticks.length = unit(.09, "cm"))


ggsave(filename =  'Figures/LiMA - Exp1_Diss.png', plot = last_plot(), dpi = 300,width =2.2, height = 2.75)


Exp1.ratio = data.frame(Subj = AllSubs.Exp1$Subj, Ratio = AllSubs.Exp1$Ratio-.5)

ggplot(Exp1.ratio, aes(x = reorder(Subj, Ratio), y = Ratio)) +  geom_col(color = "black", fill = "Gray85",width = 1, size =sLine)+
  theme_classic() + coord_flip() +
  theme(axis.line.y = element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
        axis.text.x = element_text(size=sAx, color = "black"), 
        axis.title.x = element_blank(),
        axis.line = element_line(size = sLine)) +
  ylab("Difference from chance") + xlab(" ")

ggsave(filename =  'P:/Manuscripts/LiMA/Figures/LiMA - Exp1_Hist.png', plot = last_plot(), dpi = 300,width =2.2, height = 2.75)

ggplot(Exp2.summary, aes(x = Condition, y = Fixation)) + geom_col(color = "black", fill = "Gray85", width = .5, size = sLine) + 
  geom_linerange(ymin = Exp2.summary$Fixation - Exp2.summary$SE, ymax =Exp2.summary$Fixation + Exp2.summary$SE, size = sLine) +
  xlab("Trial Type") + ylab("Mean Looking Time (s)") + scale_y_continuous(breaks = seq(0, 12, by = 2), limits=c(0,12), expand = c(0,0)) +
  theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                          axis.text.x = element_text(size=sAx, color = "black"),  
                          axis.title.x = element_blank(), 
                          axis.title.y = element_text(size=sTitle),
                          axis.line = element_line(size = sLine),
                          axis.ticks= element_line(size = sLine, color = 'black'),
                          axis.ticks.length = unit(.09, "cm"))


ggsave(filename =  'Figures/LiMA - Exp2_Diss.png', plot = last_plot(), dpi = 300,width =2.2, height = 2.75)


Exp2.ratio = data.frame(Subj = AllSubs.Exp2$Subj, Ratio = AllSubs.Exp2$Ratio-.5)

ggplot(Exp2.ratio, aes(x = reorder(Subj, Ratio), y = Ratio)) +  geom_col(color = "black", fill = "Gray85",width = 1, size =sLine)+
  theme_classic() + coord_flip() +
  theme(axis.line.y = element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
        axis.text.x = element_text(size=sAx, color = "black"), 
        axis.title.x = element_blank(),
        axis.line = element_line(size = sLine)) +
  ylab("Difference from chance") + xlab(" ")

ggsave(filename =  'P:/Manuscripts/LiMA/Figures/LiMA - Exp2_Hist.png', plot = last_plot(), dpi = 300,width =2.2, height = 2.75)
