rm(list = ls())

library(ggplot2)
library(reshape2)

#setwd('B:/home/vayzenbe/GitHub_Repos/LiMA')
setwd('C:/Users/vayze/Desktop/GitHub_Repos/LiMA')

#load('Infant_Data/LiMA_Model_Data_Size.RData')

load('Infant_Data/LiMA_data_all.RData')
load('Infant_Data/LiMA_Model_Data.RData')
load('Infant_Data/LiMA_Model_Data_MultiItem.RData')
load('Infant_Data/LiMA_AE_Data.RData')

ModelLevels = c('Skeleton',  'ResNet-IN','ResNet-SIN', 'CorNet-S', 'ResNext-SAY','Pixel', 'FlowNet')
exp = c('Exp1', 'Exp2')



sLine = .7
sAx = 8
sTitle = 10
sPlot = 2.5


#ModelCols = c('#39a055', '#8ccf8a', '#c81b1d', '#d84e3d', '#d84e3d', '#f09581', '#f9b6a6', '#ffd7cd')

ModelCols = c('#32759b','#39a055','#FFD700','#db5900', '#a65628', '#9D02D7', '#de425b')
ModelCols = c('#32759b','#46995d','#f0c630','#db5900', '#a65628', '#9D02D7', '#de425b')
for (ee in exp){
 
  
  #One-shot learning across SF change (i.e., by skeleton)
  df.SF = as.data.frame(read.table(paste("Infant_Data/",ee,"_skel_cat.csv", sep=""),header = TRUE, sep=","))
  df.SF$model = factor(df.SF$model, levels = c('Infants', ModelLevels))
  
  ggplot(df.SF, aes(x = model, y= score, fill = model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=c('#BABABA', ModelCols)) +
    geom_linerange(aes(ymin =CI_low, ymax=CI_high, x = model), size = sLine) +
    scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
    xlab("Models") + ylab("Categorization Score") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                            axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                            axis.title.x = element_blank(), 
                                                                            axis.title.y = element_text(size=sTitle),
                                                                            axis.line = element_line(size = sLine),
                                                                            axis.ticks= element_line(size = sLine, color = 'black'),
                                                                            axis.ticks.length = unit(.09, "cm"),
                                                                            legend.position ="none")
  
  ggsave(filename =  paste('Infant_Data/Figures/', ee, '_SF_AE.png', sep = ""), plot = last_plot(), dpi = 300,width =3, height = 3)
  
  #one-shot learning across skel changes (i.e., by SF)
  df.skel = as.data.frame(read.table(paste("Infant_Data/",ee,"_sf_cat.csv", sep=""),header = TRUE, sep=","))
  df.skel$model = factor(df.skel$model, levels = ModelLevels)
  
  ggplot(df.skel, aes(x = model, y= score, fill = model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=ModelCols) +
    geom_linerange(aes(ymin =CI_low, ymax=CI_high, x = model), size = sLine) +
    scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
    xlab("Models") + ylab("Categorization Score") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                            axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                            axis.title.x = element_blank(), 
                                                                            axis.title.y = element_text(size=sTitle),
                                                                            axis.line = element_line(size = sLine),
                                                                            axis.ticks= element_line(size = sLine, color = 'black'),
                                                                            axis.ticks.length = unit(.09, "cm"),
                                                                            legend.position ="none")
  
  ggsave(filename =  paste('Infant_Data/Figures/', ee, '_Skel_AE.png', sep = ""), plot = last_plot(), dpi = 300,width =2.5, height = 3)
  
  # for (cl in classifier){
  #   #Preprocess data to have factors and conditions etc.
  #   df = as.data.frame(eval(as.name(paste(ee, '.Models', sep=""))))
  #   df$Model = factor(df$Model, levels = ModelLevels)
  #   df$Condition = factor(df$Condition, levels = c('SF','View', 'Skel'))
  #   df$Acc = as.numeric(as.character(df$Acc))
  #   df$CI_Low = as.numeric(as.character(df$CI_Low))
  #   df$CI_High = as.numeric(as.character(df$CI_High))
  #   
  #   
  #   #Make figure for different SF
  #   df.SF = df[df$Condition == 'SF' & df$Classifier == cl | df$Classifier == "Infant",]
  #  
  #   
  #   ggplot(df.SF, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=c('#32759b', ModelCols)) +
  #     geom_linerange(aes(ymin =df.SF$CI_Low, ymax=df.SF$CI_High, x = Model), size = sLine) +
  #     scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  #     xlab("Models") + ylab("Categorization Score") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
  #                                                                                axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
  #                                                                                axis.title.x = element_blank(), 
  #                                                                                axis.title.y = element_text(size=sTitle),
  #                                                                                axis.line = element_line(size = sLine),
  #                                                                                axis.ticks= element_line(size = sLine, color = 'black'),
  #                                                                                axis.ticks.length = unit(.09, "cm"),
  #                                                                                legend.position ="none")
  #   
  #   
  #   ggsave(filename =  paste('Infant_Data/Figures/', ee, '_', cl,'_SF.png', sep = ""), plot = last_plot(), dpi = 300,width =3, height = 3)
  #   
  #   
  #   #reliability 'view' Figure
  #   df.view = df[df$Condition == 'View' & df$Classifier == cl,]
  #   
  #   ggplot(df.view, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=ModelCols) +
  #     geom_linerange(aes(ymin =df.view$CI_Low, ymax=df.view$CI_High, x = Model), size = sLine) +
  #     scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  #     xlab("Models") + ylab("Categorization Score") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
  #                                                                                axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
  #                                                                                axis.title.x = element_blank(), 
  #                                                                                axis.title.y = element_text(size=sTitle),
  #                                                                                axis.line = element_line(size = sLine),
  #                                                                                axis.ticks= element_line(size = sLine, color = 'black'),
  #                                                                                axis.ticks.length = unit(.09, "cm"),
  #                                                                                legend.position ="none")
  #   
  #   
  #   ggsave(filename =  paste('Infant_Data/Figures/', ee, '_', cl,'_view.png', sep = ""), plot = last_plot(), dpi = 300,width =3, height = 3)
  #   
  #   
  #   #One shot learning across changes in skeleton (i.e., by SF)
  #   df.skel = df[df$Condition == 'Skel' & df$Classifier == cl,]
  #   
  #   ggplot(df.skel, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=ModelCols) +
  #     geom_linerange(aes(ymin =df.skel$CI_Low, ymax=df.skel$CI_High, x = Model), size = sLine) +
  #     scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  #     xlab("Models") + ylab("Categorization Score") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
  #                                                                             axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
  #                                                                             axis.title.x = element_blank(), 
  #                                                                             axis.title.y = element_text(size=sTitle),
  #                                                                             axis.line = element_line(size = sLine),
  #                                                                             axis.ticks= element_line(size = sLine, color = 'black'),
  #                                                                             axis.ticks.length = unit(.09, "cm"),
  #                                                                             legend.position ="none")
  #   
  #   
  #   ggsave(filename =  paste('Infant_Data/Figures/', ee, '_', cl,'_Skel.png', sep = ""), plot = last_plot(), dpi = 300,width =3, height = 3)
  #   
  #   
  #   #SF one-class learning figure
  #   
  #   #Multi-class learning figure 
  #   # df = as.data.frame(eval(as.name(paste(ee, '.Models_MSL', sep=""))))
  #   # df = df[df$Classifier == cl,]
  #   # df$Model = factor(df$Model, levels = c( 'GBJ', 'GIST', 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SIN', 'ResNet-SIN', 'CorNet-Z', 'CorNet-S'))
  #   # df$Acc = as.numeric(as.character(df$Acc))
  #   # df$CI_Low = as.numeric(as.character(df$CI_Low))
  #   # df$CI_High = as.numeric(as.character(df$CI_High))
  #   # 
  #   # 
  #   # ggplot(df, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=ModelCols) +
  #   #   geom_linerange(aes(ymin =df$CI_Low, ymax=df$CI_High, x = Model), size = sLine) +
  #   #   scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
  #   #   xlab("Models") + ylab("Categorization Score") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
  #   #                                                                              axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
  #   #                                                                              axis.title.x = element_blank(), 
  #   #                                                                              axis.title.y = element_text(size=sTitle),
  #   #                                                                              axis.line = element_line(size = sLine),
  #   #                                                                              axis.ticks= element_line(size = sLine, color = 'black'),
  #   #                                                                              axis.ticks.length = unit(.09, "cm"),
  #   #                                                                              legend.position ="none")
  #   # 
  #   # 
  #   # ggsave(filename =  paste('Infant_Data/Figures/', ee, '_', cl,'_MSL.png', sep = ""), plot = last_plot(), dpi = 300,width =3, height = 3)
  #   # 
  #   # 
  # 
  # }
  
}