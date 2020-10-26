rm(list = ls())

library(ggplot2)
library(reshape2)

setwd('C:/Users/vayze/Desktop/GitHub_Repos/LiMA')

load('Infant_Data/LiMA_Model_Data_Size.RData')
load('Infant_Data/LiMA_Model_Data.RData')
load('Infant_Data/LiMA_Model_Data_MultiItem.RData')


exp = c('Exp1', 'Exp2')
classifier = c("OCS", "ISOF")

ModelType= c('Infant', 'FF_IN', 'R_IN', 'FF_SN', 'R_SN')
ActualName= c('Infants', 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SIN', 'ResNet-SIN')

for (mm in 1:length(ModelType)){
  Exp1.Models[,1][Exp1.Models[,1] == ModelType[mm]] = ActualName[mm]
  Exp2.Models[,1][Exp2.Models[,1] == ModelType[mm]] = ActualName[mm]
  
  Exp1.Models_MSL[,1][Exp1.Models_MSL[,1] == ModelType[mm]] = ActualName[mm]
  Exp2.Models_MSL[,1][Exp2.Models_MSL[,1] == ModelType[mm]] = ActualName[mm]

  
}

sLine = .7
sAx = 8
sTitle = 10
sPlot = 2.5


ModelCols = c('#39a055', '#8ccf8a', '#c81b1d', '#ec3f2f', '#fa7051', '#fc9e80')

for (ee in exp){
  for (cl in classifier){
    #Preprocess data to have factors and conditions etc.
    df = as.data.frame(eval(as.name(paste(ee, '.Models', sep=""))))
    df$Model = factor(df$Model, levels = c( 'Infants', 'GBJ', 'GIST', 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SIN', 'ResNet-SIN'))
    df$Condition = factor(df$Condition, levels = c('SF','View', 'Size'))
    df$Acc = as.numeric(as.character(df$Acc))
    df$CI_Low = as.numeric(as.character(df$CI_Low))
    df$CI_High = as.numeric(as.character(df$CI_High))
    
    
    #Make figure for different SF
    df.SF = df[df$Condition == 'SF' & df$Classifier == cl | df$Classifier == "Infant",]
   
    
    ggplot(df.SF, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=c('#32759b', ModelCols)) +
      geom_linerange(aes(ymin =df.SF$CI_Low, ymax=df.SF$CI_High, x = Model), size = sLine) +
      scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
      xlab("Models") + ylab("Categorization Score") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                                 axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                                 axis.title.x = element_blank(), 
                                                                                 axis.title.y = element_text(size=sTitle),
                                                                                 axis.line = element_line(size = sLine),
                                                                                 axis.ticks= element_line(size = sLine, color = 'black'),
                                                                                 axis.ticks.length = unit(.09, "cm"),
                                                                                 legend.position ="none")
    
    
    ggsave(filename =  paste('Infant_Data/Figures/', ee, '_', cl,'_SF.png', sep = ""), plot = last_plot(), dpi = 300,width =3, height = 3)
    
    
    #reliability 'view' Figure
    df.view = df[df$Condition == 'View' & df$Classifier == cl,]
    
    ggplot(df.view, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=ModelCols) +
      geom_linerange(aes(ymin =df.view$CI_Low, ymax=df.view$CI_High, x = Model), size = sLine) +
      scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
      xlab("Models") + ylab("Categorization Score") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                                 axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                                 axis.title.x = element_blank(), 
                                                                                 axis.title.y = element_text(size=sTitle),
                                                                                 axis.line = element_line(size = sLine),
                                                                                 axis.ticks= element_line(size = sLine, color = 'black'),
                                                                                 axis.ticks.length = unit(.09, "cm"),
                                                                                 legend.position ="none")
    
    
    ggsave(filename =  paste('Infant_Data/Figures/', ee, '_', cl,'_view.png', sep = ""), plot = last_plot(), dpi = 300,width =3, height = 3)
    
    
    #SF one-class learning figure
    
    #Multi-class learning figure 
    df = as.data.frame(eval(as.name(paste(ee, '.Models_MSL', sep=""))))
    df = df[df$Classifier == cl,]
    df$Model = factor(df$Model, levels = c( 'GBJ', 'GIST', 'AlexNet-IN', 'ResNet-IN', 'AlexNet-SIN', 'ResNet-SIN'))
    df$Acc = as.numeric(as.character(df$Acc))
    df$CI_Low = as.numeric(as.character(df$CI_Low))
    df$CI_High = as.numeric(as.character(df$CI_High))
    
    
    ggplot(df, aes(x = Model, y= Acc, fill = Model)) + geom_col(color = "black", width = .5, size = sLine) + scale_fill_manual(values=ModelCols) +
      geom_linerange(aes(ymin =df$CI_Low, ymax=df$CI_High, x = Model), size = sLine) +
      scale_y_continuous(breaks = seq(0, 1, by = .25), limits=c(0,1), expand = c(0,0)) + geom_hline(yintercept= .5, linetype="dashed", size = sLine) +
      xlab("Models") + ylab("Categorization Score") + theme_classic() + theme(axis.text.y = element_text(size=sAx, color = "black"), 
                                                                                 axis.text.x = element_text(size=sAx, color = "black",angle =45,hjust = 1),  
                                                                                 axis.title.x = element_blank(), 
                                                                                 axis.title.y = element_text(size=sTitle),
                                                                                 axis.line = element_line(size = sLine),
                                                                                 axis.ticks= element_line(size = sLine, color = 'black'),
                                                                                 axis.ticks.length = unit(.09, "cm"),
                                                                                 legend.position ="none")
    
    
    ggsave(filename =  paste('Infant_Data/Figures/', ee, '_', cl,'_MSL.png', sep = ""), plot = last_plot(), dpi = 300,width =3, height = 3)
    
    

  }
  
}
