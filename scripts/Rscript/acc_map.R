library(ggplot2)
AccuracyLocFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/AccuracyLocFlow.txt")
AccuracyTimeFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/AccuracyTimeFlow.txt")
sAccuracyLocFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/SAccuracyLocFlow.txt")
sAccuracyTimeFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/SAccuracyTimeFlow.txt")


fig_acc = function(df0, df1, name, width, height){
  
  acc0 = sum(df0$Value) / length(df0$Value)
  acc_r0 = paste('italic(Acc)', ' == ', acc0)
  df0 = rbind(df0, c(-100, -100, 0))
  df0 = rbind(df0, c(-100, -100, 1))
  df0$Value = factor(df0$Value, levels=c(0,1))
  
  fig_o = ggplot(df0, aes(Label, Predict)) + 
    theme(axis.text.x = element_blank(), axis.ticks = element_blank()) +
    geom_tile(aes(fill=Value), width=2, height=2) +
    geom_abline(slope=1, intercept = 0, colour='white', size=1) +
    scale_fill_manual(values=c('#c9e5ff', '#143a5e')) +
    scale_x_continuous(limits = c(0, 32)) +
    scale_y_continuous(limits = c(0, 32)) +
    labs(x = "TEST GROUP", y = 'FORECAST GROUP', fill='Accuracy', title = 'Accuracy Fitness Test') + 
    annotate('text', x=3, y=26, label = acc_r0, parse = TRUE) +
    annotate('text', x=26, y=3, label = acc_r0, parse = TRUE) +
    theme_bw() + 
    theme(axis.text.x = element_blank(), axis.text.y = element_blank(), axis.ticks = element_blank()) 
  
  print(fig_o)
  ggsave(paste("Accuracy", name, ".tiff"), fig_o, width = width, height = height)
    
  acc1 = sum(df1$Value) / length(df1$Value)
  acc_r1 = paste('italic(Acc)', ' == ', acc1)
  df1 = rbind(df1, c(-100, -100, 0))
  df1 = rbind(df1, c(-100, -100, 1))
  df1$Value = factor(df1$Value, levels=c(0,1))
  
  fig_s = ggplot(df1, aes(Label, Predict)) + 
    theme(axis.text.x = element_blank(), axis.ticks = element_blank()) +
    geom_tile(aes(fill=Value), width=2, height=2) +
    geom_abline(slope=1, intercept = 0, colour='white', size=1) +
    scale_fill_manual(values=c('#c9e5ff', '#143a5e')) +
    scale_x_continuous(limits = c(0, 32)) +
    scale_y_continuous(limits = c(0, 32)) +
    labs(x = "TEST GROUP", y = 'FORECAST GROUP', fill='Accuracy', title = 'Accuracy Fitness Test') + 
    annotate('text', x=27, y=2, label = acc_r1, parse = TRUE) +
    annotate('text', x=2, y=27, label = acc_r1, parse = TRUE) +
    theme_bw() + 
    theme(axis.text.x = element_blank(), axis.text.y = element_blank(), axis.ticks = element_blank()) 

  print(fig_s)
  ggsave(paste("SAccuracy", name, ".tiff"), fig_s, width = width, height = height)
}


otdf = AccuracyTimeFlow
oldf = AccuracyLocFlow

stdf = sAccuracyTimeFlow
sldf = sAccuracyLocFlow

fig_acc(otdf, stdf, 'TimeFlow', 5, 4)
fig_acc(oldf, sldf, 'LocFlow', 5, 4)
