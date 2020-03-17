AccuracyBinomialFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/PGCAltas/dataset/EMTAB6967/texts/AccuracyBinomialFlow.txt")
SAccuracyBinomialFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/PGCAltas/dataset/EMTAB6967/texts/SAccuracyBinomialFlow.txt")
AccuracyLocFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/PGCAltas/dataset/GSE120963/texts/AccuracyLocFlow.txt")
SAccuracyLocFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/PGCAltas/dataset/GSE120963/texts/SAccuracyLocFlow.txt")
AccuracyTimeFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/PGCAltas/dataset/GSE120963/texts/AccuracyTimeFlow.txt")
SAccuracyTimeFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/PGCAltas/dataset/GSE120963/texts/SAccuracyTimeFlow.txt")

setwd('D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/0217-large/texts')
AccuracyBinomialFlow = read.table('AccuracyBinomialFlow.txt', header = T)
SAccuracyBinomialFlow = read.table('SAccuracyBinomialFlow.txt', header = T)

fig_acc = function(ori, flt, name, width, height, loc_x=F){
  library(ggplot2)
  
  n = length(ori$Value)
  a = rep('Orignal', n)
  b = rep('Flited', n)
  ori$Predict = a
  flt$Predict = b
  
  if (loc_x == F) {
    loc_x = n * 0.1
  }
  
  acc1 = sum(ori$Value) / length(ori$Value)
  acc_r1 = paste('italic(Acc)', ' == ', round(acc1, 4))
  
  acc2 = sum(flt$Value) / length(flt$Value)
  acc_r2 = paste('italic(Acc)', ' == ', round(acc2, 4))
  
  df = rbind(ori, flt)
  
  df = rbind(df, c(-1, 'Orignal', 0))
  df = rbind(df, c(-1, 'Orignal', 1))
  
  df$Value = factor(df$Value, levels=c(0,1))
  df$Label = as.numeric(df$Label)
  
  fig = ggplot(df, aes(Label, Predict)) + 
    geom_tile(aes(fill=Value, height=0.2)) + 
    scale_fill_manual(values=c('#8FBC94', '#548687'), name='Predict Accuracy', labels=c('False', 'True')) +
    scale_x_continuous(limits = c(0, n)) +
    labs(x = "TEST CASE", y="TEST GROUP", title = 'Accuracy Fitness Test') +
    annotate('text', x=loc_x, y=2.2, label = acc_r1, parse = TRUE) +
    annotate('text', x=loc_x, y=1.2, label = acc_r2, parse = TRUE) +
    theme_bw()
  print(fig)
  ggsave(paste("Accuracy", name, ".tiff"), fig, width = width, height = height)  
}

obdf = AccuracyBinomialFlow
sbdf = SAccuracyBinomialFlow

otdf = AccuracyTimeFlow
stdf = SAccuracyTimeFlow

oldf = AccuracyLocFlow
sldf = SAccuracyLocFlow

fig_acc(obdf, sbdf, 'Binomal', 7, 3)

fig_acc(otdf, stdf, "TimeFlow", 7,3, loc_x = 10)
fig_acc(oldf, sldf, "LocFlow", 7,3, loc_x = 10)
