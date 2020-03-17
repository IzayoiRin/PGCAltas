library(ggplot2)
library(RColorBrewer)
library(scales)

setwd('D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/0217-large/processed')
ssvmeval = read.table('SSVMeval2.txt', header=T)
head(ssvmeval)

aucs = ggplot(ssvmeval, aes(NBOOST, AUC, group=GROUP)) +
  geom_line(aes(color=GROUP), size=0.8) +
  geom_point(aes(color=GROUP), size=2, shape=21, fill='white') +
  geom_vline(aes(xintercept=15), linetype='dashed', color="#CA0020", size=0.8) +
  scale_color_manual(values = brewer.pal(8, "Set1")[c(2:5,7)]) + 
  scale_x_continuous(limits = c(1, 22), breaks = pretty_breaks(12)) +
  scale_y_continuous(limits = c(0.975, 1)) +
  labs(title = 'The AUC of Different Hyperparameters Models', x='N-Boost', y='AUC', color='Models') +
  theme_bw()

aucs

unqssvmeval = unique(ssvmeval[c('TPR', 'PPV', 'F1SCORE','ACC', 'GROUP')], fromLast = F)


ac_tp_fp = function(df){
  value = c(df$TPR / (df$PPV + df$TPR), df$PPV / (df$PPV + df$TPR))
  type = c(rep('TPR', dim(df)[1]), rep('PPV', dim(df)[1]))
  acc = c(df$ACC, df$ACC)
  f1 = c(df$F1SCORE, df$F1SCORE)
  group = c(as.vector(df$GROUP), as.vector(df$GROUP))
  df = data.frame(data=cbind(group, acc, f1, type, value))
  colnames(df) = c('GROUP', 'ACC', 'F1SCORE', 'TYPE', 'VALUE')
  df$VALUE= as.numeric(as.vector(df$VALUE))
  df$ACC= as.numeric(as.vector(df$ACC))
  df$F1SCORE= as.numeric(as.vector(df$F1SCORE))
  df$S1 = as.factor(rep('Accaracy', dim(df)[1]))
  df$S2 = as.factor(rep('F1Score', dim(df)[1]))
  return(df)
}

atps = ac_tp_fp(unqssvmeval)
atps
a = atps[c('GROUP', 'ACC', 'S1')]
b = atps[c('GROUP', 'F1SCORE', 'S2')]
colnames(a) = c('GROUP', 'v', 'S')
colnames(b) = c('GROUP', 'v', 'S')
acf1 = rbind(a, b)
acf1

p = ggplot(acf1) + 
  geom_bar(aes(GROUP, VALUE, fill=TYPE), data =atps, stat = 'identity') +
  geom_line(mapping = aes(GROUP, v, color=S, group=S), size=0.8)+
  geom_point(aes(GROUP, v, color=S), size=3, shape=21, fill='white') +
  scale_fill_manual(values = c("#8FBC94","#548687"), labels=c("Precision", "Recall"), name=NULL)+
  scale_color_manual(values = c('#e97f02', '#CD5C5C'), name=NULL) +
  labs(title = 'The Bias-Error of Different Hyperparameters Models', x='Models', y='Statistics Ratio') +
  theme_bw()+
  theme(axis.text.x = element_text(face = 'bold', color=brewer.pal(8, "Set1")[c(2:5,7)]))
  
p


rocData = read.table('SSVMROC.txt', header = T)
head(rocData)
stline = data.frame(cbind(c(0, 0.3, 0.5, 0.8, 1), c(0, 0.3, 0.5, 0.8, 1)))
roc = ggplot(rocData, aes(FPR, TPR)) +
  geom_line(aes(color=GROUP), size=0.8) +
  geom_line(mapping = aes(X1, X2), data = stline, linetype='dashed', color="#CA0020", size=0.8) + 
  scale_color_manual(values = brewer.pal(8, "Set1")[c(2:5,7)]) +
  labs(title = 'The ROC of Different Hyperparameters Models', x='False Positive rate', y='True Positive Rate', color='Models') +
  theme_bw()
roc

rocdata = read.table('SVMROC.txt', header = T)
head(rocdata)
rocdata$Boost = as.factor(rocdata$Boost)
ggplot(rocdata, aes(FPR, TPR)) +
  geom_line(aes(color=Boost),size=0.8) +
  scale_y_continuous(limits = c(0.88, 1)) +
  scale_x_continuous(limits = c(0, 0.03)) +
  scale_color_manual(values = brewer.pal(8, "Set1")[c(2:5,7)]) +
  labs(title = "The magnification of Different N Models' ROC", x='False Positive rate', y='True Positive Rate', color='N-Boost') +
  theme_bw()
aucdata = read.table('SVMAUC.txt', header = T)
head(aucdata)
ggplot(aucdata, aes(Boost, AUC)) +
  geom_line(aes(group=1), size=1.2, color="#7570B3") +
  geom_point(color="#7570B3", shape=21, size=3, fill='white') +
  geom_text(aes(label=round(AUC, 4)), vjust=-2,fontface='bold', size=2.5, color='#E7298A') +
  labs(title ='', x=NULL, y='AUC') +
  coord_flip() +
  theme_bw()+
  theme(axis.text.y = element_text(face = 'bold', size=12, color=brewer.pal(8, "Set1")[c(2:5,7)]))

######################################################################

cvData = read.table('SSVMCV.txt', header = T)
head(cvData)
cvData$GROUP = as.factor(cvData$GROUP)
cvData$FOLD = as.factor(cvData$FOLD)
  
cv = ggplot(cvData, aes(GROUP, CV, group=FOLD, color=FOLD)) + 
  geom_line(position=position_dodge(0.1), size=1) +
  geom_errorbar(aes(ymin=CV-SE, ymax=CV+SE), width=0.2, position=position_dodge(0.1)) +
  geom_point(aes(color=FOLD), position=position_dodge(0.1), size=2, shape=21, fill='white') + 
  scale_color_manual(values = brewer.pal(8, "Dark2")) + 
  labs(title = 'The CV of Different Hyperparameters Models', x='Models', y='CV', color='Cross Fold') +
  theme_bw() +
  theme(axis.text.x = element_text(face = 'bold', color=brewer.pal(8, "Set1")[c(2:5,7)]))

cv

#####################################################################
cvaData = read.table('SSVMCVAUC.txt', header = T)
head(cvaData)
cvaData = cvaData[which(cvaData$AUC>0.9),]
cvaData$GROUP = as.factor(cvaData$GROUP)
ggplot(cvaData, aes(Boost, GROUP, group=S)) +
  geom_tile(aes(fill=AUC)) +
  geom_text(aes(label=round(AUC, 4)), color = "white", size = 3) +
  scale_fill_gradient(low = "#9E0142", high="#FDAE61") +
  labs(title = 'The AUC of Cross Test Models', x='N-Boost', y='Cross Test Set') +
  scale_x_continuous(limits = c(0, 16), breaks = pretty_breaks(7)) +
  theme_bw() + 
  theme(legend.key.width=unit(3,'mm'),legend.key.height=unit(1.5,'cm'))+
  theme(legend.title = element_text(size = 8)) +
  coord_fixed(ratio = 1.5)

cvadata = read.table('SVMCVAUC.txt', header=T)
head(cvadata)
cvadata$V = as.factor(cvadata$V)
rscale = function(a, f){
  r = c(
    f(a[1:5]),
    f(a[6:10]),
    f(a[11:15]),
    f(a[16:20]),
    f(a[21:25]),
    f(a[26:30]),
    f(a[31:35]),
    f(a[36:40]),
    f(a[41:45]),
    f(a[46:50])
  )
  return(r)  
}



ggplot(cvadata, aes(Boost, V)) +
  geom_tile(aes(fill=rscale(cvadata$AUC, function(x){return(rescale(x, c(0,1)))})))+
  scale_fill_gradient(low = "#9E0142", high="#FFE4E1")+
  labs(title = 'The AUC of Cross Test Models', x='N-Boost', y='Cross Test Set', fill='Scaled AUC') +
  theme_bw() + 
  theme(legend.key.width=unit(3,'mm'),legend.key.height=unit(1.5,'cm'))+
  theme(legend.title = element_text(size = 8))+
  coord_fixed(ratio = 0.8)

d = data.frame(cbind(unique(cvadata$V), rscale(cvadata$AUC, max)))
colnames(d) = c('V', 'AUC')
d$V = as.factor(d$V)
ggplot(d, aes(V, AUC)) +
  geom_line(aes(group=1), size=1.2, color="#7570B3") +
  geom_point(color="#7570B3", shape=21, size=3, fill='white') +
  geom_text(aes(label=round(AUC, 4)), vjust=-2,fontface='bold', size=2.5, color='#E7298A') +
  labs(title ='', x=NULL, y='Max AUC') +
  coord_flip() +
  theme_bw()

#############################################################
cvnp = read.table('SVMCVNP.txt', header = T)
head(cvnp)
cvnp$V = as.factor(cvnp$V)

cvst = read.table('SVMCVST.txt', header = T)
head(cvst)
cvst$V = as.factor(cvst$V)
i = which(cvst$ST %in% c('PrecisionNeg', 'F1ScoreNeg', 'F1ScoreW', 'Specificity', 'Accuracy') == F)
cvst = cvst[i,]

ggplot(cvst) + 
  geom_bar(aes(V, P, fill=TYPE), data =cvnp, stat = 'identity')+
  scale_fill_manual(values = brewer.pal(8, "Set3"), name=NULL)+
  geom_line(aes(V, Value, color=ST, group=ST), size=0.8) +
  geom_point(aes(V, Value, color=ST), shape=21, size=3, fill='white') +
  scale_color_manual(values = brewer.pal(7, "Dark2")[c(2:5)], name=NULL) +
  labs(title = 'The Variance-Error of Cross Validation', x='Cross Test set', y='Statistics Ratio') +
  theme_bw()


cvste = read.table('SVMCVSTE.txt', header = T)
head(cvste)
cvout = read.table('SVMCVOUTTER.txt', header = T)
head(cvout)

ggplot(cvste, aes(STA, AVE)) +
  geom_bar(stat = 'identity', fill=brewer.pal(8, "Dark2")) +
  geom_errorbar(aes(ymin=AVE-STE, ymax=AVE+STE), width=0.05, color='black') +
  geom_point(aes(STA, VALUE, color=TYPE), data=cvout, shape=21, size=3, fill='white', position = position_dodge(0.1)) +
  coord_polar(theta = "x") +
  labs(title = '', x=NULL, y='Average', color=NULL) +
  theme_bw() +
  theme(axis.text.x = element_text(size=16, face = 'bold', color=brewer.pal(8, "Dark2"))) +
  theme(legend.text = element_text(size=12, face='bold'))
