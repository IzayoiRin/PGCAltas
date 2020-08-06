library(ggplot2)

STOCHASTIC10 = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/processed/BinomialEstimate/rawdata/10419/TSNE10.txt")
STOCHASTIC30 = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/processed/BinomialEstimate/rawdata/10419/TSNE30.txt")
STOCHASTIC60 = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/processed/BinomialEstimate/rawdata/10419/TSNE60.txt")
STOCHASTIC80 = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/processed/BinomialEstimate/rawdata/10419/TSNE80.txt")

SPARSE_SVD10 = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/processed/BinomialEstimate/rawdata/10419/SPSVD10.txt")
SPARSE_SVD30 = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/processed/BinomialEstimate/rawdata/10419/SPSVD30.txt")
SPARSE_SVD60 = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/processed/BinomialEstimate/rawdata/10419/SPSVD60.txt")
SPARSE_SVD80 = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/processed/BinomialEstimate/rawdata/10419/SPSVD80.txt")

setwd('D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/0217-large/processed')
STOCHASTIC10 = read.delim('partTSNE.txt')
SPARSE_SVD10 = read.delim('partSPARSESVD.txt')

fig_2d=function(Estimated, f, b=144, groups='label'){
  txt = paste('italic(Filtered)', ' == ', round(1- f/b, 4))
  min_x = min(Estimated$D0)
  max_y = max(Estimated$D1)
  if (groups=='label') {
    g = Estimated$Label
  }
  else if(groups=='type'){
    g = Estimated$Type
  }
  else if(groups=='set'){
    g = Estimated$set
  }
  else{
    return()
  }
  
  p = ggplot(Estimated, aes(D0, D1)) + 
    geom_point(aes(color=g)) + 
    annotate('text', x=min_x * 0.9, y=max_y * 1.1, label = txt, parse = TRUE) + 
    labs(x = "DIMENSION I", y="DIMENSION II", color='Tendency', title = 'Eigen-Features Extraction 2D View') +
    theme_bw()
  print(p)  
}

EstimatedBinomialFlow = STOCHASTIC10

EstimatedBinomialFlow = SPARSE_SVD10

head(EstimatedBinomialFlow)
colnames(EstimatedBinomialFlow) = c('Type', 'set', 'Label', 'D0', 'D1')
head(EstimatedBinomialFlow)

fig_2d(EstimatedBinomialFlow, 92, )
