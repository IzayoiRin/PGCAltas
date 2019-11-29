library(ggplot2)

RDFLocFlow <- read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/SigScoreLocFlow.txt")
RDFTimeFlow <- read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/SigScoreTimeFlow.txt")

df = RDFLocFlow
# df = RDFTimeFlow


non_zore = function(df){
  df = df[which(df$AREA > 0), ]
  df$GENE = factor(df$GENE, levels=df[, 2], ordered = T)
  df$SIGNIFY = as.factor(df$SIGNIFY)
  df$RANK = rev(c(1: length(df$IDX)))
  return(df)
}

fig_area = function(df){
  total = paste('Total', dim(df)[1], sep=':')
  df = non_zore(df)

  b = length(which(df$SIGNIFY=='True'))-1
  s = length(df$SIGNIFY) - 1
  fig = ggplot(df, aes(x=RANK, y=IMP)) +
    geom_area(aes(fill=SIGNIFY, color=SIGNIFY)) +
    scale_color_manual(breaks = c("False", "True"),
                       values=c("gray", "black")) +
    scale_fill_manual(breaks = c("False", "True"),
                       values=c("gray", "black")) +
    scale_x_continuous(breaks=c(0, b, s))+
    labs(x = "GENE RANK", y = 'IMP SCORE', fill='Significant', color='Significant', title = 'Density Curve of Importance') +
    annotate('text', x=1500, y=280, label=total) +
    annotate('text', x=1500, y=265, label='Threshold = 0.4') +
    theme_bw()
  return(fig)
}

print(fig_area(df))

