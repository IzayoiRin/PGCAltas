library(ggplot2)
RDFLocFlow <- read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/SigScoreLocFlow.txt")
RDFTimeFlow <- read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/SigScoreTimeFlow.txt")

setwd('D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/0217-large/texts')
SigScoreBinomialFlow = read.table('SigScoreBinomialFlow.txt', header = T)
head(SigScoreBinomialFlow)

df = RDFLocFlow
# df = RDFTimeFlow
df = SigScoreBinomialFlow

non_zore = function(df){
  df = df[which(df$AREA > 0), ]
  # df$GENE = factor(df$GENE, levels=df[, 2], ordered = T)
  df$SIGNIFY = as.factor(df$SIGNIFY)
  df$RANK = rev(c(1: length(df$IDX)))
  return(df)
}

fig_area = function(df, name, width, height){
  total = paste('italic(Total)', dim(df)[1], sep=': ')
  df = non_zore(df)
  
  h = max(df$IMP)
  true_rows = which(df$SIGNIFY=='True')
  b = length(true_rows)-1
  s = length(df$SIGNIFY) - 1
  t = df[true_rows, ]$AREA
  
  thershold = paste('italic(Threshold)', ' == ', round(1 - min(t) / max(t), 2))
  dark = "#666666"
  light = "#CC9966"
  cols = c(dark, light, dark)
  area_cols = c(light, dark)
  
  fig = ggplot(df, aes(x=RANK, y=IMP)) +
    geom_area(aes(fill=SIGNIFY, color=SIGNIFY)) +
    scale_color_manual(breaks = c("False", "True"),
                       values=area_cols) +
    scale_fill_manual(breaks = c("False", "True"),
                       values=area_cols) +
    scale_x_continuous(breaks=c(0, b, s))+
    labs(x = "GENE RANK", y = 'IMP SCORE', fill='Significant', color='Significant', title = 'Density Curve of Importance') +
    annotate('text', x=floor(s * 0.1), y=floor(h * 0.95), label=total, parse = TRUE) +
    annotate('text', x=floor(s * 0.1), y=floor(h * 0.9), label=thershold, parse = TRUE) +
    theme_bw() +
    theme(axis.text.x = element_text(angle=90, hjust = 0.5, vjust = 0.5, face = 'bold', color=cols),
          axis.text.y = element_text(face='bold', color = dark))
  
  print(fig)
  ggsave(paste("ImpDensityCurve", name, ".tiff"), fig, width = width, height = height)
}

fig_area(df, 'Binomal', 12, 5)

