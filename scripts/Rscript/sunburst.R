library(sunburstR)
seq = read.csv(
  system.file('examples/visit-sequences.csv', package = 'sunburstR'),
  header=F, stringsAsFactors = F
)
head(seq)
sunburst(seq)


cell_path = cells_counts_path <- read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/PGCAltas/dataset/EMTAB6967/processed/cells_counts_path.txt")
head(cell_path)

d = matrix(unlist(strsplit(as.character(cell_path[,1]),"-")), nrow = 4)       
s = unique(c(d[1,],d[3,]))

t = unique(c(d[2,],d[4,]))
s = s[order(s)]
s

sunburst(cell_path, legendOrder = unique(unlist(strsplit(as.character(cell_path[,1]),"-"))))


c(s,t)

