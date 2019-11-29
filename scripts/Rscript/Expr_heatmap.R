library(pheatmap)
library(ggplot2)


zero = function(data){
  row = rownames(data)
  data = t(data)
  data = data[4:dim(data)[1],]
  d = apply(data, 1, as.numeric)
  z_data = apply(d, 2, scale)
  z_data = t(z_data)
  colnames(z_data) = row
  return(z_data)
}


sep_rank = function(genes){
  sep = floor(length(genes) / 10)
  start = 1
  flag = c(10, 20, 80)
  f = 1
  ret = c()
  for (i in c(1, 1, 6)) {
    last = start + i*sep - 1
    g = genes[start: last]
    ret = c(ret, rep(flag[f], times=i*sep))
    start = last + 1
    f = f + 1
  }
  ret = c(ret, rep(100, times=length(genes) - start + 1))
  ret = as.data.frame(rev(ret), row.names = genes)
  colnames(ret) = 'Rank'
  return(ret)
}


dim_groups = function(dataframe){
  ret = data.frame(df$loci, df$layer, df$stage)
  colnames(ret) = c('Loci', 'Layer', 'Stage')
  return(ret)
}


break_scale = function(dataframe){
  ret = c()
  f = 1
  s = 0
  for (i in unique(dataframe$stage)) {
    s = s + length(which(df$stage == i))
    ret[f] = s
    f = f + 1
  }
  return(ret)
}


fig_heatmap = function(dataframe, k=3, width, height){
  data = zero(dataframe)
  
  title = "Eigen Genes Expression"
  
  groups = dim_groups(dataframe)
  breaks = break_scale(dataframe)
  ranks = sep_rank(rownames(data))
  
  label_colors = list(
    Stage = c('#ff7272', "#ff0000"),
    Loci = c('#ae72ff', '#6c00ff'),
    Layer = c(EP="#ffb15e", P="#adff80")
  )
  
  hcolor = colorRampPalette(c("#00576d", "#fffdcc", "#cd009a"))(100)
  
  fig = pheatmap(data, cluster_rows=T, cluster_cols=F, cutree_rows = k,
                 annotation_col=groups, annotation_row = ranks, gaps_col = breaks,
                 cellwidth = 10, cellheight = 2, border=F,
                 show_rownames=F, show_colnames=F, main = title,
                 color=hcolor ,annotation_colors=label_colors)
  print(fig)
  ggsave("TimeEigenesExpr.tiff", fig, width = width, height = height)
  return(fig)
}


get_k_value = function(dataset, method='sse', k_max=10, iter=500){
  library(ggplot2)
  library(factoextra)
  # confirm k-value
  # by SSE 
  if (method == 'sse'){
    fviz_nbclust(dataset, kmeans, method = "wss",k.max = k_max) +
      geom_vline(xintercept = 3, linetype = 2)
  }
  # by AP
  else if(method == 'ap'){
    library(apcluster)
    ap_clust = apcluster(negDistMat(r=2), dataset)
    heatmap(ap_clust)
    return(length(ap_clust@clusters))
  }
  # by SSB
  else if(method == 'ssb'){
    library(vegan)
    ca_clust = cascadeKM(dataset, 1, k_max, iter = iter)
    calinski.best = as.numeric(which.max(ca_clust$results[2,]))
    plot(ca_clust, sortg = TRUE, grpmts.plot = TRUE)
    return(calinski.best)
  }
  # by ASC
  else if(method == 'as'){
    fviz_nbclust(dataset, kmeans, method = "silhouette", k.max = k_max)
  }
  # by GAP
  else if (method == 'gap') {
    library(cluster)
    gap_clust = clusGap(dataset, kmeans, k_max, B = iter, verbose = interactive())
    print(fviz_gap_stat(gap_clust))
    return(which.max(gap_clust$Tab[,'gap']))
  }
}


clsOrder = function(dataframe, k){
  mat = zero(dataframe)
  row_cluster = cutree(expr$tree_row, k=k)
  newOrder=mat[expr$tree_row$order,]
  row_cluster = row_cluster[match(rownames(newOrder), names(row_cluster))]
  newOrder = cbind(newOrder, row_cluster)
  colnames(newOrder)[ncol(newOrder)]="Cluster"
  return(newOrder)
}

ExprLocFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/LocFlow/ExprLocFlow.txt")
ExprTimeFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/processed/TimeFlow/ExprTimeFlow.txt")
# df  = ExprLocFlow
df = ExprTimeFlow
head(df)
# determine k value
mat = zero(df)
set.seed(1234)
get_k_value(mat, method = 'gap', k_max = floor(nrow(mat)/40), iter=1000)

# expr heapmap
expr = fig_heatmap(df, 8, 18, 12)

# cluster expr matrix
mat = zero(df)
clsmat = clsOrder(df, 2)
write.table(clsmat, file = "ClsExprTimeFlow.txt", 
            append = F, quote = F, 
            sep = "\t",eol = "\n", na = "NA", dec = ".", 
            row.names = T,col.names = T)
