library(pheatmap)
library(ggplot2)
library(cluster)
library(factoextra)
ExprBinomialFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/texts/ExprBinomialFlow.txt")
ExprBinomialFlow = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/0217-small/texts/ExprBinomialFlow.txt")

setwd('D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/0217-large/texts')
Sig = read.table('SigScoreBinomialFlow.txt', header = T)



zero = function(data){
  row = rownames(data)
  data = t(data)
  data = data[3:dim(data)[1],]
  d = apply(data, 1, as.numeric)
  z_data = apply(d, 2, scale)
  z_data = t(z_data)
  colnames(z_data) = row
  m = apply(z_data, 1, function(x){round(sum(x), 2)})
  s = apply(z_data, 1, function(x){round(sd(x), 2)})
  print(m)
  print(s)
  return(z_data)
}


gid2gname_rows = function(mat){
  egenes_id = data.frame(data = as.numeric(chartr('X', '0', rownames(mat)))) 
  egenes = as.vector(apply(egenes_id, 1, function(r){
    Sig$GENE[which(Sig$IDX == r)]
  }))
  rownames(mat) = egenes
  return(mat)
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


dim_groups = function(dataframe){
  pos = which(dataframe$label == 1)
  dataframe$label[pos] = rep('Positive', length(pos))
  neg = which(dataframe$label == -1)
  dataframe$label[neg] = rep('Negative', length(neg))
  
  ret = data.frame(dataframe$label, dataframe$ctype)
  
  colnames(ret) = c('Label', 'Type')
  return(ret)
}


break_scale = function(dataframe){
  ret = c()
  f = 1
  s = 0
  for (i in unique(dataframe$label)) {
    s = s + length(which(dataframe$label == i))
    ret[f] = s
    f = f + 1
  }
  return(ret)
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


fig_heatmap = function(dataframe, k=3, width, height, hcolor=NULL, save=T, show.colname=F){
  title = "Eigen Genes Expression"
  
  data = zero(dataframe)
  data = gid2gname_rows(data)
  
  groups = dim_groups(dataframe)
  breaks = break_scale(dataframe)
  ranks = sep_rank(rownames(data))
  
  label_colors = list(
    Label = c(Positive="#143a5e", Negative="#c9e5ff")
  )
  
  if (is.null(hcolor)) {
    hcolor = colorRampPalette(c("#00576d", "#fffdcc", "#cd009a"))(100)
  }
  
  fig = pheatmap(t(data), cluster_rows=F, cluster_cols=T, cutree_cols = k,
                 annotation_row=groups, annotation_col = ranks, gaps_row = breaks,
                 cellwidth = 10, cellheight = 2, border=F,
                 show_rownames=F, show_colnames=show.colname, main = title,
                 color=hcolor ,annotation_colors=label_colors)
  print(fig)
  if (save) {
    ggsave("BinomialEigenesExpr.tiff", fig, width = width, height = height, limitsize=F)
  }
  return(fig)
}


clusting = function(dataset, k_value, color=NULL, layer=F, star.plot = T, repel = T, labelsize=12){
  if(layer){
    result = dist(dataset, method = "euclidean")
    result_hc = hclust(d = result, method = "ward.D2")
    fviz_dend(result_hc, cex = 0.6)
    a = fviz_dend(result_hc, k = k_value, 
              cex = 0.5, 
              k_colors = color,
              color_labels_by_k = TRUE, 
              rect = TRUE          
    )
  }
  else{
    km = c()
    km.res = kmeans(dataset, k_value, nstart = 24)
    km.fig = fviz_cluster(km.res, data = dataset,
                 palette = color,
                 # ellipse.type = "euclid",
                 star.plot = star.plot, 
                 repel = F,
                 labelsize = labelsize,
                 ggtheme = theme_bw()
    )
    
    
    # 提取轮廓图
    sil = silhouette(km.res$cluster, dist(m))
    rownames(sil) = rownames(m)
    km.sil = sil
    
    km.sh = fviz_silhouette(sil)
    
    
    # 负轮廓系数
    neg_sil_index = which(sil[, "sil_width"] < 0)
    km.negsil = sil[neg_sil_index, , drop = F]
    
    print(km.fig)
    print(km.sh)
    print(km.negsil)
    
    return(sil)
  }
}





df = ExprBinomialFlow
df = ExprTimeFlow

mat = zero(df)
m = gid2gname_rows(mat)
set.seed(1234)

# 检查聚类性能 < 0.5
res = get_clust_tendency(m, 40, graph = TRUE)
res$hopkins_stat
res$plot


# 获取最佳聚类K值
get_k_value(m, method = 'ap', k_max = floor(nrow(mat)/10), iter=100)
k = 2

# K-means聚类可视化及评估
sil = clusting(m, k, layer=F, repel = T, labelsize = 0)
cls = data.frame(sil[, 'cluster'])
colnames(cls) = c('cluster')
write.table(cls, file = "ClsBinomail.txt", 
            append = F, quote = F, 
            sep = "\t",eol = "\n", na = "NA", dec = ".", 
            row.names = T,col.names = T)

# H-clust聚类可视化及评估
clusting(m, k, layer=T)


# 绘制表达热图
hcol = colorRampPalette(c("#336666", "#FFCC99", "#C65146"))(100)
expr = fig_heatmap(df, k, 30, 30, hcolor = hcol, save=F)

clsOrder = function(dataframe, k){
  mat = zero(dataframe)
  row_cluster = cutree(expr$tree_col, k=k)
  newOrder=mat[expr$tree_col$order,]
  row_cluster = row_cluster[match(rownames(newOrder), names(row_cluster))]
  newOrder = cbind(newOrder, row_cluster)
  colnames(newOrder)[ncol(newOrder)]="Cluster"
  return(data.frame(newOrder))
}

clsmat = clsOrder(df, k)
clsmat$Cluster
clsmat = gid2gname_rows(clsmat)
write.table(clsmat, file = "ClsExprBinomail.txt", 
            append = F, quote = F, 
            sep = "\t",eol = "\n", na = "NA", dec = ".", 
            row.names = T,col.names = T)








df = ExprBinomialFlow

setwd('D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/0217-large/texts')
ppos = read.table('SVMPredictExprPos.txt', header=T)
pneg = read.table('SVMPredictExprNeg.txt', header=T)
ppos$label = rep(1, length(ppos$label))
df = rbind(ppos,pneg)

setwd('D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/0217-large/processed')

findGeEXP = function(data, tar='gt.txt'){
  a = as.factor(colnames(data)[3:dim(data)[2]])
  gt = read.table(tar, header = F)
  b = as.numeric(chartr('X', '0', a))
  idx = which(b %in% gt$V1)
  nidx = c("label", "ctype", as.vector(a[idx]))
  ndf = df[nidx]
  return(ndf)
}

ndf = findGeEXP(df)
dim(ndf)
groups = dim_groups(ndf)
rownames(ndf) = rownames(groups)
hcol = colorRampPalette(c("#336666", "#FFCC99", "#C65146"))(100)
fig_heatmap(ndf, k=1, 30, 30, hcolor = hcol, save=F, show.colname = T)


fig_heatmap = function(dataframe, k=3, width, height, hcolor=NULL, save=T, show.colname=F){
  title = "Eigen Genes Expression"
  
  data = t(ndf)
  data = data[3:dim(data)[1],]
  d = apply(data, 1, as.numeric)
  rownames(d) = rownames(dataframe)
  data = t(d)
  data = gid2gname_rows(data)
  
  groups = dim_groups(dataframe)
  breaks = break_scale(dataframe)
  ranks = sep_rank(rownames(data))
  
  label_colors = list(
    Label = c(Positive="#143a5e", Negative="#c9e5ff")
  )
  
  if (is.null(hcolor)) {
    hcolor = colorRampPalette(c("#00576d", "#fffdcc", "#cd009a"))(100)
  }
  
  fig = pheatmap(t(data), cluster_rows=F, cluster_cols=T, cutree_cols = k,
                 annotation_row=groups, annotation_col = ranks, gaps_row = breaks,
                 cellwidth = 10, cellheight = 2, border=F,
                 show_rownames=F, show_colnames=show.colname, main = title,
                 color=hcolor ,annotation_colors=label_colors)
  print(fig)
  if (save) {
    ggsave("BinomialEigenesExpr.tiff", fig, width = width, height = height, limitsize=F)
  }
  return(fig)
}



