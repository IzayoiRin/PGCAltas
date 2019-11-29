setwd()
TPM <- read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/TPM.txt")
tpm = TPM[,c(7:12)]
r = TPM[,1][!duplicated(TPM[,1])]
tpm = tpm[r,]
rownames(tpm) = r
head(tpm)

zero = function(data){
  col = colnames(data)
  d = apply(data, 1, as.numeric)  
  z_data = apply(d, 2, scale)
  z_data = t(z_data)
  colnames(z_data) = col
  return(z_data)
}


get_k_value = function(dataset, method='sse', k_max=10, iter=500){
  library(ggplot2)
  library(factoextra)
  ret = FALSE
  # confirm k-value
  # by SSE 
  if (method == 'sse'){
    fig = fviz_nbclust(dataset, kmeans, method = "wss",k.max = k_max) +
      geom_vline(xintercept = 3, linetype = 2)
  }
  # by AP
  else if(method == 'ap'){
    library(apcluster)
    ap_clust = apcluster(negDistMat(r=2), dataset)
    fig = heatmap(ap_clust)
    ret = length(ap_clust@clusters)
  }
  # by SSB
  else if(method == 'ssb'){
    library(vegan)
    ca_clust = cascadeKM(dataset, 1, k_max, iter = iter)
    calinski.best = as.numeric(which.max(ca_clust$results[2,]))
    fig = plot(ca_clust, sortg = TRUE, grpmts.plot = TRUE)
    ret = calinski.best
  }
  # by ASC
  else if(method == 'as'){
    fig = fviz_nbclust(dataset, kmeans, method = "silhouette", k.max = k_max)
  }
  # by GAP
  else if (method == 'gap') {
    library(cluster)
    gap_clust = clusGap(dataset, kmeans, k_max, B = iter, verbose = interactive())
    fig = fviz_gap_stat(gap_clust)
    ret = which.max(gap_clust$Tab[,'gap'])
  }
  print(fig)
  return(ret)
}

clusting = function(dataset, k_value, layer=F){
  if(layer){
    result = dist(dataset, method = "euclidean")
    result_hc = hclust(d = result, method = "ward.D2")
    fviz_dend(result_hc, cex = 0.6)
    fviz_dend(result_hc, k = k_value, 
              cex = 0.5, 
              # k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
              color_labels_by_k = TRUE, 
              rect = TRUE          
    )
  }
  else{
    km.res = kmeans(dataset, k_value, nstart = 24)
    fviz_cluster(km.res, data = dataset,
                 # palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
                 ellipse.type = "euclid",
                 star.plot = TRUE, 
                 repel = TRUE,
                 ggtheme = theme_minimal()
    )
  }
}

set.seed(1234)


dat = zero(tpm)

tdat = t(dat)
k = get_k_value(tdat, method = 'sse', k_max = 5, iter=100)
k = 2
clusting(tdat, k, layer=F)
clusting(tdat, k, layer=T)
