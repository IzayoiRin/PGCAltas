#######################library
library(ggplot2)
library(reshape2)
library(ggcor)
library(pheatmap)
library(DESeq2)
library(stringr)
library(clusterProfiler)
library(org.Mm.eg.db)


sampleNames <- c("C1", "C2", "C3", "M2", "M3", "M4")

data <- read.table("sample_counts.txt", header=TRUE, quote="\t", skip=1)

names(data)[7:12] <- sampleNames
countData <- as.matrix(data[7:12])
rownames(countData) <- data$Geneid
database <- data.frame(name=sampleNames, condition=c("C", "C", "C", "M", "M", "M"))
rownames(database) <- sampleNames


dds <- DESeqDataSetFromMatrix(countData, colData=database, design= ~ condition)
dds <- dds[ rowSums(counts(dds)) > 1, ]

### set control or alphabtically
### dds$condition <- relevel(dds$condition, ref = "C")

dds <- DESeq(dds)
res <- results(dds)
write.csv(res, "res_des_output.csv")
resdata <- merge(as.data.frame(res), as.data.frame(counts(dds, normalized=TRUE)),by="row.names",sort=FALSE)
write.csv(resdata, "all_des_output.csv", row.names=FALSE)


############# plot MA
plotMA(res, main="DESeq2", ylim=c(-2, 2))


#### factor data
resdata$change <- as.factor(
  ifelse(
    resdata$padj<0.05 & abs(resdata$log2FoldChange)>1,
    ifelse(resdata$log2FoldChange>1, "Up", "Down"),
    "NoDiff"
  )
)
valcano <- ggplot(data=resdata, aes(x=log2FoldChange, y=-log10(padj), color=change)) + 
  geom_point(alpha=0.8, size=1) + 
  theme_bw(base_size=15) + 
  theme(
    panel.grid.minor=element_blank(),
    panel.grid.major=element_blank()
  ) + 
  ggtitle("DESeq2 Valcano") + 
  scale_color_manual(name="", values=c("red", "green", "black"), limits=c("Up", "Down", "NoDiff")) + 
  geom_vline(xintercept=c(-1, 1), lty=2, col="gray", lwd=0.5) + 
  geom_hline(yintercept=-log10(0.05), lty=2, col="gray", lwd=0.5)


############# plot Volcano
valcano



#########################################  GO analysis ###################################################################

library(clusterProfiler)
library(org.Mm.eg.db)
library(stringr)
library(DOSE)

db = org.Mm.eg.db

d<-countData
vs<-"C_M"

for (w in c("all", "up", "down")) {
  
  if(w=="all"){
    deseq2.sig<-subset(res, padj<0.05 & abs(log2FoldChange) > 1)
  }else if(w=="up"){
    deseq2.sig<-subset(res, padj<0.05 & log2FoldChange > 1)
  }else if(w=="down"){
    deseq2.sig<-subset(res, padj<0.05 & log2FoldChange < -log10(1))
  }
  
  
  sig <- merge(as.data.frame(deseq2.sig), as.data.frame(counts(dds, normalized=TRUE)),by="row.names",sort=FALSE)
  
  m <- sig[,8:(7+ncol(d))]
  m <- apply(m,1,scale)
  m<-t(m)
  gene<-sig[,1]
  for (i in 1:length(gene)) {
    a <- str_split(gene[i],"\\.")[[1]][1];
    gene[i] <- a
  }
  gene <- mapIds(db, keys = gene, column = "SYMBOL", keytype = "ENSEMBL")
  rownames(m)<-gene
  
  
  
  gene<-gene[!is.na(gene)]
  write.csv(gene, paste0(vs, "_go_", w, "_gene_output.csv"))
  
  m<-m[gene,]
  colnames(m)<-colnames(sig)[8:(7+ncol(d))]
  
  pheatmap(m, filename = paste0(vs, "_go_", w, "_heatmap.pdf"), width = 8, height = 8, border_color = NA)
  
  
  ego <- enrichGO(
    gene = gene,
    OrgDb = db,
    keyType = "SYMBOL",
    ont = "BP"
  )
  
  pdf(paste0(vs, "_go_",w,".pdf"), width = 12, height = 8)
  print(dotplot(ego))
  dev.off()
  
}



dotplot(ego, font.size = 5)
emapplot(ego)
plotGOgraph(ego)

genelist <- deseq2.sig$log2FoldChange # also can do padj
names(genelist) <- gene
genelist <- sort(genelist, decreasing = TRUE)
gsemf <-gseGO(
  genelist,
  OrgDb = db,
  keyType = "ENSEMBL",
  ont = "BP"
)




head(gsemf)

gseaplot(gsemf, geneSetID = "GO:0004871")

#########################################  KEGG analysis ###################################################################

gene_list <- mapIds(db, keys = gene, column = "ENTREZID", keytype = "ENSEMBL")
ekg <- enrichKEGG(gene_list,
                  keyType = "ncbi-geneid",
                  pvalueCutoff = 0.05, pAdjustMethod = "BH",
                  qvalueCutoff = 0.1)

dotplot(ekg)

g <- c(664783,78887,216881,18191,20191,100040852,100041678,268816,223726,268859,16522,13508,100861531,12941,93882,17919,329942,100038949,13483,18125,14738,210274,12554)

g<- as.character(g)

g <- mapIds(db, keys = g, column = "SYMBOL", keytype = "ENTREZID")

write.csv(g, "gene.csv")





