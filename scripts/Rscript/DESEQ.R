
library("DESeq2")


sampleNames <- c("C2", "C5", "M1", "M5")
# 第一行是命令信息，所以跳过
data <- read.table("./data/wyh/wyh_counts_1023/sample_counts.txt", header=TRUE, quote="\t", skip=1)
# 前六列分别是Geneid Chr Start End Strand Length
# 我们要的是count数，所以从第七列开始
names(data)[c(8,9,10,12)] <- sampleNames
countData <- as.matrix(data[c(8,9,10,12)])
rownames(countData) <- data$Geneid
database <- data.frame(name=sampleNames, condition=c("C", "C", "M", "M"))
rownames(database) <- sampleNames

## 设置分组信息并构建dds对象
dds <- DESeqDataSetFromMatrix(countData, colData=database, design= ~ condition)
dds <- dds[ rowSums(counts(dds)) > 1, ]

## 使用DESeq函数估计离散度，然后差异分析获得res对象
dds <- DESeq(dds)
res <- results(dds)
write.csv(res, "res_des_output.csv")
resdata <- merge(as.data.frame(res), as.data.frame(counts(dds, normalized=TRUE)),by="row.names",sort=FALSE)
write.csv(resdata, "all_des_output.csv", row.names=FALSE)


plotMA(res, main="DESeq2", ylim=c(-2, 2))
ntd <- normTransform(dds)
library("pheatmap")
select <- order(rowMeans(counts(dds,normalized=TRUE)),
                decreasing=TRUE)[1:20]
df <- as.data.frame(colData(dds)[,c("condition")])
heatmap(assay(ntd)[select,])
vsd <- vst(dds, blind=FALSE)
rld <- rlog(dds, blind=FALSE)
pheatmap(assay(rld)[select,])
