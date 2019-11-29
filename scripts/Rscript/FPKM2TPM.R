fpkm2tpm = function(fpkm){
  exp(log(fpkm) - log(sum(fpkm)) + log(1e6))
}


count2tpm = function(counts, efflen){
  rate = log(counts) - log(efflen)
  denom = log(sum(exp(rate)))
  exp(rate - denom + log(1e6))
}


count2fpkm = function(counts, efflen){
  n = sum(counts)
  exp(log(counts) + log(1e9) - log(efflen) - log(n))
}


count2effcount = function(counts, len, efflen){
  counts * (len / efflen)
}


main = function(){
  expMatrix = read.table("D:\D\Desktop\SIBS-S324-IZ@YOI\work\project 1018\script\data\GSE120963\E6.5E1.txt", header = T, row.names = 1)
  head(expMatrix, 5)
  # tpms = apply(expMatrix, 2, fpkm2tpm)
  # write.table('D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/code/data/E5.5E1.txt', row.names = F, col.names = F)
}

main()




expMatrix = read.delim("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project 1018/script/data/GSE120963/E6.5E1.txt", row.names=1)
head(expMatrix, 5)
tpms = apply(expMatrix, 2, fpkm2tpm)
head(tpms)
