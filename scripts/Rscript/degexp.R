library(clusterProfiler)
library(org.Mm.eg.db)


db = org.Mm.eg.db

c2up = c('Cdx2', 'Phlda2', 'Ifitm1', 'Msx2', 'Psme1', 'Ifitm3', 'Psme2', 'Msx1', 'Gjb3', 'Rspo3', 'Aldh2', 'Dppa3', 'Psmb8', 'Vim', 
            'Igfbp3', 'S100a11', 'Id1', 'Tmem185a', 'Dnd1', 'Tapbp', 'Efna1', 'Fgf8', 'Sct', 'Pdlim4', 'Pdgfa', 'T', 'Sprr2a3', 'Dlk1',
            'Ifitm2', 'Lrpap1', 'Slc2a3', 'Bmp4', 'Hand1', 'H2afy', 'Klf2', 'Lxn', 'Dnaaf3', 'Isl1', 'Evx1os', 'Zfand6', 'Cited1', 'Croth')
c1down = c('Car4', 'Dnmt3b', 'Igfbp2', 'G3bp2', 'Ptma', 'Sumo2', 'Fgf5', 'Rangrf', 'Pim2', 'Ranbp1', 'Ybx1', 'Ran', 'Eif5a', 'Ddx39', 'Lyar',
          'Lsm2', 'Hnrnpa1', 'Otx2', 'H2afz', 'Sumo1', 'Pcsk1n', 'Cycs', 'Nme1', 'Rps27l', 'Pfn1', 'Hnrnpf', 'Cbx1', 'Srsf3', 'Srm', 'Timm13', 
          'Snrpn', 'Glrx5', 'Higd1a', 'Cfl1', 'Cnbp', 'Calm1', 'Utf1', 'Srsf2', 'Odc1', 'Tomm20', 'Atp5g1', 'Sssca1', 'Hspe1', 'Pou3f1', 
          'Pebp1', 'Serbp1', 'Hmgn1', 'Ncl', 'Fkbp4', 'Srsf7', 'Snrpa1', 'Phb2', 'Snrpd1', 'Ldha', 'Npm1', 'Fkbp1a', 'Slc7a3', 'Srsf1', 
          'Zfp706', 'Prmt1', 'Tomm5', 'Tuba1a', 'Mthfd2')

ranktop = c('Cdx2', 'Phlda2', 'Ifitm1', 'Msx2', 'Psme1', 'Ifitm3', 'Psme2', 'Msx1', 'Gjb3', 'Car4', 'Dnmt3b', 'Igfbp2', 'G3bp2', 'Rspo3',
            'Ptma', 'Sumo2', 'Aldh2', 'Dppa3', 'Fgf5', 'Psmb8', 'Rangrf')






setwd('D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/PGCAltas/dataset/EMTAB6967/0217-large/processed')
cls = read.delim('genesEO.txt', header = F)
g = as.vector(cls$V1)

gene = g
gene
ego = enrichGO(
  gene = gene,
  OrgDb = db,
  keyType = "SYMBOL",
  ont = "BP"
)

print(dotplot(ego))
