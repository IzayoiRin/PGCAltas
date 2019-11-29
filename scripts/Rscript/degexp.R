library(clusterProfiler)
library(org.Mm.eg.db)


db = org.Mm.eg.db

Trank10 = c("Pmepa1","Hoxa1","Hoxb3","Laptm4b","Lpp","Hoxb2","Hoxb1","Cdx2","Phlda2","Zfp703","Lhx1","Gal",
            "G0s2","Gsc","Tspan13","Car3","Gbx2","Aldh1a2","Tbx6","Peg3as","Amph","Eomes","Arid3a","Fgf5",
            "Hoxa3","Foxb1","Nodal","Cyp26a1","Ndufb11","Serf2","Msgn1","Ifitm1","Dppa5a","Gpx4","Slc31a2","Rbm24")
Tcls2 = c('Cxcl12','Dnmt3b','Dppa5a','Fgf5','Gm13051','Gm20815','Gm4297','Gsc','L1td1','Myb','Nav2','Otx2',
          'Pmepa1','Rragd','Slc7a3','Smad7','Syt11','Unc5b','Utf1','Xist','Zfp345')

Lrank10 = c('Hkdc1','Pla2g12b','Podxl','Ppfibp2','Spink3','Afp','Tfpi','Trap1a','Col4a2','Soat2','Pdgfrl','Tmprss2')
Lcls3 = c('Afp','AI662270','Amot','Ang','Apoa1','Apoa4','Apob','Apoe','Apom','Atp1b1','B4galnt2','Cd59a','Clic6','Cpn1',
          'Ctsh','Cubn','Dab2','Dpp4','F5','Folr1','Gsn','Hkdc1','Malat1','Nostrin','Pga5','Pla2g12b','Podxl','Ppfibp2',
          'Prss12','Rab11fip5','Rhox5','Rnase4','Slc13a4','Slc39a5','Slc39a8','Soat2','Sox17','Spink3','Spp2','Stard8',
          'Tfpi','Tmprss2','Trap1a','Xlr4b','Xlr5a','Xlr5b')

gene = Lcls3

ego = enrichGO(
  gene = gene,
  OrgDb = db,
  keyType = "SYMBOL",
  ont = "BP"
)

print(dotplot(ego))
