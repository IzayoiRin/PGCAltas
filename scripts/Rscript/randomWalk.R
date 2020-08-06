random_walk = read.csv("D:/D/Desktop/SIBS-S324-IZ@YOI/work/project1018/script/data/random_walk.txt", row.names=1)
random_walk$color = apply(random_walk, 1, function(x){x[6] = rgb(x[3], x[4], x[5])})
random_walk$step = rownames(random_walk)
head(random_walk)

label_walk = random_walk[seq(1, dim(random_walk)[1], 10),]
head(label_walk)

library(ggplot2)
fig = ggplot(random_walk, aes(position_x, position_y, group=1)) +
  geom_path(aes(color=color)) +
  geom_point(aes(color=color, alpha=0.1)) +
  geom_text(data=label_walk, aes(position_x, position_y, label=step), hjust = 0, nudge_x = 0.5, size=3) +
  theme_bw() +
  theme(legend.position = "none")
print(fig)
