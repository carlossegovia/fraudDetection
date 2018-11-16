# Title     : TODO
# Objective : TODO
# Created by: figuerru
# Created on: 9/11/18

library(rgl)

clusters_data <-
    read.csv(pipe("/Users/figuerru/PycharmProjects/fraudDetectionPlotting/part-00000"))
clusters <- clusters_data[1]
data <- data.matrix(clusters_data[ -c(1)])
rm(clusters_data)

# Make a random 3D projection and normalize
random_projection <- matrix(data = rnorm(3*ncol(data)), ncol = 3)
random_projection_norm <-
  random_projection / sqrt(rowSums(random_projection*random_projection))

# Project and make a new data frame
projected_data <- data.frame(data %*% random_projection_norm)

num_clusters <- max(clusters)
palette <- rainbow(num_clusters)
colors = sapply(clusters, function(c) palette[c])
plot3d(projected_data, col = colors, size = 10)