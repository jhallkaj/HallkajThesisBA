# Setup -------------------------------------------------------------------

library(vegan)
library(ggplot2)

# Set the working directory
setwd("C:/Users/Lenovo/Desktop/Thesis Data")
options(max.print = .Machine$integer.max)

# Load the data
data_1l = read.csv("OMP8X_1_LOW.TXT", header = FALSE)
data_1h = read.csv("OMP8X_1_HIGH.TXT", header = FALSE)
data_2l = read.csv("OMP8X_2_LOW.TXT", header = FALSE)
data_2h = read.csv("OMP8X_2_HIGH.TXT", header = FALSE)
data_3l = read.csv("OMP8X_3_LOW.TXT", header = FALSE)
data_3h = read.csv("OMP8X_3_HIGH.TXT", header = FALSE)
data_4l = read.csv("OMP8X_4_LOW.TXT", header = FALSE)
data_4h = read.csv("OMP8X_4_HIGH.TXT", header = FALSE)
data_list = list(data_1h,data_2h,data_3h,data_4h,data_1l,data_2l,data_3l,data_3l)
data_list = lapply(data_list, function(x) t(x))
odorants = read.csv("odorants.csv")
high_list = data_list[1:4]
low_list = data_list[5:8]
# Indexed data_list with row names
data_list = lapply(data_list, function(x) as.matrix(x))
idxd_list = lapply(data_list, function(x) {
  rownames(x) = odorants$Odorant
  return(x)
})



# Distance Matrices -------------------------------------------------------

# Transform the data into Distance Matrices

dist_list = lapply(data_list, function(x) as.matrix(dist(x, method="euclidean", upper=TRUE, diag=TRUE)))
print(dist_list[[1]])
# double checking dimensions here
dimensions = lapply(dist_list, function(x) dim(x))
# print(dimensions) # To check the dimensions of the distance matrices

dist_idxd = dist_list = lapply(idxd_list, function(x) as.matrix(dist(x, method="euclidean", upper=TRUE, diag=TRUE)))
names(dist_list)= c("Glomerulus 1 High", "Glomerulus 2 High", "Glomerulus 3 High", "Glomerulus 4 High",
                    "Glomerulus 1 Low", "Glomerulus 2 Low", "Glomerulus 3 Low", "Glomerulus 4 Low")

# We're back in business baby! Do the Heatmaps bad boy

heatmap(dist_idxd[[1]])
heatmap(dist_idxd[[5]])

# Mantel test
x = mantel_test(dist_idxd[[2]], dist_idxd[[2]])
print(x)


mt_list = mapply(function(x,y) mantel_test(x,y), high_list, low_list, SIMPLIFY = FALSE)
print(mt_list)

# Why am I doing this?

# Mantel Correlogram
mcl = mantel.correlog(dist_idxd[[1]], dist_idxd[[5]])
print(mcl)
plot(mcl)
 
mcl_list = mapply(function(x,y) mantel.correlog(x,y), high_list, low_list, SIMPLIFY = FALSE)

# Mantel for all.

mantel_result = list()
for (i in 1:(length(dist_list) - 1)) {
  for (j in (i + 1):length(dist_list)) {
    mantel_result = mantel(dist_list[[i]], dist_list[[j]])
    mantel_results[[paste0("Mantel_", i, "_", j)]] = mantel_result
  }
}
mantel_result


head(dist_list[[1]])
