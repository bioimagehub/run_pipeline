library(ggplot2)
#library(tidyverse)
#library(patchwork)
#library(gridExtra)


# Get current script directory (VS Code compatible)
# (current_path = dirname(rstudioapi::getSourceEditorContext()$path))  # RStudio only
current_path = "E:/Coen/Sarah/6849908-IMB-Coen-Sarah-Photoconv_global"

folder_paths = c("/quantification_bulk") # "/quantification_edge"

#path = folder_paths[1]
dfs = lapply(folder_paths, function(path){
  files = list.files(paste0(current_path, path), pattern = "bulk_per_distance.tsv$",full.names = T)
  print(paste("Found", length(files), "files"))
  #files = files[1:10]
  dfs = lapply(files, read.csv, header = T, sep = "\t" )
  df = do.call(rbind, dfs)
  df$folder = path
  df
})

df = do.call(rbind, dfs)

head(df)


# signal distance spread over time
# ggplot(df) +
#   geom_smooth(aes(x = Timepoint, y = Mask_Index , col = Group)) +
#   #facet_wrap(~folder) +
#   NULL


ggplot(df[df$Timepoint == 1, ]) +
  geom_boxplot(aes(x = Group, y = Sum_50_Random , col = Replicate)) +
  theme_classic() + 
  #facet_wrap(~folder) +
  NULL











