# Options
options(max.print = 100)

# Load in CRAN packages
ll <- c('tidyverse','magrittr','forcats','stringr','cowplot','broom','scales','reshape2','ggrepel')
sapply(ll,library,character.only=T)

# Clear up
rm(list=ls())
# Set up directory
dir.base <- 'C:/Users/erikinwest/Documents/bioeconometrician/github/erikdrysdale.github.io/_rmd/extra_delta/'
setwd(dir.base)

######################################
###### ----- QUESTION 1  ----- #######
######################################

virus <- c(3,2,6,0,2,2,1,4,8,1,1,3,2,0,1,5,2,3,4,1)
n <- length(virus)