# Loads necessary R libraries and scripts: 'inla_fct.R':
require(INLA)
require(lattice)
require(classInt)
require(FITSio)
require(reshape2)
require(jsonlite)
source("inla_fct.R")

args = commandArgs(trailingOnly=TRUE)
# setwd('~/Documents/inla/')

# meta <- read_json('meta.json')
meta <- read_json(args[1])
data <- read.table(meta$input_inla, quote="\"")

# Run INLA pipeline on the mass map using a stationary model, elliptical distance, 
# and a voronoi mesh cutoff  of 1: 
data_inla <- stationary_inla(data$V2, data$V1, data$V3,
                             xsize=meta$data_dim[[2]], ysize=meta$data_dim[[1]],
                             xfin=meta$data_dim[[2]], yfin=meta$data_dim[[1]],
                             weight=1/data$V4,
                             shape='ellipse', cutoff=1,
                             )

# Save output
out <- toJSON(data_inla, pretty=TRUE,auto_unbox=TRUE)
write(out, paste(meta$output,'.json', sep=''))
