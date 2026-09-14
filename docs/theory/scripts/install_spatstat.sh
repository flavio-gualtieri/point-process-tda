#!/bin/bash
# One-time setup for the Strauss route of docs/theory/generation.tex (WP0):
# install spatstat.random and its spatstat.* dependencies into a user library in
# $HOME (not /gpfs/scratch, which is purged), then record the exact versions,
# which the DV3 manifest pins. Needs network (CRAN), so run it on the login node:
#
#   bash docs/theory/scripts/install_spatstat.sh      # from the repo root
#
# It compiles C/C++ for a few minutes. If the R container cannot compile, this is
# where it fails; the fallback is the Metropolis-Hastings route (generation.tex, Sec. 3.5).

set -euo pipefail
module load R/4.5.1
LIB="$HOME/R/library-4.5"
mkdir -p "$LIB"
Rscript -e "install.packages('spatstat.random', lib = '$LIB', repos = 'https://cloud.r-project.org', Ncpus = 2)"
Rscript -e ".libPaths(c('$LIB', .libPaths())); ip <- installed.packages(); \
  v <- ip[grepl('^spatstat', rownames(ip)), 'Version']; print(v); \
  cat(R.version.string, '\n'); writeLines(paste(names(v), v), 'docs/theory/scripts/out/spatstat_versions.txt')"
