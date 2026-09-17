#!/usr/bin/env Rscript
# Timing probe for spatstat.random::rStrauss (dominated CFTP): generation.tex, WP2(b).
# A LIBRARY CHECK, not data generation: each draw is timed, its point count kept, the
# pattern discarded. Answers one question: where in (tau, gamma, nbar) does exact
# simulation finish within the cap? That region becomes the Strauss prior constraint F.
#
#   Rscript strauss_cftp_probe.R <grid.csv> <cell_index> <out.csv> [reps=5] [cap_seconds=600]
#
# A draw that exceeds the cap is killed and recorded as "timeout". Timeouts here only map
# the feasible region; in DV3 generation a slow draw is NEVER dropped and redrawn, because
# CFTP output is not independent of its run time (generation.tex, Sec. 3.5).

suppressPackageStartupMessages(library(spatstat.random))
library(parallel)

args <- commandArgs(trailingOnly = TRUE)
grid <- read.csv(args[1])
g <- grid[as.integer(args[2]) + 1L, ]
out <- args[3]
reps <- if (length(args) >= 4) as.integer(args[4]) else 5L
cap <- if (length(args) >= 5) as.numeric(args[5]) else 600

rows <- vector("list", reps)
for (rep in seq_len(reps)) {
  seed <- 1000000L + 100L * g$cell + rep            # probe-only seeds, not a DV3 stream
  t0 <- proc.time()[["elapsed"]]
  job <- mcparallel({                                # forked child, so a slow draw can be killed
    set.seed(seed)
    X <- rStrauss(beta = g$beta, gamma = g$gamma, R = g$R)   # W = unit square, expanded by 2R
    c(X$n, attr(X, "times"))
  })
  res <- mccollect(job, wait = FALSE, timeout = cap)
  status <- "ok"
  val <- c(NA, NA, NA)
  if (is.null(res)) {
    tools::pskill(job$pid)
    mccollect(job, wait = FALSE)
    status <- "timeout"
  } else if (inherits(res[[1]], "try-error")) {
    status <- "error"
  } else {
    val <- res[[1]]
  }
  rows[[rep]] <- data.frame(cell = g$cell, tau = g$tau, gamma = g$gamma, nbar = g$nbar, R = g$R,
                            beta = g$beta, rep = rep, seed = seed, status = status,
                            seconds = proc.time()[["elapsed"]] - t0,
                            n = val[1], cftp_start = val[2], cftp_end = val[3])
}
write.csv(do.call(rbind, rows), out, row.names = FALSE)
