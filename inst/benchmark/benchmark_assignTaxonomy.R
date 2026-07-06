#!/usr/bin/env Rscript
# Benchmark assignTaxonomy wall time.
#
# Usage:
#   Rscript inst/benchmark/benchmark_assignTaxonomy.R
#   Rscript inst/benchmark/benchmark_assignTaxonomy.R --queries 1200 --genera 5000
#   Rscript inst/benchmark/benchmark_assignTaxonomy.R --target-secs 120
#   Rscript inst/benchmark/benchmark_assignTaxonomy.R 200          # light mode (no stress)
#
# Stress mode builds a synthetic training set (default 5000 genera, 3 refs/genus).
# Query sequences are recycled from inst/extdata/example_seqs.fa.
#
# Scaling guide (approximate, single-thread):
#   Runtime scales ~linearly with --queries and ~linearly with --genera.
#   If 100 queries @ 5000 genera takes ~10 s, then:
#     --queries 1200 --genera 5000  -> ~2 min
#     --queries 100  --genera 20000 -> ~40 s (heavier reference)

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  out <- list(
    queries = 200L,
    genera = 5000L,
    refs_per_genus = 3L,
    stress = TRUE,
    target_secs = NA_real_
  )
  if (length(args) == 0) return(out)

  # Legacy: numeric first arg = light mode replicate count
  if (length(args) >= 1 && grepl("^[0-9]+$", args[[1]]) && !startsWith(args[[1]], "--")) {
    if (length(args) >= 2 && tolower(args[[2]]) == "stress") {
      out$stress <- TRUE
      out$queries <- as.integer(args[[1]])
    } else {
      out$stress <- FALSE
      out$queries <- as.integer(args[[1]])
    }
    return(out)
  }

  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--queries" && i < length(args)) {
      out$queries <- as.integer(args[[i + 1L]]); i <- i + 2L
    } else if (a == "--genera" && i < length(args)) {
      out$genera <- as.integer(args[[i + 1L]]); i <- i + 2L
    } else if (a == "--refs-per-genus" && i < length(args)) {
      out$refs_per_genus <- as.integer(args[[i + 1L]]); i <- i + 2L
    } else if (a == "--target-secs" && i < length(args)) {
      out$target_secs <- as.numeric(args[[i + 1L]]); i <- i + 2L
    } else if (a == "--light") {
      out$stress <- FALSE; i <- i + 1L
    } else if (a == "--stress") {
      out$stress <- TRUE; i <- i + 1L
    } else {
      stop("Unknown argument: ", a)
    }
  }
  out
}

cfg <- parse_args(args)

if (!requireNamespace("dada2", quietly = TRUE)) {
  stop("Install dada2 before running this benchmark.")
}
library(dada2)

ref_path <- system.file("extdata", "example_seqs.fa", package = "dada2")
if (!nzchar(ref_path)) stop("example_seqs.fa not found in dada2 package")

base_seqs <- getSequences(ref_path)
long_seqs <- base_seqs[nchar(base_seqs) >= 50]
if (length(long_seqs) == 0) stop("No sequences >= 50 nts in example data")

build_stress_ref <- function(ref_path, ngenus_target, refs_per_genus, n_queries) {
  lines <- readLines(ref_path)
  sq <- tax <- character()
  for (i in seq(1, length(lines), by = 2)) {
    if (startsWith(lines[i], ">") && (i + 1) <= length(lines)) {
      s <- lines[i + 1]
      if (nchar(s) >= 50) {
        sq <- c(sq, s)
        tax <- c(tax, sub("^>", "", lines[i]))
      }
    }
  }
  train_path <- tempfile(fileext = ".fa")
  con <- file(train_path, "w")
  for (g in seq_len(ngenus_target)) {
    idx <- ((g - 1L) %% length(sq)) + 1L
    parts <- strsplit(tax[idx], ";", fixed = TRUE)[[1]]
    parts[6] <- paste0("Genus_", g)
    tstr <- paste0(paste(parts, collapse = ";"), ";")
    for (r in seq_len(refs_per_genus)) {
      writeLines(c(paste0(">", tstr), sq[idx]), con)
    }
  }
  close(con)
  list(
    ref_path = train_path,
    query_seqs = rep(sq, length.out = n_queries)
  )
}

if (cfg$stress) {
  n_queries <- cfg$queries
  if (!is.na(cfg$target_secs)) {
    # Calibrate with a short run, then scale query count to target duration.
    cat("Calibrating for --target-secs", cfg$target_secs, "...\n")
    cal <- build_stress_ref(ref_path, cfg$genera, cfg$refs_per_genus, 50L)
    on.exit(unlink(cal$ref_path), add = TRUE)
    gc()
    t_cal <- system.time(
      assignTaxonomy(cal$query_seqs, cal$ref_path, multithread = 1L, minBoot = 50)
    )[["elapsed"]]
    if (t_cal <= 0) stop("Calibration run failed.")
    n_queries <- max(50L, as.integer(50 * cfg$target_secs / t_cal))
    cat(sprintf("  50 queries took %.2f s -> using %d queries\n\n", t_cal, n_queries))
    unlink(cal$ref_path)
  }
  built <- build_stress_ref(ref_path, cfg$genera, cfg$refs_per_genus, n_queries)
  ref_path <- built$ref_path
  query_seqs <- built$query_seqs
  on.exit(unlink(ref_path), add = TRUE)
} else {
  query_seqs <- rep(long_seqs, length.out = cfg$queries)
}

run_bench <- function(label, multithread) {
  gc()
  t <- system.time({
    assignTaxonomy(query_seqs, ref_path, multithread = multithread, minBoot = 50)
  })
  cat(sprintf("%s (n=%d, multithread=%s): %.3f sec\n",
              label, length(query_seqs), multithread, t[["elapsed"]]))
  invisible(t[["elapsed"]])
}

cat("assignTaxonomy benchmark", if (cfg$stress) "(stress)", "\n")
if (cfg$stress) {
  cat("  genera:", cfg$genera, "\n")
  cat("  refs/genus:", cfg$refs_per_genus, "\n")
}
cat("  queries:", length(query_seqs), "\n")
cat("  reference:", ref_path, "\n\n")

run_bench("single-thread", 1L)
run_bench("multi-thread", TRUE)
