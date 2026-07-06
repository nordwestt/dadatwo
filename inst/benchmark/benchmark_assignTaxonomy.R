#!/usr/bin/env Rscript
# Benchmark assignTaxonomy wall time.
# Usage:
#   Rscript inst/benchmark/benchmark_assignTaxonomy.R [n_replicate] [stress]
# stress: if "stress", builds a ~5000-genus synthetic training set.

args <- commandArgs(trailingOnly = TRUE)
n_replicate <- if (length(args) >= 1) as.integer(args[[1]]) else 200L
stress <- length(args) >= 2 && tolower(args[[2]]) == "stress"

if (!requireNamespace("dada2", quietly = TRUE)) {
  stop("Install dada2 before running this benchmark.")
}
library(dada2)

ref_path <- system.file("extdata", "example_seqs.fa", package = "dada2")
if (!nzchar(ref_path)) stop("example_seqs.fa not found in dada2 package")

base_seqs <- getSequences(ref_path)
long_seqs <- base_seqs[nchar(base_seqs) >= 50]
if (length(long_seqs) == 0) stop("No sequences >= 50 nts in example data")

if (stress) {
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
  ngenus_target <- 5000L
  train_path <- tempfile(fileext = ".fa")
  con <- file(train_path, "w")
  for (g in seq_len(ngenus_target)) {
    idx <- ((g - 1L) %% length(sq)) + 1L
    parts <- strsplit(tax[idx], ";", fixed = TRUE)[[1]]
    parts[6] <- paste0("Genus_", g)
    tstr <- paste0(paste(parts, collapse = ";"), ";")
    for (r in seq_len(3L)) {
      writeLines(c(paste0(">", tstr), sq[idx]), con)
    }
  }
  close(con)
  ref_path <- train_path
  query_seqs <- rep(sq, length.out = 100L)
  on.exit(unlink(train_path), add = TRUE)
} else {
  query_seqs <- rep(long_seqs, length.out = n_replicate)
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

cat("assignTaxonomy benchmark", if (stress) "(stress mode)", "\n")
cat("  queries:", length(query_seqs), "\n")
cat("  reference:", ref_path, "\n\n")

run_bench("single-thread", 1L)
run_bench("multi-thread", TRUE)
