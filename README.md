
[![Build Status](https://github.com/benjjneb/dada2/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/benjjneb/dada2/actions/workflows/pages/pages-build-deployment)

# dada2

Exact sample inference from high-throughput amplicon data. Resolves real variants differing by as little as one nucleotide. Visit [the DADA2 website](https://benjjneb.github.io/dada2/index.html) for the most detailed and up-to-date documentation.

### Installation

The dada2 package binaries are available through Bioconductor:

```S
## try http:// if https:// URLs are not supported
if (!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager")
BiocManager::install("dada2")
```

In order to install dada2 from source (and get the latest and greatest new features) see our [installation from source instructions](https://benjjneb.github.io/dada2/dada-installation.html).

### Documentation

The [tutorial walkthrough of the DADA2 pipeline on paired end Illumina Miseq data](https://benjjneb.github.io/dada2/tutorial.html). 

The [dada2 R package manual](https://www.bioconductor.org/packages/3.6/bioc/manuals/dada2/man/dada2.pdf).

Further documentation is available on [the DADA2 front page](http://benjjneb.github.io/dada2/). 

### Efficient taxonomy assignment (`assignTaxonomy`)

Large projects (many ASVs × full Silva/RDP training databases) can use tens of gigabytes of RAM and run for minutes because the naive Bayesian classifier builds a large kmer probability matrix. The same `assignTaxonomy()` call as before is now faster and more memory-friendly:

- **Automatic reference caching** — for a training fasta path on disk, the classifier matrix is built once and stored under `tools::R_user_dir("dada2", which="cache")/taxonomy-ref`. Later calls with the same file reuse the cache (set `verbose=TRUE` to see cache hits).
- **Automatic batching** — query sets larger than 5000 sequences are processed in chunks to cap peak memory.
- **Memory-mapped classifier storage** on Linux/macOS so the cached matrix is not fully duplicated in RAM.

#### Example (unchanged API)

```r
library(dada2)

asv <- getSequences(seqtab)   # character vector of ASV sequences
taxa <- assignTaxonomy(asv, "silva_nr99_v138.1_train_set.fa.gz", multithread = TRUE)

# Second call with the same training file is faster (uses disk cache)
taxa2 <- assignTaxonomy(other_asvs, "silva_nr99_v138.1_train_set.fa.gz",
                        multithread = TRUE, verbose = TRUE)
```

#### Memory and runtime tips

| Tip | Why |
|-----|-----|
| Reuse the same training fasta path | Enables automatic disk cache across runs and R sessions |
| Use genus-level training fastas | Fewer unique taxonomy strings → much smaller matrix (see [training sets](https://benjjneb.github.io/dada2/training.html)) |
| Keep `tryRC = FALSE` unless needed | Skips reverse-complement extraction |
| Run heavy jobs outside RStudio | Plain `Rscript` avoids IDE memory overhead and crashes |
| Call `gc()` after taxonomy | Release reference strings before downstream steps |

Approximate classifier matrix size: **unique taxonomy strings × 0.25 MB**. Example: 80,000 taxonomies ≈ 20 GB on disk (memory-mapped at assignment time).

#### Benchmarking

```bash
# Quick stress test (~default 100 queries, 5000 genera)
Rscript inst/benchmark/benchmark_assignTaxonomy.R --queries 100 --genera 5000

# Longer run (~2 minutes; auto-calibrates query count)
Rscript inst/benchmark/benchmark_assignTaxonomy.R --target-secs 120 --genera 5000

# Heavier load: more queries and/or genera
Rscript inst/benchmark/benchmark_assignTaxonomy.R --queries 1200 --genera 5000
```

Runtime scales roughly linearly with `--queries` and `--genera`.

### DADA2 Articles

[DADA2: High resolution sample inference from Illumina amplicon data. Nature Methods, 2016.](http://dx.doi.org/10.1038/nmeth.3869) [(Open Access link.)](http://rdcu.be/ipGh)

[Bioconductor workflow for microbiome data analysis: from raw reads to community analyses. F1000 Research, 2016.](https://f1000research.com/articles/5-1492)

[Exact sequence variants should replace operational taxonomic units in marker-gene data analysis. ISMEJ, 2017.](http://dx.doi.org/10.1038/ismej.2017.119)

[High-throughput amplicon sequencing of the full-length 16S rRNA gene with single-nucleotide resolution. Nucleic Acids Research, 2019.](http://dx.doi.org/10.1093/nar/gkz569)

### Other Resources

Planned feature improvements are publicly catalogued at the main DADA2 development site on github, specifically on the "Issues" page for DADA2:

https://github.com/benjjneb/dada2/issues

If the feature you are hoping for is not listed, you are welcome to add it as a feature request "issue" on this page. This request will be publicly available and listed on the page.

Bugs and difficulties in using DADA2 are also welcome on [the issue tracker](https://github.com/benjjneb/dada2/issues).
