Claude proposed
```
pi_run <- function(prompt, dir = ".", tools = NULL) {
  args <- c("--print", "--no-session",
            "--provider", "llama-cpp", "--model", "qwen3.6-35b-a3b")
  if (!is.null(tools)) args <- c(args, "--tools", paste(tools, collapse = ","))
  old <- setwd(dir); on.exit(setwd(old))
  system2("pi", c(args, shQuote(prompt)), stdout = TRUE)
}
```
We can then perform the following in Rstudio console, once `t4-coder start` has run.
```
> pi_run("Create a function to compute the median absolute deviation of a vector in R, ignoring NA.
Write the function code in a new folder called '~/agent-test/madr' and call it robust_scale.R")
```
Response:
```
  "Done. The file is at `~/agent-test/madr/robust_scale.R` and contains two functions:"                                  
```
| Function | Purpose |
|---|---|                                                                                                         
| `mad_na_robust(x, na.rm = TRUE, constant = 1.4826)` | Computes the MAD of a numeric vector, ignoring `NA` values. Returns the raw MAD (with `constant = 1`) or the consistency-scaled version (default `1.4826`, matching base R). |
| `robust_scale(x, na.rm = TRUE)` | Centers data by the median and scales by the raw MAD, returning values with median 0 and MAD 1. Preserves `NA` positions. |
Usage:
```                                                                                                                     Quick usage:                                                                                                                                                                                                                                    
source("~/agent-test/madr/robust_scale.R")
x <- c(1, 2, 3, 4, 5, NA, 100)                                                                                          mad_na_robust(x)       # ~7.413 (scaled) or ~5 (raw with constant = 1)
robust_scale(x)        # robust z-scores, NA preserved
```
Continue:
```
pi_run("Summarize what R/robust_scale.R does and list any edge cases it misses",
       dir = "~/agent-test/madr", tools = c("read", "grep", "find", "ls"))
```
