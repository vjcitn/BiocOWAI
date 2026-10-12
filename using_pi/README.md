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
## Summary

  This file defines two functions for robust statistical scaling:

  1. **`mad_na_robust(x, na.rm = TRUE, constant = 1.4826)`** — Computes the Median Absolute Deviation (MAD). It calculates `median(|X_i - median(X)|)` and optionally multiplies by a consistency constant (default `1.4826`, which makes MAD consistent with σ under normality).

  2. **`robust_scale(x, na.rm = TRUE)`** — Centers a vector by its median and scales by the MAD (with `constant = 1`), producing a vector with median 0 and MAD 1. Missing values are preserved as `NA`.
  ---
 ## Edge Cases It Misses                                                                                                                      
 | Edge Case | Problem |
 |---|---|
 | **MAD = 0** (all values identical, or exactly half are identical around the median) | Division by zero in `robust_scale`, returning `Inf`/`NaN` with no warning. |
 | **`length(x) == 1`** | `mad_na_robust` returns `0` (MAD of a single value is 0), so `robust_scale` divides by zero. |
 | **Non-numeric input** (character, logical, factor) | No type checking; `median()` and arithmetic will produce errors or silent coercion. |
 | **`NaN` values** | `is.na(NaN)` is `TRUE`, so `NaN` is stripped when `na.rm = TRUE`, but `NaN` propagates through arithmetic in the `na.rm = FALSE` path, producing `NaN` results silently. |
 | **`Inf` / `-Inf` values** | `median()` handles them, but MAD can become `Inf`, again causing division-by-`Inf` (→ 0) or `NaN` silently. |
 | **Empty string / zero-length input** | Returns `NA_real_` from `mad_na_robust`, and `NaN` from `robust_scale` (0/0), with no explicit handling. |
 | **Multi-dimensional input** (matrix/array) | No check — `median()` and arithmetic will behave unexpectedly (e.g., operating on the flattened vector). |
 | **Very large vectors** | No memory or performance guardrails, though this is minor. |
 
 The most practically important missing case is **MAD = 0** (constant data), which would be best handled with a guard that returns `0` (or `NA`) with a warning instead of silently producing `NaN`.

Approximate resource usage for the work noted here:
```
rstudio@6adc68012135:~$ t4-coder status
[t4-coder] Running, pid 8026; health: {"status":"ok"}
12435 MiB, 15360 MiB, 0 %
10.23.781.802 I slot print_timing: id  0 | task 3696 | prompt eval time =    3795.16 ms /   572 tokens (    6.63 ms per token,   150.72 tokens per second)
10.23.781.816 I slot print_timing: id  0 | task 3696 |        eval time =   19787.96 ms /   566 tokens (   35.02 ms per token,    28.55 tokens per second)
10.23.781.818 I slot print_timing: id  0 | task 3696 |       total time =   23583.12 ms /  1138 tokens
10.23.781.820 I slot print_timing: id  0 | task 3696 |    graphs reused =       4202
10.23.782.176 I slot      release: id  0 | task 3696 | stop processing: n_tokens = 2823, truncated = 0
```
