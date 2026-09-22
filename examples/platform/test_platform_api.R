# Test script for the Scienceverse platform API (submit -> poll -> download).
#
# Mirrors the metacheck sv_convert() function for debugging the async job queue.
#
# Usage:
#   Rscript examples/platform/test_platform_api.R <file.pdf>
#   PLATFORM_API_KEY=sv_... Rscript examples/platform/test_platform_api.R paper.pdf

library(httr2)
library(jsonlite)

`%||%` <- function(x, fallback) if (is.null(x)) fallback else x

# ── Configuration ────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: Rscript test_platform_api.R <file_path> [api_url]")
}

FILE_PATH  <- args[1]
API_URL    <- if (length(args) >= 2) args[2] else "https://platform.metacheck.app"
API_KEY    <- Sys.getenv("PLATFORM_API_KEY")
POLL_INTERVAL <- 2
TIMEOUT    <- 600

if (nchar(API_KEY) == 0) {
  stop("Set the PLATFORM_API_KEY environment variable")
}
if (!file.exists(FILE_PATH)) {
  stop("File not found: ", FILE_PATH)
}

cat("Platform:", API_URL, "\n")
cat("File:    ", basename(FILE_PATH), "\n\n")

# ── 1. Health check ─────────────────────────────────────────────────────────

tryCatch({
  health <- request(API_URL) |>
    req_url_path_append("health") |>
    req_auth_bearer_token(API_KEY) |>
    req_perform() |>
    resp_body_json()
  cat("Health:  ", health$status, "\n")
}, error = \(e) cat("Health check failed:", e$message, "\n"))

tryCatch({
  ready <- request(API_URL) |>
    req_url_path_append("ready") |>
    req_auth_bearer_token(API_KEY) |>
    req_perform() |>
    resp_body_json()
  cat("Ready:   ", ready$status, "\n")
}, error = \(e) cat("Ready check failed:", e$message, "\n"))

# ── 2. Submit job ────────────────────────────────────────────────────────────

cat("\nSubmitting", basename(FILE_PATH), "...\n")

submit_resp <- request(API_URL) |>
  req_url_path_append("jobs") |>
  req_auth_bearer_token(API_KEY) |>
  req_body_multipart(file = curl::form_file(FILE_PATH)) |>
  req_timeout(60) |>
  req_perform()

if (resp_status(submit_resp) != 200) {
  stop("Submit failed (HTTP ", resp_status(submit_resp), "): ",
       resp_body_string(submit_resp))
}

job <- resp_body_json(submit_resp)
job_id <- job$job_id
cat("Job ID:  ", job_id, "\n")
cat("Status:  ", job$status, "\n")

# ── 3. Poll for completion ───────────────────────────────────────────────────

cat("\nPolling ...\n")
elapsed <- 0
last_stage <- ""

repeat {
  Sys.sleep(POLL_INTERVAL)
  elapsed <- elapsed + POLL_INTERVAL

  status_resp <- request(API_URL) |>
    req_url_path_append("jobs", job_id) |>
    req_auth_bearer_token(API_KEY) |>
    req_timeout(30) |>
    req_perform()

  status <- resp_body_json(status_resp)
  stage <- status$stage %||% ""

  if (stage != last_stage) {
    cat(sprintf("  [%5.1fs] %s%s\n",
                elapsed, status$status,
                if (nchar(stage) > 0) paste0(" (", stage, ")") else ""))
    last_stage <- stage
  }

  if (identical(status$status, "complete")) break

  if (identical(status$status, "failed")) {
    stop("Job failed: ", status$stage %||% "unknown error")
  }

  if (elapsed >= TIMEOUT) {
    stop("Timed out after ", TIMEOUT, "s (last: ", status$status, ")")
  }
}

# ── 4. Download result ───────────────────────────────────────────────────────

cat("\nDownloading result ...\n")

result_resp <- request(API_URL) |>
  req_url_path_append("jobs", job_id, "result") |>
  req_auth_bearer_token(API_KEY) |>
  req_timeout(120) |>
  req_perform()

if (resp_status(result_resp) != 200) {
  stop("Download failed (HTTP ", resp_status(result_resp), ")")
}

json_string <- resp_body_string(result_resp)

# Save JSON to disk
json_path <- gsub("\\.[^.]+$", ".json", basename(FILE_PATH))
writeLines(json_string, json_path)
cat("Saved:   ", json_path, " (", nchar(json_string), " chars)\n\n")

# ── 5. Quick peek at result ──────────────────────────────────────────────────

data <- fromJSON(json_string, simplifyVector = TRUE, flatten = TRUE)
metadata <- data$metadata %||% data$info

cat("bibr version:", data$extraction$producer$version %||% data$extraction$bibr_version %||%
  metadata$bibr_version, "\n")
cat("Paper ID:    ", data$paper_id, "\n\n")

# Helper for safe row count
nrow_safe <- function(x) {
  if (is.null(x) || length(x) == 0) return(0L)
  nrow(as.data.frame(x))
}

table_names <- c("author", "text", "section", "url", "bib", "xref", "figure", "table", "eq")
for (tbl_name in table_names) {
  n <- nrow_safe(data[[tbl_name]])
  cat(sprintf("  %-10s %d rows\n", paste0(tbl_name, ":"), n))
}

# Show title if available
if (!is.null(metadata$title)) {
  cat("\nTitle:      ", metadata$title, "\n")
}
if (!is.null(metadata$doi)) {
  cat("DOI:        ", metadata$doi, "\n")
}

cat("\nDone.\n")
