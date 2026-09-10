library(httr2)
library(jsonlite)

# ── Configuration ──────────────────────────────────────────────────────────────

# API endpoint — change to your bibr server URL
BASE_URL <- Sys.getenv("BIBR_API_URL", "http://localhost:8000")
API_KEY <- Sys.getenv("BIBR_API_KEY")
add_auth <- function(req) {
  if (nzchar(API_KEY)) req_auth_bearer_token(req, API_KEY) else req
}

# Path to the file you want to upload
args <- commandArgs(trailingOnly = TRUE)
FILE_PATH <- if (length(args)) args[1] else Sys.getenv("PAPER_PATH", "paper.pdf")
if (!file.exists(FILE_PATH)) stop("File not found: ", FILE_PATH)

# ── Send request ───────────────────────────────────────────────────────────────

cat("Sending", basename(FILE_PATH), "to", BASE_URL, "...\n")

resp <- request(BASE_URL) |>
  req_url_path_append("papers", "extract") |>
  add_auth() |>
  req_body_multipart(
    file = curl::form_file(FILE_PATH)
  ) |>
  req_timeout(300) |>
  req_perform()

cat("Status:", resp_status(resp), "\n")

if (resp_status(resp) != 200) {
  stop("Request failed: ", resp_body_string(resp))
}

# ── Parse the JSON response ──────────────────────────────────────────────────

json_string <- resp_body_string(resp)
data <- fromJSON(json_string, simplifyVector = TRUE, flatten = TRUE)

# Keep nullable and nested paper metadata as a named list.
info_df     <- data$info

to_df <- function(x) {
  if (is.null(x) || length(x) == 0) return(data.frame())
  as.data.frame(x, stringsAsFactors = FALSE)
}

text_df     <- to_df(data$text)
section_df  <- to_df(data$section)
author_df   <- to_df(data$author)
bib_df      <- to_df(data$bib)
url_df      <- to_df(data$url)
xref_df     <- to_df(data$xref)
figure_df   <- to_df(data$figure)
table_df    <- to_df(data$table)
eq_df       <- to_df(data$eq)

# ── Print summary ──────────────────────────────────────────────────────────────

cat("\n=== PAPER INFORMATION ===\n")
cat("Paper ID:", data$paper_id, "\n")
cat("Title:   ", info_df$title[1], "\n")
cat("DOI:     ", info_df$doi[1], "\n")
cat("Format:  ", info_df$input_format[1], "\n")
cat("File:    ", info_df$file_name[1], "\n")
cat("Version: ", info_df$bibr_version[1], "\n\n")

cat("=== CONTENT SUMMARY ===\n")
cat("Sentences:  ", nrow(text_df), "\n")
cat("Sections:   ", nrow(section_df), "\n")
cat("Authors:    ", nrow(author_df), "\n")
cat("References: ", nrow(bib_df), "\n")
cat("URLs:       ", nrow(url_df), "\n")
cat("Tables:     ", nrow(table_df), "\n")
cat("Xrefs:      ", nrow(xref_df), "\n")
cat("Figures:    ", nrow(figure_df), "\n")
cat("Equations:  ", nrow(eq_df), "\n")

# Show authors
if (nrow(author_df) > 0) {
  cat("=== AUTHORS ===\n")
  for (i in seq_len(min(nrow(author_df), 10))) {
    a <- author_df[i, ]
    cat(sprintf("  %s %s\n", a$given, a$family))
  }
  cat("\n")
}

# Show first few sentences
if (nrow(text_df) > 0) {
  cat("=== SENTENCES (First 3) ===\n")
  for (i in seq_len(min(nrow(text_df), 3))) {
    cat(sprintf("  [%d] %s\n", text_df$text_id[i], text_df$text[i]))
  }
  cat("\n")
}

cat("Successfully retrieved and parsed metadata.\n")
