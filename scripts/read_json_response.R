# R Helper for Reading bibr JSON Format (v12; v11 and legacy v10 read too)
#
# The main record tables and paper metadata are:
#   paper_id, metadata, source, author, affiliation, funding, text, section, url, bib, xref,
#   figure, table, eq, metadata_match, affiliation_match, funding_match, bib_match
#
# Each listed key except paper_id, metadata, and source contains an array of objects.
# The "metadata" key ("info" in legacy exports) is a single object with paper-level metadata. Other top-level
# fields (including validation and provenance) are preserved.

library(jsonlite)

#' Read a bibr JSON file into a structured list
#'
#' @param json_path Path to a bibr JSON file on disk
#' @return A named list with paper_id, metadata (named list),
#'   and data.frames for author, text, section, url, bib, bib_match, xref, figure, table, eq
#' @export
read_bibr_json <- function(json_path) {
  raw <- fromJSON(json_path, simplifyVector = TRUE, flatten = TRUE)
  .structure_bibr(raw)
}

#' Parse a bibr JSON string into a structured list
#'
#' @param json_string A JSON character string (e.g. from an API response body)
#' @return A named list (same structure as read_bibr_json)
#' @export
parse_bibr_json <- function(json_string) {
  raw <- fromJSON(json_string, simplifyVector = TRUE, flatten = TRUE)
  .structure_bibr(raw)
}

#' Parse raw bytes from an API response into a structured list
#'
#' @param json_bytes Raw bytes from an API response (will be converted to string)
#' @return A named list (same structure as read_bibr_json)
#' @export
read_bibr_response <- function(json_bytes) {
  json_string <- rawToChar(json_bytes)
  parse_bibr_json(json_string)
}

# Internal: convert the parsed JSON list into a clean structure with data.frames
.structure_bibr <- function(raw) {
  # Keep metadata and source as lists: null scalars and nested metadata are valid in exports.
  # Preserve other fields, including provenance and validation receipts.
  table_names <- c("author", "affiliation", "funding", "text", "section", "url", "bib",
                   "xref", "figure", "table", "eq", "metadata_match", "affiliation_match",
                   "funding_match", "bib_match")
  for (name in table_names) {
    value <- raw[[name]]
    raw[[name]] <- if (is.null(value) || length(value) == 0) {
      data.frame()
    } else {
      as.data.frame(value, stringsAsFactors = FALSE)
    }
  }
  raw
}

# Example usage:
#
# # From API response (using httr2)
# resp <- request("http://localhost:8000") |>
#   req_url_path_append("papers", "extract") |>
#   req_body_multipart(file = curl::form_file("paper.pdf")) |>
#   req_timeout(300) |>
#   req_perform()
# data <- read_bibr_response(resp_body_raw(resp))
#
# # From a saved JSON file
# data <- read_bibr_json("output.json")
#
# # Access the data
# data$paper_id           # character: paper identifier
# data$metadata$title         # paper title
# data$metadata$doi           # DOI
# data$extraction$producer$version  # producing package version
# data$author             # data.frame of authors
# data$text               # data.frame of sentences
# data$section            # data.frame of sections
# data$url                # data.frame of URL links
# data$bib                # data.frame of references
# data$xref               # data.frame of cross-references
# data$figure             # data.frame of figures
# data$table              # data.frame of tables
# data$eq                 # data.frame of equations
#
# # Print summary
# cat("Paper ID:", data$paper_id, "\n")
# cat("Title:   ", data$metadata$title[1], "\n")
# cat("DOI:     ", data$metadata$doi[1], "\n")
# cat("Sentences:", nrow(data$text), "\n")
# cat("Authors:  ", nrow(data$author), "\n")
# cat("Refs:     ", nrow(data$bib), "\n")
