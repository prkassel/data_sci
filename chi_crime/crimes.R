library(tidyverse)

if (!file.exists("chi_crime/crimes.csv")) {
  url <- "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD&api_foundry=true"
  download.file(url, "crimes.csv")
}

dat <- read_csv("chi_crime/crimes.csv")
