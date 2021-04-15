##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#### Start with just 1000 rows for now
edx <- edx[1:1000,]

### Get the unique Genres and make a vector
genres<- unlist(unique(simplify(strsplit(edx$genres, split="\\|"))))

genreRatingsFunction <- function(g) {
  genre_match <- str_detect(edx$genres, g)
  edx[genre_match] %>% select(rating) %>% mutate(genre=g)
}

genreRatings <- lapply(genres, function(x) genreRatingsFunction(x)) %>% bind_rows()

### Which genre has the highest avg rating?
genreRatings %>% group_by(genre) %>% summarise(avg_rating = mean(rating)) %>% ggplot(aes(genre, avg_rating)) + geom_bar(stat="identity")

### Film Noir, what has the most ratings?
genreRatings %>% group_by(genre) %>% summarise(num_ratings =n()) %>% ggplot(aes(genre, num_ratings)) + geom_bar(stat="identity")

### Comparatively, Film Noir has very few ratings, but other ratings seem to make sense. Documentaries are ranked slightly higher than avg, action and comedy slightly lower
mu <- mean(edx$rating)
genre_bi <- genreRatings %>% group_by(genre) %>% summarise(b_i = mean(rating - mu)) %>% data.frame()


### calculate the Genre effect for a given rating
calcGenreEffect <- function(g) {
  match <- str_detect(edx$genres, g)
  b_i <- genre_bi%>% filter(g == genre) %>% select(b_i)
  sapply(match, function(x) as.numeric(x) * as.numeric(b_i))
}

genre_df <- as.data.frame(sapply(genres, function(x) calcGenreEffect(x)))

edx <- cbind(edx, genre_df)


