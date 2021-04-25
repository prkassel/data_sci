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
library(lubridate)

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

###############################################
# End Create edx set
###############################################

edx$release_year <- str_sub(edx$title,start= -6)
edx$release_year <- as.numeric(str_extract(edx$release_year, "\\d+"))
edx$rating_year <- year(as.Date(as.POSIXct(edx$timestamp,origin="1970-01-01")))
# Round to the nearest 10 year increment
edx$years_since_release <- round((edx$rating_year - edx$release_year) / 10) * 10

validation$release_year <- str_sub(validation$title,start= -6)
validation$release_year <- as.numeric(str_extract(validation$release_year, "\\d+"))
validation$rating_year <- year(as.Date(as.POSIXct(validation$timestamp,origin="1970-01-01")))
# Round to the nearest 10 year increment
validation$years_since_release <- round((validation$rating_year - validation$release_year) / 10) * 10

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

###############################################
# TRAIN MODEL
###############################################

mu <- mean(edx$rating)

#### Calculating movie effect
movies_df <- edx %>% group_by(movieId) %>% summarise(movie_bias=sum(rating - mu)/(n() + 2.5 )) %>% select(movieId, movie_bias)
edx <- edx %>% inner_join(movies_df, on="movieId")


### Calculating User Effect
users_df <- edx %>% group_by(userId) %>% summarise(user_bias=sum(rating - movie_bias - mu)/(n())) %>% select(userId, user_bias)
edx <- edx %>% inner_join(users_df, on="userId")

### Calculate User Genre Effect
genre_ratings_df <- edx %>% separate_rows(genres, sep="\\|")
user_genres_df <- genre_ratings_df %>% group_by(userId, genres) %>% 
  summarise(genre_bias = sum(rating - movie_bias - user_bias - mu)/(n() + 10)) %>% 
  inner_join(genre_ratings_df, on=genres) %>% select(userId, genres, genre_bias, movieId)

user_genres_df_wide <- user_genres_df %>% pivot_wider(names_from = genres, values_from=genre_bias)
edx <- edx %>% inner_join(user_genres_df_wide, on=c("userId", "movieId"))
edx$user_genre_bias <- apply(X=edx[,12:ncol(edx)], MARGIN=1, FUN=mean, na.rm=TRUE)


edx <- edx %>% select(-title, -genres, -timestamp, -release_year, -rating_year)
edx$user_genre_bias <- apply(X=edx[,6:ncol(edx)], MARGIN=1, FUN=mean, na.rm=TRUE)

#### Time Effect
recency_df <- edx %>% group_by(years_since_release) %>% 
  summarise(recency_bias=sum(rating - movie_bias - user_bias - user_genre_bias - mu)/(n() + 10)) %>% select(years_since_release, recency_bias)

#############
# Validation Test
############
validation_genre_ratings_df <- validation %>% separate_rows(genres, sep="\\|")

validation <- validation %>% left_join(movies_df, on="movieId") %>% 
  left_join(users_df, on="userId") %>% left_join(recency_df, on="years_since_release")
validation_user_genres_df <- user_genres_df %>% select(-movieId) %>% group_by(userId, genres) %>% summarise(genre_bias = mean(genre_bias))
validation_genre_biases <- validation_genre_ratings_df %>% left_join(validation_user_genres_df, on=(c("userId", "genres"))) %>% select(userId, movieId, genres, genre_bias)
validation_genre_biases <- validation_genre_biases %>% pivot_wider(names_from = genres, values_from=genre_bias)
validation <-validation %>% left_join(validation_genre_biases, on=c("userId", "movieId"))

validation$user_genre_bias <- apply(X=validation[,12:ncol(validation)], MARGIN=1, FUN=mean, na.rm=TRUE)
validation$user_genre_bias[is.na(validation$user_genre_bias)] = 0

validation <- validation %>% select(userId, movieId, rating, movie_bias, user_bias, user_genre_bias, recency_bias)

validation[is.na(validation)] = 0

validation$pred<- validation$movie_bias + validation$user_bias + validation$user_genre_bias + validation$recency_bias + mu

print(RMSE(validation$rating, validation$pred))