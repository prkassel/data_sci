##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

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
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

edx <- edx[1:1000000,]
edx$release_year <- str_sub(edx$title,start= -6)
edx$release_year <- as.numeric(str_extract(edx$release_year, "\\d+"))
edx$rating_year <- year(as.Date(as.POSIXct(edx$timestamp,origin="1970-01-01")))
# Round to the nearest 10 year increment
edx$years_from_release <- round((edx$rating_year - edx$release_year) / 10) * 10

ind <- createDataPartition(edx$rating, times = 1, p=0.2, list=FALSE)

train_set <- edx[-ind]
test_set <- edx[ind]

### we want to make sure that the test set only has movies and users that are in
### the train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "years_from_release")


mu <- mean(train_set$rating)
lambdas <- seq(0,10,0.1)

tune_lambdas <- function(grouping) {

  if (grouping == "genre") {
    biases <- genre_ratings_df %>% group_by(userId, genres)
  }

  else {
    biases <- train_set %>% group_by(!!as.symbol(grouping))
  }
  rate_lambda <- function(lambda) {
    if (grouping == "genre") {
      temp <- biases %>% summarise(genre_bias = sum(rating - movie_bias - user_bias - mu)/(n() + lambda))
      temp <- temp %>% inner_join(test_genre_ratings_df, on=genres)
      predictions <- temp %>% 
        left_join(movies_df, on="movieId") %>% left_join(users_df, on="userId") %>%
        mutate(pred=mu + genre_bias + movie_bias + user_bias)
    }
    else if (grouping == "userId") {
      temp <- biases %>% summarise(user_bias = mean(rating - movie_bias - mu))
      predictions <- test_set %>% inner_join(temp, on=!!as.symbol(grouping)) %>% 
        left_join(movies_df, on=movieId) %>% mutate(pred = mu + movie_bias + user_bias) %>% select(rating, pred)
    }
    else if (grouping == "years_from_release") {
      ug <- user_genres_df %>% select(-movieId) %>% group_by(userId, genres) %>% summarise(genre_bias = mean(genre_bias))
      tg <- test_genre_ratings_df %>% inner_join(ug, on=genres) %>% group_by(userId, movieId) %>% 
        summarise(user_genre_bias = mean(genre_bias))
      temp <- biases %>% summarise(bias=sum(rating - mu - movie_bias - user_bias - user_genre_bias) / (n() + lambda))

      predictions <- test_set %>% left_join(temp, on=years_from_release) %>% 
        left_join(movies_df, on=movieId) %>% left_join(users_df, on=userId) %>%
        left_join(tg, on=c(userId, movieId))
      predictions[is.na(predictions)] = 0
      predictions <- predictions %>% mutate(pred = mu + movie_bias + user_bias + user_genre_bias + bias)
    }
    else {
    temp <- biases %>% summarise(bias=sum(rating - mu) / (n() + lambda))
    predictions <- test_set %>% inner_join(temp, on=!!as.symbol(grouping)) %>% mutate(pred = mu + bias) %>% select(rating, pred)
    }
  RMSE(predictions$rating, predictions$pred)
  }
  sapply(lambdas, FUN = function(x) rate_lambda(x))
}

#### Calculating movie effect
movies_lambda <- tune_lambdas("movieId")
movies_df <- train_set %>% group_by(movieId) %>% summarise(movie_bias=sum(rating - mu)/(n() + lambdas[which.min(movies_lambda)] )) %>% select(movieId, movie_bias)
train_set <- train_set %>% inner_join(movies_df, on="movieId")


### Calculating User Effect
users_lambda <- tune_lambdas("userId")
users_df <- train_set %>% group_by(userId) %>% summarise(user_bias=sum(rating - movie_bias - mu)/(n() + lambdas[which.min(users_lambda)])) %>% select(userId, user_bias)
train_set <- train_set %>% inner_join(users_df, on="userId")

### Calculate User Genre Effect
genre_ratings_df <- train_set %>% separate_rows(genres, sep="\\|")
test_genre_ratings_df <- test_set %>% separate_rows(genres, sep="\\|")
genres_lambda <- tune_lambdas("genre")
user_genres_df <- genre_ratings_df %>% group_by(userId, genres) %>% 
  summarise(genre_bias = sum(rating - movie_bias - user_bias - mu)/(n() + lambdas[which.min(genres_lambda)])) %>% 
  inner_join(genre_ratings_df, on=genres) %>% select(userId, genres, genre_bias, movieId)

user_genres_df_wide <- user_genres_df %>% pivot_wider(names_from = genres, values_from=genre_bias)
train_set <- train_set %>% inner_join(user_genres_df_wide, on=c("userId", "movieId"))
train_set$user_genre_bias <- apply(X=train_set[,12:ncol(train_set)], MARGIN=1, FUN=mean, na.rm=TRUE)


train_set <- train_set %>% select(-title, -genres, -timestamp, -release_year, -rating_year)
train_set$user_genre_bias <- apply(X=train_set[,6:ncol(train_set)], MARGIN=1, FUN=mean, na.rm=TRUE)

#### Time Effect
years_since_release_lambda <- tune_lambdas("years_from_release")
recency_df <- train_set %>% group_by(years_from_release) %>% 
  summarise(recency_bias=sum(rating - movie_bias - user_bias - user_genre_bias - mu)/(n() + lambdas[which.min(years_since_release_lambda)])) %>% select(years_from_release, recency_bias)

train_set <- train_set %>% inner_join(recency_df, on="years_from_release")

train_set <- train_set %>% select(userId, movieId, rating, movie_bias, user_bias, user_genre_bias, recency_bias)

train_set$pred<- train_set$movie_bias + train_set$user_bias + train_set$user_genre_bias + train_set$recency_bias + mu


print(RMSE(train_set$rating, train_set$pred))


### Preparing Test Data 
test_set <- test_set %>% left_join(movies_df, on="movieId") %>% 
  left_join(users_df, on="userId") %>% left_join(recency_df, on="years_from_release")
test_user_genres_df <- user_genres_df %>% select(-movieId) %>% group_by(userId, genres) %>% summarise(genre_bias = mean(genre_bias))
test_genre_biases <- test_genre_ratings_df %>% left_join(test_user_genres_df, on=(c("userId", "genres"))) %>% select(userId, movieId, genres, genre_bias)
test_genre_biases <- test_genre_biases %>% pivot_wider(names_from = genres, values_from=genre_bias)
test_set <-test_set %>% left_join(test_genre_biases, on=c("userId", "movieId"))

test_set$user_genre_bias <- apply(X=test_set[,12:ncol(test_set)], MARGIN=1, FUN=mean, na.rm=TRUE)
test_set[is.na(test_set)] = 0

test_set <- test_set %>% select(userId, movieId, rating, movie_bias, user_bias, user_genre_bias, recency_bias)

test_set$pred<- test_set$movie_bias + test_set$user_bias + test_set$user_genre_bias + test_set$recency_bias + mu

print(RMSE(test_set$rating, test_set$pred))

test_set %>% filter(abs(rating - pred) > 1) %>% slice(1:10)

