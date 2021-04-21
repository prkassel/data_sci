##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(Rfast)) install.packages("Rfast", repos = "http://cran.us.r-project.org")

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

###############################################
# End Create edx set
###############################################

#### Start with just 10000 rows for now


# edx <- edx[1:10000,]
# cp <- edx

lambda = 3

### Average rating
mu <- mean(edx$rating)

movies_df <- edx %>% group_by(movieId) %>% summarise(movie_bias=sum(rating - mu)/(n() + lambda)) %>% select(movieId, movie_bias)


genres_df <- edx %>% separate_rows(genres, sep="\\|")

genre_biases <- genres_df %>% group_by(genres) %>% summarise(genre_bias = sum(rating - mu)/(n() + lambda))

user_genre_biases <- genres_df %>% group_by(userId, genres) %>% summarise(user_genre_bias = sum(rating - mu)/(n() + lambda))

user_genre_biases <- genres_df %>% inner_join(user_genre_biases, on=c("userId", "genres")) %>% inner_join(genre_biases, on="genres")

user_genre_biases <- user_genre_biases %>% select(-rating,-timestamp,-title) %>% pivot_wider(names_from=genres, values_from=c("user_genre_bias", "genre_bias"))

edx <- edx %>% inner_join(user_genre_biases, on=c("userId", "movieId"))
edx <- edx %>% inner_join(movies_df, on="movieId")

edx[is.na(edx)] <- 0


### Testing an ML algorithm
ind <- createDataPartition(edx$rating, times=1, p=0.2, list=FALSE)
test_set <- edx[ind]
train_set <- edx[-ind]

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

train_set <- train_set %>% select(-timestamp, -title, -genres)
test_set <- test_set %>% select(-timestamp, -title, -genres)

fit <- train(rating ~ ., method="glm", data=train_set)

predictions <- predict(fit, test_set)
predictions <- ifelse((round(predictions/.5)*.5) > 5, 5, round(predictions/.5)*.5)

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

glmRMSE <- RMSE(test_set$rating, predictions)