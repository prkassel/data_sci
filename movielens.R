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

###############################################
# End Create edx set
###############################################



edx <- edx[1:100000,]

ind <- createDataPartition(edx$rating, times = 1, p=0.1, list=FALSE)

train_set <- edx[-ind]
test_set <- edx[ind]

### we want to make sure that the test set only has movies and users that are in
### the train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


lambda = 3

mu <- mean(train_set$rating)

movies_df <- train_set %>% group_by(movieId) %>% summarise(movie_bias=sum(rating - mu)/(n() + lambda )) %>% select(movieId, movie_bias)
users_df <- train_set %>% group_by(userId) %>% summarise(user_bias=sum(rating - mu)/(n() + lambda + 20)) %>% select(userId, user_bias)


genre_ratings_df <- train_set %>% separate_rows(genres, sep="\\|")


user_genre_biases <- genre_ratings_df %>% group_by(userId, genres) %>% summarise(user_genre_bias = sum(rating - mu)/(n() + lambda))

user_genre_ratings <- genre_ratings_df %>% inner_join(user_genre_biases, on=c("userId", "genres"))
user_genre_ratings <- user_genre_ratings %>% select(-rating,-timestamp,-title) %>% pivot_wider(names_from=genres, values_from="user_genre_bias")


train_set <- train_set %>% inner_join(user_genre_ratings, on=c("userId", "movieId"))
train_set <- train_set %>% inner_join(movies_df, on="movieId")
train_set <- train_set %>% inner_join(users_df, on="movieId")
train_set <- train_set %>% select(-title, -genres, -timestamp)

train_set$user_genre_bias <- apply(X=train_set[,4:(ncol(train_set) - 2)], MARGIN=1, FUN=mean, na.rm=TRUE)
train_set$max_bias <- apply(X=train_set[,(ncol(train_set) - 3): ncol(train_set)], MARGIN=1, FUN=max, na.rm=TRUE)
train_set$min_bias <- apply(X=train_set[,(ncol(train_set) - 4): (ncol(train_set) - 1)], MARGIN=1, FUN=min, na.rm=TRUE)

#### the idea here is that the biases compete with each other. If a movie is rated 
### exceptionally high, but the 

train_set$net_bias <- train_set$max_bias + train_set$min_bias


train_set$pred<- train_set$net_bias + mu


RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

print(RMSE(train_set$rating, train_set$pred))


### Preparing Test Data
test_set <- test_set %>% select(-timestamp, -title)
test_set_ratings_by_genre <- test_set %>% separate_rows(genres, sep="\\|")
test_set_ratings_by_genre <- test_set_ratings_by_genre %>% inner_join(user_genre_biases, on=c("userId", "genres"))
test_set_user_genre_biases <-  test_set_ratings_by_genre %>% select(-rating) %>% pivot_wider(names_from=genres, values_from="user_genre_bias")

test_set <- test_set %>% inner_join(test_set_user_genre_biases, on=c("userId", "movieId"))
test_set <- test_set %>% inner_join(movies_df, on="movieId")
test_set <- test_set %>% inner_join(users_df, on="movieId")
test_set <- test_set %>% select(-genres)

test_set$user_genre_bias <- apply(X=test_set[,4:(ncol(test_set) - 2)], MARGIN=1, FUN=mean, na.rm=TRUE)
test_set$max_bias <- apply(X=test_set[,(ncol(test_set) - 3): ncol(test_set)], MARGIN=1, FUN=max, na.rm=TRUE)
test_set$min_bias <- apply(X=test_set[,(ncol(test_set) - 4): (ncol(test_set) - 1)], MARGIN=1, FUN=min, na.rm=TRUE)
test_set$net_bias <- test_set$max_bias + test_set$min_bias


test_set$pred <- test_set$net_bias + mu

print(RMSE(test_set$rating, test_set$pred))


test_set %>% mutate(residual = rating - pred) %>% filter(residual %between% c(-3,-1)) %>% select(movieId, userId, residual) %>% slice(1:10)

#### TODO create a function to prepare both the edx training set and the validation set with genre,movie,user biases