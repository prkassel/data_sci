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

###############################################
# TRAIN MODEL
###############################################

mu <- mean(edx$rating)

movies_df <- edx %>% group_by(movieId) %>% summarise(movie_bias=sum(rating - mu)/(n() + 2 )) %>% select(movieId, movie_bias)
users_df <- edx %>% group_by(userId) %>% summarise(user_bias=sum(rating - mu)/(n() + 6)) %>% select(userId, user_bias)


genre_ratings_df <- edx %>% separate_rows(genres, sep="\\|")

user_genre_biases <- genre_ratings_df %>% group_by(userId, genres) %>% summarise(user_genre_bias = sum(rating - mu)/(n() + 3.5))

user_genre_ratings <- genre_ratings_df %>% inner_join(user_genre_biases, on=c("userId", "genres"))
user_genre_ratings <- user_genre_ratings %>% select(-rating,-timestamp,-title) %>% pivot_wider(names_from=genres, values_from="user_genre_bias")

#############
# Validation Test
############

validation <- validation %>% select(-timestamp, -title)
validation_ratings_by_genre <- validation %>% separate_rows(genres, sep="\\|")
validation_ratings_by_genre <- validation_ratings_by_genre %>% inner_join(user_genre_biases, on=c("userId", "genres"))
validation_user_genre_biases <-  validation_ratings_by_genre %>% select(-rating) %>% pivot_wider(names_from=genres, values_from="user_genre_bias")

validation <- validation %>% inner_join(validation_user_genre_biases, on=c("userId", "movieId"))
validation <- validation %>% inner_join(movies_df, on="movieId")
validation <- validation %>% inner_join(users_df, on="movieId")
validation <- validation %>% select(-genres)

validation$user_genre_bias <- apply(X=validation[,4:(ncol(validation) - 2)], MARGIN=1, FUN=mean, na.rm=TRUE)
validation$max_bias <- apply(X=validation[,(ncol(validation) - 3): ncol(validation)], MARGIN=1, FUN=max, na.rm=TRUE)
validation$min_bias <- apply(X=validation[,(ncol(validation) - 4): (ncol(validation) - 1)], MARGIN=1, FUN=min, na.rm=TRUE)
validation$net_bias <- validation$max_bias + validation$min_bias


validation$pred <- validation$net_bias + mu

print(RMSE(validation$rating, validation$pred))