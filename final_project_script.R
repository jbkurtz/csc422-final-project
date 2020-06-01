#--------------------------
# Group P09
# Jack Kurtz, jbkurtz
# Chang Lee, cnlee2
# Schendong Chen, schen42
#--------------------------

# Install
# install.packages("dplyr")
# install.packages("glmnet")
# install.packages("monmlp")
# install.packages("e1071")
# install.packages("caret")

# Libraries
library(dplyr)
library(glmnet)
library(monmlp)
library(e1071)
library(caret)

#--------------------------
# Initial Pre-processing
#--------------------------

music <- read.csv("music.csv")

music <- music %>% filter(song.hotttnesss > 0) %>% mutate(popularity = song.hotttnesss > 0.685)  %>% select(popularity, everything())

#--------------------------
# Feature Selection
#--------------------------

# PCA - No useful information
# music.pca <- prcomp(music[,c(2:31)])

full <- lm(music$song.hotttnesss ~ music$artist.familiarity + music$artist.hotttnesss + music$artist.latitude + music$artist.location + music$artist.longitude + music$artist.similar + music$artist.terms_freq + music$release.id + music$release.name + music$song.artist_mbtags + music$song.artist_mbtags_count + music$song.bars_confidence + music$song.bars_start + music$song.beats_confidence + music$song.beats_start + music$song.duration + music$song.end_of_fade_in + music$song.key + music$song.key_confidence + music$song.loudness + music$song.mode + music$song.mode_confidence + music$song.start_of_fade_out + music$song.tatums_confidence + music$song.tatums_start + music$song.tempo + music$song.time_signature + music$song.time_signature_confidence + music$song.title + music$song.year)
null <- lm(music$song.hotttnesss ~ 1)

# Forward Selection
# forward_selection <- step(null, direction = "forward", scope=list(lower=null, upper=full))

# Backward Elimination
# backward_elimination <- step(full, direction = "backward")

# Remove the features identified by forward selection and backward elimination
music <- music %>% select(popularity, everything(), -c(song.hotttnesss))
music.forward <- music %>% select(popularity, song.mode, song.key, release.name, song.end_of_fade_in, song.time_signature_confidence, song.bars_start, song.time_signature, song.key_confidence, song.title, song.start_of_fade_out, song.beats_start, song.mode_confidence, artist.longitude, song.bars_confidence, song.duration, song.tatums_start, song.beats_confidence)
music.backward <- music %>% select(popularity, song.tatums_confidence, artist.latitude, song.tempo, song.artist_mbtags_count, artist.terms_freq, song.duration, song.start_of_fade_out, release.id, song.loudness, artist.hotttnesss, artist.familiarity, song.year)

#--------------------------
# Model building
#--------------------------

music.x_train <- music[1:3214,] %>% select(-c(popularity)) 
music.y_train <- music[1:3214,] %>% select(popularity)

music.x_test <- music[3215:4214,] %>% select(-c(popularity)) 
music.y_test <- music[3215:4214,] %>% select(popularity)

music.y_factor <- factor(music.y_test$popularity, labels = c(FALSE, TRUE))

music.forward.x_train <- music.forward[1:3214,] %>% select(-c(popularity)) 
music.forward.y_train <- music.forward[1:3214,] %>% select(popularity)

music.forward.x_test <- music.forward[3215:4214,] %>% select(-c(popularity)) 
music.forward.y_test <- music.forward[3215:4214,] %>% select(popularity)

music.forward.y_factor <- factor(music.forward.y_test$popularity, labels = c(FALSE, TRUE))

music.backward.x_train <- music.backward[1:3214,] %>% select(-c(popularity)) 
music.backward.y_train <- music.backward[1:3214,] %>% select(popularity)

music.backward.x_test <- music.backward[3215:4214,] %>% select(-c(popularity)) 
music.backward.y_test <- music.backward[3215:4214,] %>% select(popularity)

music.backward.y_factor <- factor(music.backward.y_test$popularity, labels = c(FALSE, TRUE))

#--------------------------
# Logistic Regression
#--------------------------

# Change alpha value to change regularization technique
# 1 = lasso regression
# 0 = ridge regression
glm <- cv.glmnet(data.matrix(music.x_train), data.matrix(music.y_train), family = "binomial", alpha = 1)

glm.predictions <- predict(glm, data.matrix(music.x_test), s = "lambda.min", type="class")
glm.classes <- rep(FALSE, length(glm.predictions))
glm.classes[glm.predictions == 1] <- TRUE

glm.factor <- factor(glm.classes, labels = c(FALSE, TRUE))

confusionMatrix(music.y_factor, glm.factor)

# PRODUCES NO TRUES
glm.forward <- cv.glmnet(data.matrix(music.forward.x_train), data.matrix(music.forward.y_train), family = "binomial", alpha = 1)

glm.forward.predictions <- predict(glm.forward, data.matrix(music.forward.x_test), s = "lambda.min", type="class")
glm.forward.classes <- rep(FALSE, length(glm.forward.predictions))
glm.forward.classes[glm.forward.predictions == 1] <- TRUE

glm.forward.factor <- factor(glm.forward.classes, labels = c(FALSE))

# Will cause an error, as glm.forward.factor contains no TRUEs
#confusionMatrix(music.forward.y_factor, glm.forward.factor)


glm.backward <- cv.glmnet(data.matrix(music.backward.x_train), data.matrix(music.backward.y_train), family = "binomial", alpha = 1)

glm.backward.predictions <- predict(glm.backward, data.matrix(music.backward.x_test), s = "lambda.min", type="class")
glm.backward.classes <- rep(FALSE, length(glm.backward.predictions))
glm.backward.classes[glm.backward.predictions == 1] <- TRUE

glm.backward.factor <- factor(glm.backward.classes, labels = c(FALSE, TRUE))

confusionMatrix(music.backward.y_factor, glm.backward.factor)

#--------------------------
# Multi-Layer Perceptron
#--------------------------

mlp <- monmlp.fit(data.matrix(music.x_train), y = data.matrix(music.y_train), hidden1 = 2, iter.max = 500)

mlp.predictions <- monmlp.predict(data.matrix(music.x_test), mlp)
mlp.classes <- rep(FALSE, length(mlp.predictions))
mlp.classes[mlp.predictions >= 0.4] <- TRUE

mlp.factor <- factor(mlp.classes, labels = c(FALSE, TRUE))

confusionMatrix(music.y_factor, mlp.factor)

mlp.forward <- monmlp.fit(data.matrix(music.forward.x_train), y = data.matrix(music.forward.y_train), hidden1 = 2, iter.max = 500)

mlp.forward.predictions <- monmlp.predict(data.matrix(music.forward.x_test), mlp.forward)
mlp.forward.classes <- rep(FALSE, length(mlp.forward.predictions))
mlp.forward.classes[mlp.forward.predictions >= 0.4] <- TRUE

mlp.forward.factor <- factor(mlp.forward.classes, labels = c(FALSE, TRUE))

confusionMatrix(music.forward.y_factor, mlp.forward.factor)

mlp.backward <- monmlp.fit(data.matrix(music.backward.x_train), y = data.matrix(music.backward.y_train), hidden1 = 2, iter.max = 500)

mlp.backward.predictions <- monmlp.predict(data.matrix(music.backward.x_test), mlp.backward)
mlp.backward.classes <- rep(FALSE, length(mlp.backward.predictions))
mlp.backward.classes[mlp.backward.predictions >= 0.4] <- TRUE

mlp.backward.factor <- factor(mlp.backward.classes, labels = c(FALSE, TRUE))

confusionMatrix(music.backward.y_factor, mlp.backward.factor)

#--------------------------
# Support Vector Machine
#--------------------------
# Tuning models - is done once to determine the optimal parameters for each kernel
# tune(svm, train.x = music.forward.x_train, train.y = music.forward.y_train, kernel = "linear", type = "C", ranges = list(cost = c(0.01, 0.1, 1, 10)))
# tune(svm, train.x = music.forward.x_train, train.y = music.forward.y_train, kernel = "radial", type = "C", ranges = list(cost = c(0.01, 0.1, 1, 10), gamma = c(0.05, 0.5, 1, 2)))
# tune(svm, train.x = music.forward.x_train, train.y = music.forward.y_train, kernel = "radial", type = "C", ranges = list(cost = c(0.01, 0.1, 1, 10), gamma = c(0.05, 0.5, 1, 2), degree = c(1, 2, 3, 4, 5)))
# tune(svm, train.x = music.forward.x_train, train.y = music.forward.y_train, kernel = "sigmoid", type = "C", ranges = list(cost = c(0.01, 0.1, 1, 10), gamma = c(0.05, 0.5, 1, 2)))

# Models with optimal parameters
svm.forward.linear <- svm(music.forward.y_train$popularity ~ ., data = music.forward.x_train, cross = 10, type = "C", kernel = "linear", cost = 0.01)
svm.forward.radial <- svm(music.forward.y_train$popularity ~ ., data = music.forward.x_train, cross = 10, type = "C", kernel = "radial", cost = 1, gamma = 0.5)
svm.forward.polynomial <- svm(music.forward.y_train$popularity ~ ., data = music.forward.x_train, cross = 10, type = "C", kernel = "polynomial", cost = 1, gamma = 0.5, degree = 1)
svm.forward.sigmoid <- svm(music.forward.y_train$popularity ~ ., data = music.forward.x_train, cross = 10, type = "C", kernel = "sigmoid", cost = 0.01, gamma = 0.05)

svm.forward.predictions <- predict(svm.forward.radial, music.forward.x_test)

confusionMatrix(music.forward.y_factor, svm.forward.predictions)


# Tuning models - is done once to determine the optimal parameters for each kernel
# tune(svm, train.x = music.forward.x_train, train.y = music.forward.y_train, kernel = "linear", type = "C", ranges = list(cost = c(0.01, 0.1, 1, 10)))
# tune(svm, train.x = music.forward.x_train, train.y = music.forward.y_train, kernel = "radial", type = "C", ranges = list(cost = c(0.01, 0.1, 1, 10), gamma = c(0.05, 0.5, 1, 2)))
# tune(svm, train.x = music.forward.x_train, train.y = music.forward.y_train, kernel = "radial", type = "C", ranges = list(cost = c(0.01, 0.1, 1, 10), gamma = c(0.05, 0.5, 1, 2), degree = c(1, 2, 3, 4, 5)))
# tune(svm, train.x = music.forward.x_train, train.y = music.forward.y_train, kernel = "sigmoid", type = "C", ranges = list(cost = c(0.01, 0.1, 1, 10), gamma = c(0.05, 0.5, 1, 2)))

# Models with optimal parameters
svm.backward.linear <- svm(music.backward.y_train$popularity ~ ., data = music.backward.x_train, cross = 10, type = "C", kernel = "linear", cost = 0.01)
svm.backward.radial <- svm(music.backward.y_train$popularity ~ ., data = music.backward.x_train, cross = 10, type = "C", kernel = "radial", cost = 1, gamma = 0.5)
svm.backward.polynomial <- svm(music.backward.y_train$popularity ~ ., data = music.backward.x_train, cross = 10, type = "C", kernel = "polynomial", cost = 1, gamma = 0.5, degree = 1)
svm.backward.sigmoid <- svm(music.backward.y_train$popularity ~ ., data = music.backward.x_train, cross = 10, type = "C", kernel = "sigmoid", cost = 0.01, gamma = 0.05)

svm.backward.predictions <- predict(svm.backward.radial, music.backward.x_test)

confusionMatrix(music.backward.y_factor, svm.backward.predictions)