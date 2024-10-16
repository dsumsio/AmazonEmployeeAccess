library(vroom)
library(tidymodels)
library(embed)
library(tidyverse)
library(doParallel)

train_data = vroom('train.csv')
test_data = vroom('test.csv')

train_data$ACTION = as.factor(train_data$ACTION)

lr_recipe <- recipe(ACTION ~ ., data=train_data) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .05) %>% # combines categorical values that occur
  step_dummy(all_nominal_predictors())

## knn model
knn_model <- nearest_neighbor(neighbors = 25) %>%
  set_mode('classification') %>%
  set_engine('kknn')

knn_wf <- workflow() %>%
  add_recipe(lr_recipe) %>%
  add_model(knn_model) %>%
  fit(data=train_data)




amazon_preds <- predict(knn_wf, new_data=test_data, type = 'prob')

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  amazon_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) #rename pred to count (for submission to Kaggle)


# Make File
vroom_write(x=kaggle_submission, file="./knn.csv", delim=",")
