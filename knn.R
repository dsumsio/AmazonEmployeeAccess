library(vroom)
library(tidymodels)
library(embed)
library(tidyverse)
library(doParallel)
library(themis)

parallel::detectCores() #How many cores do I have?3
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

train_data = vroom('train.csv')
test_data = vroom('test.csv')

train_data$ACTION = as.factor(train_data$ACTION)

lr_recipe <- recipe(ACTION ~ ., data=train_data) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_lencode_mixed(all_nominal_predictors(), outcome =vars(ACTION)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_nominal_predictors()) %>%
  step_pca(all_predictors(), threshold=0.85)  %>%
  step_smote(all_outcomes(), neighbors=5)

## knn model
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode('classification') %>%
  set_engine('kknn')

# CV
## Grid of values to tune
tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)


knn_wf <- workflow() %>%
  add_recipe(lr_recipe) %>%
  add_model(knn_model)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid)

## Find best tuning params
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

## Findlize the workflow & fit
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

#make predictions
amazon_preds <- predict(final_wf, new_data=test_data, type = 'prob')

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  amazon_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) #rename pred to count (for submission to Kaggle)


# Make File
vroom_write(x=kaggle_submission, file="./knn_smote.csv", delim=",")

stopCluster(cl)
