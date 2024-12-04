library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(themis)


## SVM models
svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>%
  set_mode('classification') %>%
  set_engine('kernlab')

train_data = vroom('train.csv')
test_data = vroom('test.csv')

train_data$ACTION = as.factor(train_data$ACTION)


lr_recipe <- recipe(ACTION ~ ., data=train_data) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.8) %>%
  step_smote(all_outcomes(), neighbors=5)


nb_wf <- workflow() %>%
  add_recipe(lr_recipe) %>%
  add_model(svmRadial)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(rbf_sigma(), cost(),
                                      levels = 5)

## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(roc_auc)
  )


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

preds <- final_wf %>%
  predict(new_data = test_data, type = 'prob')

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) #rename pred to count (for submission to Kaggle)


## Write out file
vroom_write(x=kaggle_submission, file="./svm_smote.csv", delim=",")


