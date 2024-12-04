library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)
library(themis)

numcores <- parallel::detectCores() #How many cores do I have?
cl <- makePSOCKcluster(numcores)
registerDoParallel(cl)


train_data = vroom('train.csv')
test_data = vroom('test.csv')

train_data$ACTION = as.factor(train_data$ACTION)


lr_recipe <- recipe(ACTION ~ ., data=train_data) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  #step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_rm(ROLE_CODE) %>%
  step_pca(all_predictors(), threshold=0.85) %>%
  step_smote(all_outcomes(), neighbors=5)


rand_for <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

## Workflow
rf_wf <- workflow() %>%
  add_recipe(lr_recipe) %>%
  add_model(rand_for) 

## Grid of values to tune over (rand forest)
grid_of_tuning_params_randfor <- grid_regular(mtry(range = c(1,100)),
                                              min_n(),
                                              levels = 5)

## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)

## Run the CV
CV_results <- rf_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params_randfor,
    metrics = metric_set(roc_auc)
  )


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

## predict
preds <- final_wf %>%
  predict(new_data = test_data, type = 'prob')

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) #rename pred to count (for submission to Kaggle)


## Write out file
vroom_write(x=kaggle_submission, file="./randfor_smote.csv", delim=",")



stopCluster(cl)












