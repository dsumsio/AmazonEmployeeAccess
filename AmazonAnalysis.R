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
  #step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  #step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_rm(ROLE_CODE) %>%
  step_pca(all_predictors(), threshold=0.85) %>%
  step_smote(all_outcomes(), neighbors=5)

## apply the recipe to your data
prep <- prep(lr_recipe)
baked <- bake(prep, new_data = train_data)

## Model
logregmodel <- logistic_reg() %>%
  set_engine('glm')

## Workflow
logregwrkflow <- workflow() %>%
  add_recipe(lr_recipe) %>%
  add_model(logregmodel) %>%
  fit(data=train_data)

## Make predictions
amazon_preds <- predict(logregwrkflow, 
                        new_data = test_data,
                        type = 'prob')

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  amazon_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) #rename pred to count (for submission to Kaggle)
  

# Make File
vroom_write(x=kaggle_submission, file="./lr_smote.csv", delim=",")




########################## Penalized Log Reg #######################
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)


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


my_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine('glmnet')

amazon_workflow <- workflow() %>%
  add_recipe(lr_recipe) %>%
  add_model(my_mod)

## Grid of values to tune
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats = 1)

## Run the CV
CV_results <- amazon_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid)

## Find best tuning params
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

## Findlize the workflow & fit
final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

## Predict
amazon_preds <- final_wf %>%
  predict(new_data = test_data, type = 'prob')

## Format the Predictions for Submission to Kaggle
kaggle_submission <-  amazon_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(ACTION=.pred_1) #rename pred to count (for submission to Kaggle)

# Make File
vroom_write(x=kaggle_submission, file="./plr_smote.csv", delim=",")

stopCluster(cl)



