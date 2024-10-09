library(vroom)
library(tidymodels)
library(embed)
library(tidyverse)


train_data = vroom('train.csv')
test_data = vroom('test.csv')

train_data$ACTION = as.factor(train_data$ACTION)


lr_recipe <- recipe(ACTION ~ ., data=train_data) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .05) %>% # combines categorical values that occur
  step_dummy(all_nominal_predictors()) # %>% # dummy variable encoding
  # step_lencode_mixed(vars_I_want_to_target_encode, outcome = vars(target_var)) #target encoding (must
# also step_lencode_glm() and step_lencode_bayes()

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
vroom_write(x=kaggle_submission, file="./lr_1.csv", delim=",")







