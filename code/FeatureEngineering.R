# Load libraries:
library(plyr)
library(dplyr)
library(readr)
library(caret)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(ranger)
library(e1071)
library(pROC)
library(randomForest)
library(glmnet)
library(ggmosaic)
library(ggbiplot)

# Common functions
# Pre-process data:
pre_process <- function(data){
  data %>% 
    mutate(Gender = factor(Gender),
           Married = factor(Married),
           Dependents = as.integer(gsub("+", "", Dependents, fixed = TRUE)),
           Education = factor(Education),
           Self_Employed = factor(Self_Employed),
           Credit_History = factor(Credit_History),
           Property_Area = factor(Property_Area),
           Loan_Amount_Term = factor(Loan_Amount_Term))
}

# clean data: impute missing values.
impute_value <- function(data, column, value){
  data[[column]][which(is.na(data[[column]]))] <- value
  return(data)
}

clean <- function(data){
  data %>% 
    # Impute Categorical variables with the most common value
    impute_value("Gender", "Male") %>% 
    impute_value("Married", "Yes") %>% 
    impute_value("Self_Employed", "No") %>% 
    impute_value("Loan_Amount_Term", 360) %>% 
    impute_value("Credit_History", 1) %>% 
    # Impute median number of dependents, by gender and marital status
    group_by(Gender, Married) %>% 
    mutate(Dependents = ifelse(is.na(Dependents), 
                               as.integer(median(Dependents, na.rm = TRUE)), 
                               Dependents)) %>% 
    ungroup() %>% 
    # Impute mean loan amount, by property area and loan term length
    group_by(Property_Area, Loan_Amount_Term) %>% 
    mutate(LoanAmount = ifelse(is.na(LoanAmount), 
                               as.integer(mean(LoanAmount, na.rm = TRUE)), 
                               LoanAmount)) %>% 
    ungroup()
}

# plotting functions:
mosaic_plot <- function(data, varx, vary){
  stats <- data %>%
    count_(varx)
  
  total_rows <- dim(data)[1]
  data$weight = 0
  
  for (s in 1:dim(stats)[1]) {
    data$weight[which(data[[varx]] == stats[[1]][[s]])] = stats$n[[s]]/total_rows
  }
  
  data %>% 
    ggplot() +
    geom_mosaic(aes(weight = weight, x = product(get(vary), get(varx)), fill = get(vary))) +
    labs(x = varx,
         y = "Percent",
         title = sprintf("%s vs %s", vary, varx)) + 
    scale_fill_discrete(name = vary)
  
}

box_plot <- function(data, varx, vary){
  data %>% 
    ggplot(aes_string(x = varx, y = vary)) + 
    geom_boxplot(varwidth = TRUE) +
    labs(x = varx,
         y = vary,
         title = sprintf("%s vs %s", vary, varx))  
  
  
}


# plot pca
plot_pca <- function(data){
  train_pca <- data %>% 
    select(-Loan_ID) %>%
    # convert factors to numeric values
    mutate_if(is.factor, as.numeric) %>% 
    # Calculate PCA
    prcomp(center = TRUE, scale = TRUE) 

  ggbiplot(train_pca) +
    lims(y = c(-3,3) )
}

# Define a common training control:
train_control <- trainControl(method = "repeatedcv",
                              number = 1,
                              repeats = 1,
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE,
                              verboseIter = FALSE)

# run logistic regression, only pass LoanStatus, and input variables as part of train/test DFs.
logistic_regression <- function(data){
  # make training/testing partitions
  
  train_idx = createDataPartition(data$Loan_Status, p = 0.7, list = FALSE)
  train_data <- data[train_idx,] %>% select(-Loan_ID)
  test_data <- data[-train_idx,]
  
  logit_model <- train(Loan_Status ~ .,
                       data = train_data,
                       method = "glm",
                       metric = "ROC",
                       trControl = train_control,
                       family = binomial(link = "logit"))
  logit_pred <- predict(logit_model, test_data, type = "prob")
  logit_accuracy <- data.frame(loan_id = test_data$Loan_ID,
                               loan_granted = test_data$Loan_Status == "Y",
                               logit = logit_pred$Y > 0.5) %>%
    summarize(n = n(),
              accuracy = sum(logit == loan_granted)/n)
  
  print(round(logit_accuracy$accuracy[1]*100))
  print(summary(logit_model)$coefficients)
}

# load training dataset:
loan_data <- read_csv(file.path("..","data","train.csv")) %>% 
  pre_process() %>% 
  clean() %>% 
  mutate(Loan_ID, Loan_Status = factor(Loan_Status))

loan_data %>% 
  select(Loan_ID, Loan_Status, Credit_History) %>% 
  logistic_regression()

loan_data %>% 
  plot_pca()
