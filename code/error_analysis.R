# Exploring the sources of error... 

# load libraries
library(readr)
library(dplyr)
library(caret)
#library(rpart)
#library(rattle)
#library(rpart.plot)
library(RColorBrewer)
#library(ranger)
#library(e1071)
#library(pROC)
#library(randomForest)
library(glmnet)


# Load data, do cleanup and missing value imputation
loan_data <- read_csv(file.path("..","data","train.csv")) %>% 
  mutate(Gender = factor(Gender),
         Married = factor(Married),
         Dependents = as.integer(gsub("+", "", Dependents, fixed = TRUE)),
         Education = factor(Education),
         Self_Employed = factor(Self_Employed),
         Credit_History = factor(Credit_History),
         Property_Area = factor(Property_Area),
         Loan_Status = factor(Loan_Status),
         Loan_Amount_Term = factor(Loan_Amount_Term))

summary(loan_data)
# We have a number of variables with NAs. We can impute these using some of the other data:
# Since 5/6ths of the applicants are male, we will impute the NA's with Male:
# 4/6ths are married, so impute NA's with Yes
# 5/6ths are not self employed, impute with No
# Loan_Amount_Term: impute with median (360)
# Credit_History: 5/6ths have good credit, impute with 1
impute_value <- function(data, column, value){
  data[[column]][which(is.na(data[[column]]))] <- value
  return(data)
}

loan_data <- loan_data %>% 
  impute_value("Gender", "Male") %>% 
  impute_value("Married", "Yes") %>% 
  impute_value("Self_Employed", "No") %>% 
  impute_value("Loan_Amount_Term", 360) %>% 
  impute_value("Credit_History", 1)


summary(loan_data)

# for the Dependents, impute by the mean number of dependents based on gender and married status
loan_data <- loan_data %>% 
  group_by(Gender, Married) %>% 
  mutate(Dependents = ifelse(is.na(Dependents), as.integer(median(Dependents, na.rm = TRUE)), Dependents)) %>% 
  ungroup()
summary(loan_data)

# for the LoanAmount: impute by the mean loan amount, based on Property_Area and Loan_Amount_Term
loan_data <- loan_data %>% 
  group_by(Property_Area, Loan_Amount_Term) %>% 
  mutate(LoanAmount = ifelse(is.na(LoanAmount), as.integer(mean(LoanAmount, na.rm = TRUE)), LoanAmount)) %>% 
  ungroup()

# Quick check:use credit history as sole predictor:
perf_metrics <- function(df){
  # loan rejected is the smaller class, use as positive:
  tp <- sum(!df$loan_granted & !df$prediction)
  fp <- sum(df$loan_granted & !df$prediction)
  fn <- sum(!df$loan_granted & df$prediction)
  tn <- sum(df$loan_granted & df$prediction)
  res <- list()
  accuracy <- (tp + tn)/(tp + fp + fn + tn)
  precision <- tp/(tp + fp)
  recall <- tp/(tp + fn)
  res["accuracy"] <- accuracy
  res["precision"] <- precision
  res["recall"] <- recall
  res["Fscore"] <- 2*(precision*recall)/(precision + recall)
  res <- data.frame(res)
}


conf <- loan_data %>% 
  group_by(Credit_History) %>% 
  select(Credit_History, Loan_Status) %>% 
  summarize(N = sum(Loan_Status == "N"),
            Y = sum(Loan_Status == "Y"))

credit_only <- loan_data %>% select(Credit_History, Loan_Status) %>% 
  mutate(loan_granted = Loan_Status == "Y",
         prediction = Credit_History == 1)
  
# Setup training and testing partitions
train_idx <-  createDataPartition(loan_data$Loan_Status, p = 0.7, list = FALSE)
train <-  loan_data[train_idx,]
test <-  loan_data[-train_idx,]
train <- train %>% select(-Loan_ID)
test <- test %>% select(-Loan_ID)
# Investigate causes of error, use logistic regression:

# Logistic Regression
train_control <- trainControl(method = "none",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE,
                              verboseIter = FALSE)

glm_model <- train(Loan_Status ~ . ,
                   data = train,
                   method = "glm",
                   metric = "ROC",
                   trControl = train_control,
                   family = binomial(link = "logit"))

summary(glm_model)
glm_pred <- predict(glm_model, test, type = "prob")
glm_comp <- data.frame(loan_granted = test$Loan_Status == "Y",
                       prediction = glm_pred$Y > 0.5)

perf_metrics(glm_comp)

# Are we dealing with a rare event?
loan_data %>%
  mutate(credit_binary = Credit_History==1,
         approved_binary = Loan_Status == "Y") %>% 
  summarize(frac_rejected = mean(Loan_Status=="N"),
            frac_poor_credit = mean(Credit_History==0),
            frac_not_predicted = mean(credit_binary != approved_binary)
            )


# Plot learning curves:
num_train = c(1:length(train_idx))
train_err = numeric(length(train_idx))
test_err = numeric(length(train_idx))
for (i in 2:length(train_idx)) {
  glm_model <- train(Loan_Status ~ . ,
                     data = train[1:i,],
                     method = "glm",
                     metric = "ROC",
                     trControl = train_control,
                     family = binomial(link = "logit"))
  
  train_pred <- predict(glm_model, train[1:i,], type = "prob")
  comp_train <- data.frame(loan_granted = train[1:i,]$Loan_Status == "Y",
                          prediction = train_pred[1:i,]$Y > 0.5)

  test_pred <- predict(glm_model, test, type = "prob")
  comp_test <- data.frame(loan_granted = test$Loan_Status == "Y",
                           prediction = test_pred$Y > 0.5)
  
  train_err[i] <-  1 - perf_metrics(comp_train)$accuracy
  test_err[i] <-  1 - perf_metrics(comp_test)$accuracy
}
train_err <- train_err[2:length(train_err)]
test_err <- test_err[2:length(test_err)]
num_train <- num_train[2:length(num_train)]
learning_curve <- data.frame(num_train, train_err, test_err)

learning_curve %>% 
  ggplot(aes(x = num_train)) +
  geom_line(aes(y = train_err, color = "Training Error")) +
  geom_line(aes(y = test_err, color = "Testing Error"))

# suffering from high bias...

# Look at examples of test data that was not correctly classified:
classification_errors <- test %>% 
  cbind(Prediction = test_pred$Y > 0.5) %>% 
  mutate(Loan_approved = Loan_Status == "Y",
         classified_correctly = Loan_approved == Prediction) %>% 
  filter(Loan_approved != Prediction)

# 35 examples incorrectly classified
