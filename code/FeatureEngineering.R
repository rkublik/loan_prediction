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
           Property_Area = factor(Property_Area)
           #Loan_Amount_Term = factor(Loan_Amount_Term)
           )
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
    impute_value("Credit_History", 0) %>% # if we don't have credit history, don't give benefit of doubt 
    # Impute median number of dependents, by gender and marital status
    group_by(Gender, Married) %>% 
    mutate(Dependents = ifelse(is.na(Dependents), 
                               as.integer(median(Dependents, na.rm = TRUE)), 
                               Dependents)) %>% 
    ungroup()
    # Impute mean loan amount, by property area and loan term length
    # group_by(Property_Area, Loan_Amount_Term) %>% 
    # mutate(LoanAmount = ifelse(is.na(LoanAmount), 
    #                            as.integer(mean(LoanAmount, na.rm = TRUE)), 
    #                            LoanAmount)) %>% 
    # ungroup()
    # Impute mean loan amount, by Education and Self_Employed
    #group_by(Education, Self_Employed) %>% 
    #mutate(LoanAmount = ifelse(is.na(LoanAmount), 
    #                            as.integer(mean(LoanAmount, na.rm = TRUE)), 
    #                            LoanAmount)) %>% 
    #ungroup()
    
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

  ggbiplot(train_pca)
}

# Define a common training control:
train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 5,
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
  mutate(Loan_Status = factor(Loan_Status))

# Test that my main functions work...
loan_data %>% 
  select(Loan_ID, Loan_Status, Credit_History) %>% 
  logistic_regression()



# Basic approach:
# make new variables, check with pca, run logistic regression.

# 1. look at distributions of continuous variables:
# Applicant Income
ggplot(data = loan_data, aes(x = ApplicantIncome)) + geom_histogram(binwidth = 1000)
summary(loan_data$ApplicantIncome)
## so, no missing values, min income of 150 seems a bit odd...
## what kind of loan is that person looking for?
loan_data %>% 
  filter(ApplicantIncome < 500)
## In these cases, the coapplication income is quite high... let's look at the total income.

loan_data <- loan_data %>% 
  mutate(TotalIncome = ApplicantIncome + CoapplicantIncome)

# TotalIncome:
summary(loan_data$TotalIncome)
ggplot(data = loan_data, aes(x = log(TotalIncome))) +
  geom_histogram(bins = 50)
## some definite outliers:
loan_data %>% 
  select(ApplicantIncome, CoapplicantIncome, TotalIncome, LoanAmount, Loan_Status) %>% 
  filter(TotalIncome > 20000) 

## we'll make a new variable with the log_income:
loan_data <- loan_data %>% 
  mutate(log_income = log(TotalIncome))

# Loan Amount:
summary(loan_data$LoanAmount)
# have 22 NA's
# look at some groupings:
ggplot(data = loan_data, aes(x = Loan_Amount_Term, y = LoanAmount)) + 
  geom_boxplot() + 
  facet_wrap(~ Property_Area) 
  
ggplot(data = loan_data, aes(x = Education, y = LoanAmount)) + 
  geom_boxplot() + 
  facet_wrap(~ Self_Employed)
#seems to be more variation with Education and Self-Employed.
# impute missing values this way:
loan_data <- loan_data %>% 
  group_by(Education, Self_Employed) %>% 
  mutate(LoanAmount = ifelse(is.na(LoanAmount), 
                            as.integer(mean(LoanAmount, na.rm = TRUE)), 
                            LoanAmount)) %>% 
  ungroup()


# Now, let's look at the Loan Amount:
ggplot(data = loan_data, aes(x = LoanAmount)) + 
  geom_histogram()

loan_data %>% 
  filter(LoanAmount <20) %>% 
  select(LoanAmount, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term) %>% 
  arrange(LoanAmount)
# a $9000 loan for 30 years seems unlikely. How should we fill this?
#Could it by a typo, really 90K?
loan_data %>% 
  filter(LoanAmount > 90, LoanAmount < 95) %>% 
  select(LoanAmount, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term) %>% 
  arrange(LoanAmount)

# that seems reasonable. 
loan_data$LoanAmount[which(loan_data$LoanAmount == 9)] = 90

# now look at log(LoanAmount)
ggplot(data = loan_data, aes(x = log(LoanAmount))) + 
  geom_histogram()

# make a new column:
loan_data <- loan_data %>% 
  mutate(log_LoanAmount = log(LoanAmount))

plot_pca(loan_data)

## Make some new variables:
loan_data <- loan_data %>% 
  mutate(loan_income_ratio = 1000*LoanAmount/TotalIncome,
         household_size = 1 + as.integer(Married == "Yes") + Dependents,
         monthly_payment = 1000*LoanAmount/Loan_Amount_Term,
         payment_income_ratio = monthly_payment / TotalIncome,
         log_payment_ratio = log(payment_income_ratio),
         income_person = TotalIncome/household_size,
         single_parent = (Married == "No" & Dependents > 0))

loan_data$single_parent

# now let's explore these new variables:
ggplot(loan_data, aes(x = loan_income_ratio, fill = Loan_Status)) + 
  geom_histogram() + 
  facet_wrap(~Credit_History)
# maybe need to handle loan_income_ratio > 40 separately?
loan_data %>% 
  filter(loan_income_ratio > 40) %>% 
  select(LoanAmount, loan_income_ratio, Credit_History, Loan_Status) %>% 
  mutate(ch_status = Credit_History == as.integer(Loan_Status == "Y")) %>% 
  summarize(ch_success = mean(ch_status))

# Quick inspection suggests that credit_history is not a good predictor here... only 50% accuracy

ggplot(loan_data, aes(x = monthly_payment, fill = Loan_Status)) + 
  geom_histogram() + 
  facet_wrap(~Credit_History)

loan_data %>% 
  filter(monthly_payment > 200) %>% 
  select(LoanAmount, monthly_payment, TotalIncome, Credit_History, Loan_Status) %>% 
  mutate(ch_status = Credit_History == as.integer(Loan_Status == "Y"))
# for high monthly payments, credit history is a poor predictor

ggplot(loan_data, aes(x = payment_income_ratio, fill = Loan_Status)) + 
  geom_histogram() + 
  facet_wrap(~Credit_History)

ggplot(loan_data, aes(x = payment_income_ratio, fill = Loan_Status)) + 
  geom_histogram() + 
  facet_grid(single_parent ~ Credit_History)


loan_data %>% 
  filter(payment_income_ratio > 0.2) %>% 
  select(LoanAmount, Loan_Amount_Term, payment_income_ratio, TotalIncome, Credit_History, Loan_Status) %>% 
  mutate(ch_status = Credit_History == as.integer(Loan_Status == "Y")) %>% 
  arrange(payment_income_ratio)

# Credit history not a good indicator once payment_income_ratio > 0.5?

ggplot(loan_data, aes(x = TotalIncome, fill = Loan_Status)) + geom_histogram(bins = 100)

ggplot(loan_data, aes(x = TotalIncome, y = income_person, col = factor(household_size))) +
  geom_point() + 
  facet_wrap(~ Loan_Status )

loan_data %>% 
  #select(Loan_ID, Credit_History, Loan_Status, household_size, monthly_payment, TotalIncome, log_income, 
  #       payment_income_ratio, loan_income_ratio) %>% 
  plot_pca() 

loan_data %>% 
  #select(-Credit_History) %>% 
  select(Credit_History, Loan_ID, Loan_Status, household_size, TotalIncome, log_income, 
         payment_income_ratio, loan_income_ratio, Education, Self_Employed, Property_Area, LoanAmount,
         monthly_payment) %>% 
  logistic_regression()  

# still not making much difference....
# Try some more variables:
# may need to look for outliers and deal with them differently....
loan_data %>% 
  summarize(has_coapp = mean(CoapplicantIncome > 0))

loan_data_new <-  loan_data %>% 
  mutate(TotalIncome = ApplicantIncome + CoapplicantIncome,
         LoanAmount = 1000*LoanAmount,
         payment = LoanAmount/Loan_Amount_Term,
         payment_ratio = payment / TotalIncome,
         household_size = 1 + as.integer(Married == "Yes") + Dependents,
         suburban = as.integer(Property_Area == "Semiurban"),
         has_coapplicant = as.numeric(CoapplicantIncome > 0),
         Graduate_Education = as.numeric(Education == "Graduate"),
         Self_Employed = as.numeric(Self_Employed == "Yes"), 
         Credit_History = as.numeric(Credit_History)) #%>% 
  #select(Loan_ID, TotalIncome, LoanAmount, payment, payment_ratio, household_size, suburban,
  #       has_coapplicant, Graduate_Education, Self_Employed, Credit_History, Loan_Status, 
  #       Married, Dependents)
loan_data_new %>% plot_pca()
loan_data_new %>% logistic_regression()

decision_tree <- function(data){
  
  train_idx = createDataPartition(data$Loan_Status, p = 0.7, list = FALSE)
  train_data <- data[train_idx,] %>% select(-Loan_ID)
  test_data <- data[-train_idx,]
  
  rpart_model <- train(Loan_Status ~ .,
                       data = train_data,
                       method = "rpart",
                       metric = "ROC",
                       trControl = train_control)
  
   rpart_pred <- predict(rpart_model, test_data, type = "prob")
   rpart_accuracy <- data.frame(loan_id = test_data$Loan_ID,
                                loan_granted = test_data$Loan_Status == "Y",
                                logit = rpart_pred$Y > 0.5) %>%
     summarize(n = n(),
               accuracy = sum(logit == loan_granted)/n)
   
   print(round(rpart_accuracy$accuracy[1]*100))
   fancyRpartPlot(rpart_model$finalModel)
}


loan_data_new %>% 
  decision_tree()
