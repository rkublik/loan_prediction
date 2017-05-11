## Initial exploration of the data:

library(readr)
library(dplyr)
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

# Quick check: Credit status is not sufficient to reject a loan, but almost...
loan_data %>% 
  group_by(Credit_History) %>% 
  select(Credit_History, Loan_Status) %>% 
  summarize(N = sum(Loan_Status == "N"),
            Y = sum(Loan_Status == "Y"))


# Now, we have filled in all the missing values.... Can we make any plots that will give us some insight?
quick_plot <- function(data, x, y){
  plot(data[[x]], data[[y]])
}
quick_plot(loan_data,"Gender", "Loan_Status")
quick_plot(loan_data,"Credit_History", "Loan_Status")
quick_plot(loan_data,"Dependents", "Loan_Status")
quick_plot(loan_data,"LoanAmount", "Loan_Status")
quick_plot(loan_data,"Married", "Loan_Status")
quick_plot(loan_data,"Education", "Loan_Status")
quick_plot(loan_data,"Self_Employed", "Loan_Status")
quick_plot(loan_data,"ApplicantIncome", "Loan_Status")
quick_plot(loan_data,"Property_Area", "Loan_Status")
quick_plot(loan_data,"Property_Area", "Credit_History")

# There isn't a single variable that seems to explain everything....
loan_data %>% 
  select(-Loan_ID) %>% 
  plot()
# Let's look at a few different machine learning algorithms: We will use the Caret package to take 
# care of the details of cross-validation for us. and determining which variables to include.

# first we will make a test and train split:
train_idx = createDataPartition(loan_data$Loan_Status, p = 0.7, list = FALSE)
loan = list(train = loan_data[train_idx,],
            test = loan_data[-train_idx,])

# Make a common training control:
train_control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 10,
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE,
                              verboseIter = FALSE)

# look for any co-linear variables, use pca for a quick check:

train_pca <- loan$train %>% 
  select(-Loan_ID) %>% 
  mutate(Gender = as.numeric(Gender),
         Married = as.numeric(Married),
         Education = as.numeric(Education),
         Self_Employed = as.numeric(Self_Employed),
         Loan_Amount_Term = as.numeric(Loan_Amount_Term),
         Credit_History = as.numeric(Credit_History),
         Property_Area = as.numeric(Property_Area),
         Loan_Status = as.numeric(Loan_Status)) %>% 
  prcomp(center = TRUE, scale = TRUE)

biplot(train_pca)

# see that LoanAmount and ApplicantIncome are almost co-linear
# Dependents, Gender, and Self_Employed are almost co-linear
# -- do we care about Gender and Self_Employed status? I think we shouldn't...
# Loan Term and Married status almost co-linear

# Algorithms to consider:
# logistic regression, decisiontree (rpart), randomForest, 

# Logistic Regression
glm_model <- train(Loan_Status ~ Credit_History + Married + Dependents + Education + 
                     ApplicantIncome + CoapplicantIncome + LoanAmount + Property_Area,
                   data = loan$train,
                   method = "glm",
                   metric = "ROC",
                   trControl = train_control,
                   family = binomial(link = "logit"))

summary(glm_model)
# Looking at the coefficients, it seems that Credit_History, and Property_Area are the most significant factors
# recall that the loan amount is mostly proportional to the applicant income...
# anyway, try a predictive model based on only credit history and propery area, and compare their performance

glm_model_reduced <- train(Loan_Status ~ Credit_History + Property_Area,
                          data = loan$train,
                          method = "glm",
                          metric = "ROC",
                          trControl = train_control,
                          family = binomial(link = "logit"))

summary(glm_model_reduced)


# compare performance on the test data set:
glm_pred <- predict(glm_model, loan$test, type = "prob")
glm_pred_r <- predict(glm_model_reduced, loan$test, type = "prob")

glm_comp <- data.frame(loan_id = loan$test$Loan_ID,
                       loan_granted = loan$test$Loan_Status == "Y",
                       glm = glm_pred$Y > 0.5,
                       glm_r = glm_pred_r$Y > 0.5)

glm_comp %>% 
  summarize(n = n(),
          glm_accuracy = sum(glm == loan_granted)/n,
          glm_r_accuracy = sum(glm_r == loan_granted)/n)

# so we see that the reduced model has higher accuracy on the test data. (81% over 80%)
# let's compare where the models disagree:
ld <- glm_comp %>% 
  filter(glm != glm_r) %>% 
  select(loan_id)
  
# only on 3 loans.... let's look at the data for those three:
loan$test %>% 
  filter(Loan_ID %in% ld$loan_id) %>% 
  print()

# This doesn't actually tell us much...
# is there something that could have told us that these are the most important variables?
# looking at the plots again, I don't see anything obvious... Other than the credit_History

# Just use Credit History
loan$test %>% 
  mutate(loan_approved = Loan_Status == "Y",
         credit_only = Credit_History == 1) %>% 
  summarize(n = n(),
            success = sum(credit_only)/n)

# And, going by credit history only, we have an 85% success rate....

## Decision Tree (rpart)
# look at titanic tutorial for some details of how decision trees work.

rpart_model <- train(Loan_Status ~ Credit_History + Married + Dependents + Education + ApplicantIncome + CoapplicantIncome + LoanAmount + Property_Area,
                     #Loan_Status ~ Credit_History + Married + Education + Property_Area,
                     data = loan$train,
                     method = "rpart",
                     metric = "ROC",
                     trControl = train_control)


plot(rpart_model$finalModel)
text(rpart_model$finalModel)
str(rpart_model)
fancyRpartPlot(rpart_model$finalModel)


# now how does this prediction work out?
rpart_pred <- predict(rpart_model,loan$test, type = "raw")

rpart_res <- data.frame(loan_id = loan$test$Loan_ID,
                       loan_granted = loan$test$Loan_Status,
                       rpart = rpart_pred)
rpart_res %>% 
  summarize(n = n(),
            success = sum(rpart == loan_granted)/n)

## rpart give 80% success rate on new data.


## Random Forest (ranger)
# make many decision trees, and provide the classification that is most predicted
ranger_model <- train(Loan_Status ~ Credit_History + Married + Dependents + #Education #+ 
                       ApplicantIncome + CoapplicantIncome + LoanAmount + Property_Area
                      ,
                     data = loan$train,
                     method = "ranger",
                     tuneLength = 10,
                     trControl = trainControl(method = "repeatedcv",
                                              number = 10,
                                              repeats = 10,
                                              verboseIter = FALSE,
                                              classProbs = TRUE))



# now how does this prediction work out?
ranger_pred <- predict(ranger_model,loan$test, type = "raw")

ranger_res <- data.frame(loan_id = loan$test$Loan_ID,
                        loan_granted = loan$test$Loan_Status,
                        rpart = rpart_pred)
ranger_res %>% 
  summarize(n = n(),
            success = sum(rpart == loan_granted)/n)

# This also gives an 80% success rate...


# Support Vector Machine?




# now that we've compared the models, we need to look at the test data set:

# Instead of just throwing all variables at the problem, is there some justification to focus on a few?
# Credit History, Loan Amount, Income for example


loan_test <- read_csv(file.path("..","data","test.csv"))
summary(loan_test)

#As this has some missing values, we will need to impute them.  Of note, is 29 missing 
#credit history values. Not sure if there's a better way than just setting missing to 1...

loan_test %>% 
  mutate(Credit_History = ifelse(is.nan(Credit_History), 1, Credit_History),
         Loan_Status = ifelse(Credit_History == 1, "Y", "N")) %>% 
  select(Loan_ID, Loan_Status) %>% 
  write_csv(file.path("..","output","credit_only.csv"))

# This gives 73% accuracy on the test data, puts me in 1070th place.

# work through each method as part of writeup, give how new solution changes both accuracy and position in leaderboard
# need to think about how to impute values.... use same approach as before. What happens if I leave missing values out?
# Had missing credit history values before, can I set them to -1, and try again? how will that affect things? -- no got errors when doing this... need to think about why....



