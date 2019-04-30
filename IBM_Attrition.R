#Importing Libraries:
library(neuralnet)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(e1071)

#Importing the dataset 

#setwd("~/Documents/stanford/datascience/employee_attrition_prediction")
data = read.csv("C:/Users/mahesh/Downloads/Attrition(1).csv")
head(data)

#Removing columns which have same value for all and determining class of importance
summary(data$EmployeeCount)
summary(data$EmployeeNumber)
summary(data$Over18)
summary(data$StandardHours)

cleaned_data=data[,-c(9,10,22,27)]

cleaned_data$Attrition<-ifelse(cleaned_data$Attrition=='Yes' ,1,0)

#Data Pre-Processing
cleaned_data$Attrition<-as.factor(cleaned_data$Attrition)
cleaned_data$Education<-as.factor(cleaned_data$Education)
cleaned_data$EnvironmentSatisfaction<-as.factor(cleaned_data$EnvironmentSatisfaction)
cleaned_data$JobInvolvement<-as.factor(cleaned_data$JobInvolvement)
cleaned_data$JobLevel<-as.factor(cleaned_data$JobLevel)
cleaned_data$JobSatisfaction<-as.factor(cleaned_data$JobSatisfaction)
cleaned_data$PerformanceRating<-as.factor(cleaned_data$PerformanceRating)
cleaned_data$RelationshipSatisfaction<-as.factor(cleaned_data$RelationshipSatisfaction)
cleaned_data$StockOptionLevel<-as.factor(cleaned_data$StockOptionLevel)
cleaned_data$WorkLifeBalance<-as.factor(cleaned_data$WorkLifeBalance)

str(cleaned_data)

#Finding NA's in dataset
anyNA(data)

summary(data)
#Step:2 Cleaning data

#Step3: detecting outliers

boxplot(cleaned_data$DailyRate)
boxplot(cleaned_data$DistanceFromHome)
boxplot(cleaned_data$HourlyRate)
boxplot(cleaned_data$JobLevel)
boxplot(cleaned_data$MonthlyIncome)
boxplot(cleaned_data$MonthlyRate)
boxplot(cleaned_data$PercentSalaryHike)

#Splitting data and Applying logistic regression
set.seed(321)
seqsample<-sample(seq_len(nrow(cleaned_data)), size= floor(0.75 * nrow(cleaned_data)))
train<- cleaned_data[seqsample,]
test<- cleaned_data[-seqsample,]
attach(train)

mdl<- glm(train$Attrition~., family = binomial(link='logit'), data= train[,-2])
summary(mdl)

log_predict<- predict(mdl, newdata = test[-2], type = "response")

#Tuning the model:
log_predict <- ifelse(log_predict > 0.25,1,0)

conf <-table(test$Attrition, log_predict, dnn = c("Actual", "Predicted"))
conf

#SVM Model:
set.seed(321)

svm1 <- svm(Attrition ~ ., data=train)
svm_pred <- predict(svm1,test)
conf <-table(test$Attrition, svm_pred, dnn = c("Actual", "Predicted"))
conf

svm2 <- svm(Attrition ~ ., data=train, kernel="radial", cost=0.1, gamma=0.0004)
svm_pred <- predict(svm2,test)
print(table(test$Attrition,svm_pred))

#Random Forest:
set.seed(321)
fit <- randomForest(train$Attrition ~ ., data=train, importance=TRUE, ntree=10000)

Prediction <- predict(fit, test)

conf<-table(test$Attrition, Prediction)
conf

#Neural networks:
#Step:6 Train the model
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

model_trained_nn=train(Attrition ~., train, method="dnn", trControl=trctrl)

#Step:7 Predict using model and dataset
predicted_attrition_nn=predict(model_trained_nn,test)

conf<- table(test$Attrition, predicted_attrition_nn)
conf



xgbGrid <- expand.grid(nrounds = 300,
                       max_depth = 1,
                       eta = 0.3,
                       gamma = 0.01,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = 0.9)

set.seed(12)
mdl1 <- train(Attrition ~.,train,method = 'xgbTree',tuneGrid = xgbGrid,trControl = trctrl) 

Predictions_xgb <- predict(mdl1, test)
confusionMatrix(Predictions_xgb,test$Attrition)

