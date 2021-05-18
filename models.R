#Loading required packages
#library(tidyverse)
library(caretEnsemble)
library(psych)
library(Amelia)
library(mice)
library(GGally)
library(rpart)
library(magrittr)
library(dplyr)
library(ggplot2)
library(dummies)
library(CatEncoders)
library(keras)
library(rsample)
library(recipes)
library(caret)
library(randomForest)
library(corrplot)
library(gridExtra)
library(MASS)
#library(car)
library(e1071)
library(class)
library(caret)

accuracy <- function(x){
  sum(diag(x)/(sum(rowSums(x)))) * 100
}


model.NaiveBayes <- function(train_dataframe,test_dataframe) 
{
  y <- train_dataframe$Churn
  Y <- as.factor(y)
  myvars <- c("gender", "SeniorCitizen", "Partner","Dependents","tenure","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","MonthlyCharges")
  x <- train_dataframe[myvars]
  print("x y sorted")
  
  # train_dataframe$Churn <- factor(train_dataframe$Churn, levels= c("Yes","No"), labels = c(1,0))
  model = train(x,Y,'nb',trControl=trainControl(method='cv',number=10))
  
  print("after model")
  
  predictions <- predict(model, newdata = test_dataframe)
  print("after predictions",predictions)
  
  #now lets create a confusion matrix
  cm<-confusionMatrix(data=predictions, as.factor(test_dataframe$Churn),dnn = c("Prediction", "Actual"))
  print(cm)
  
  # Probability_pred <- predict(model, newdata = test_dataframe, type = "prob")
  # print("Here:   ",Probability_pred)
  
 

  # roc_qda <- roc(test_dataframe$Churn,Probability_pred[,1])
  
  # roc_qda <- roc(response = test_dataframe$Churn, predictor = Probability_pred[,1])
  # plot(roc_qda, col="red", lwd=3, main="ROC curve QDA")
  # auc(roc_qda)
  
  print("after roc")
return(cm)
}




model.Logistic_cv <- function(train_dataframe,test_dataframe) 
{
  train_dataframe$Churn[train_dataframe$Churn==1]<-"yes"
  train_dataframe$Churn[train_dataframe$Churn==0]<-"no"
  test_dataframe$Churn[test_dataframe$Churn==1]<-"yes"
  test_dataframe$Churn[test_dataframe$Churn==0]<-"no"
  
  #Converting them to factor data type
  train_dataframe$Churn<-as.factor(train_dataframe$Churn)
  test_dataframe$Churn<-as.factor(test_dataframe$Churn)
  
  #Now we have to define the # of fold
  ctrlspecs<-trainControl(method="cv",number=10,
                          savePredictions = "all", classProbs = TRUE)#here we have define the method 'CV' means cross validation, number=10 means no of folds, ClassProbs=true means save probality with prediction
  
  #set random seed
  set.seed(1985)
  # specify the logistic regression model
  model_1<-train(Churn ~ tenure + Partner + InternetService.xFiber.optic + InternetService.xNo +
                   OnlineSecurity + MonthlyCharges + SeniorCitizen+
                   StreamingTV + Contract.xTwo.year + PaperlessBilling+
                   PaymentMethod.xElectronic.check + OnlineBackup + TechSupport +Contract.xOne.year + Contract.xTwo.year,family = "binomial", data = train_dataframe,method="glm",trControl=ctrlspecs )
  
  
  print(model_1)
  
  #output in terms of regression coefficients
  summary(model_1)

  #Variable importance of the predictor variables
  y <- varImp(model_1)
  plot(y)
  ###now lets apply our modle to test_df data frame
  
  #predict the outcome by using the model train_df on test_df
  predictions <- predict(model_1, newdata = test_dataframe)
  
  #now lets create a confusion matrix
  cm1<-confusionMatrix(data=predictions, factor(test_dataframe$Churn),dnn = c("Prediction", "Actual"))
  # print(cm1)
  
  #cm1 <- confusionMatrix(factor(predictions), factor(test_dataframe$Churn), dnn = c("Prediction", "Actual"))
  # print("CM1",cm1)
  
  # ggplot(as.data.frame(cm1$table), aes(Prediction,sort(Actual,decreasing = T), fill= Freq)) +
  #   geom_tile() + geom_text(aes(label=Freq))
  # scale_fill_gradient(low="white", high="#009194")
  # labs(x = "Actual",y = "Prediction") +
  #   scale_x_discrete(labels=c("Class_1","Class_2","Class_3","Class_4")) +
  #   scale_y_discrete(labels=c("Class_4","Class_3","Class_2","Class_1"))
  # 
 return (cm1)
}





model.svm <- function(train_dataframe,test_dataframe){
  y <- train_dataframe$Churn
  Y <- as.factor(y)
  print("after x y")
  
  svm_m = svm(formula = Churn ~ tenure + MonthlyCharges + SeniorCitizen + 
                Partner + InternetService.xFiber.optic + InternetService.xNo + 
                OnlineSecurity + OnlineBackup + TechSupport + 
                StreamingTV + Contract.xOne.year + Contract.xTwo.year + PaperlessBilling+
                PaymentMethod.xElectronic.check,
              data = train_dataframe,
              type = 'C-classification',
              kernel = 'linear',
              ernal = "radial", gamma = 0.1, cost = 10, probability = TRUE)
  print("after svm")
  #predict(svm_m, data = train_dataframe, type = "response") -> train_pred
  #predict(svm_m, newdata = test_dataframe, type = "response") -> test_pred

  print(summary(svm_m))

  # Predicting the Test set results
  y_pred = predict(svm_m, newdata = test_dataframe)
  print(summary(y_pred))
  print("after ypred")

  test_dataframe$Churn<-as.factor(test_dataframe$Churn)
  print("after testdf")
  #Building a Confusion Matrix
  c_mat <- confusionMatrix(data=y_pred, reference=test_dataframe$Churn)
  # print(cmat)

  return (c_mat)
}


model.knn <- function(train_dataframe,test_dataframe){
  dim(train_dataframe)
  dim(test_dataframe)
  dim(train_dataframe$Churn)
  
  pr <- knn(train_dataframe,test_dataframe,cl=train_dataframe$Churn,k=2)
  tab <- table(pr,test_dataframe$Churn)
  
  acc <- accuracy(tab)
  print(acc)
  #print(tab)
  #print(tab)
  
  return (acc)
}

