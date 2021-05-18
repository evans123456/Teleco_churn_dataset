rm(list=ls()) #clearing all the objects in the global environment
cat("\014") #clearing the console area





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
# install.packages("pROC")
library(pROC) # install with install.packages("pROC")

#model 
source("models.R")




#reading the dataset and assigning it to variable
churn_data<-read.csv("churn.csv") 
print(head(churn_data))




# EDA

# visual 1
#plot missing values
missmap(churn_data)

#number of people who churned (Ratio)
p<-ggplot(churn_data, aes(x=Churn))
p<-p+geom_bar(stat = "count")
p<-p+theme_classic()
p<-p+labs(title="Churn Count")
p<-p+labs(subtitle="Specifing the count of types(yes/no) of Churns")
p


#visual 2
#if people opt for phone service the tenure is more and less if vice versa
s<-ggplot(churn_data, aes(x=PhoneService, y=tenure))
s<-s+geom_bar(stat= "identity")
s<-s+theme_classic()
s<-s+labs(title="Phone service vs Tenure")
s<-s+labs(subtitle="Plotting the relationship between the Phone Service and tenure")
#s<-s+labs(subtitle="A customer with PhoneService has a longer tenure")
s


#visual 3
#internet services based on gender
a<-ggplot(churn_data, aes(x=InternetService, y = TotalCharges,fill=gender))
a<-a+geom_bar(stat = "Identity", position="dodge")
a<-a+theme_classic()
a<-a+labs(title="Total charges vs Internet Service")
a<-a+labs(subtitle="Total charges paid on the types of Internet Services based on Gender")
a

#Visual 4
#churn rates based on monthly charges
ggplot(churn_data, aes(MonthlyCharges, colour = Churn)) +geom_freqpoly(binwidth = 1) + labs(title="Churn Rates based on Monthly Charges")

#Visual 5
#churn rates based on total charges
ggplot(churn_data, aes(TotalCharges, colour = Churn)) +geom_freqpoly(binwidth = 1) + labs(title="Churn Rates based on Total Charges")


#visual 6
#churn rate based on gender
#separate male and female
female <- churn_data[ which(churn_data$gender=='Female'), ]
male <- churn_data[ which(churn_data$gender=='Male'), ]

#count gender churns
male_yes <- nrow(male[ which(male$Churn=='Yes'), ])
male_no <- nrow(male[ which(male$Churn=='No'), ])
female_yes <- nrow(male[ which(female$Churn=='Yes'), ])
female_no <- nrow(male[ which(female$Churn=='No'), ])

genderChurnRate <- data.frame(Category = c("Males who Churn", "Male who didn't churn", "Female who Churn", "Female who didn't churn"), "freq" = c(male_yes, male_no, female_yes, female_no))
ggplot(genderChurnRate, aes (x="", y = freq, fill = factor(Category))) + 
  geom_col(position = 'stack', width = 1) +
  geom_text(aes(label = paste(round(freq / sum(freq) * 100, 1), "%"), x = 1.3),
            position = position_stack(vjust = 0.5)) +
  theme_classic() +
  theme(plot.title = element_text(hjust=0.5),
        axis.line = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank()) +
  labs(fill = "Category",
       x = NULL,
       y = NULL,
       title = "Gender Churn Rate") + 
  coord_polar("y")


# visual 7
# males churn ratio
ggplot(male, aes(x="", y=Churn, fill= Churn)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y", start=0) +
  labs(title="Male churn ratio")+
  theme_void() # remove background, grid, numeric labels

#visual 8
#females churn ratio
ggplot(female, aes(x="", y=Churn, fill= Churn)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y", start=0) +
  labs(title="Female churn ratio")+
  theme_void() # remove background, grid, numeric labels


#visual 9
# monthly charges boxplot
ggplot(churn_data, aes(x="", y=MonthlyCharges)) + 
  geom_boxplot() +
  #scale_fill_viridis(discrete = TRUE, alpha=0.6) +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  #theme_ipsum() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Monthly charges boxplot") +
  xlab("")

#cleaning dataset



#checking whether the dataset has null values or not
print(is.null(churn_data))
























#checking in each columns if it has 'na' values
print(apply(churn_data, 2, function(x) any(is.na(x))))



#checking the count of churn values 
print(table(churn_data$Churn))



print(colSums(is.na(churn_data)))
#We have have 11 na values in TotalCharges 



missing <- filter(churn_data, is.na(churn_data$TotalCharges) == TRUE )
churn_data_m <- churn_data %>% mutate(TotalCharges = ifelse(is.na(churn_data$TotalCharges), churn_data$MonthlyCharges*churn_data$tenure, TotalCharges) )



print(colSums(is.na(churn_data_m)))



churn_data_m_update <-churn_data_m

#correlation matrix for numerical columns
numeric.var <- sapply(churn_data_m_update, is.numeric)
corr.matrix <- cor(churn_data_m_update[,numeric.var])
corrplot(corr.matrix, main="\n\nCorrelation Plot for after conversion Numerical Variables", method="number")




churn_data_m_update <- churn_data_m_update[complete.cases(churn_data_m_update),]
churn_data_m_update$SeniorCitizen <- as.factor(ifelse(churn_data_m_update$SeniorCitizen==1, 'YES', 'NO'))



churn_data_m_update <- data.frame(lapply(churn_data_m_update, function(x) {
  gsub("No internet service", "No", x)}))



churn_data_m_update <- data.frame(lapply(churn_data_m_update, function(x) {
  gsub("No phone service", "No", x)}))



columns <- c("tenure", "MonthlyCharges", "TotalCharges")
churn_data_m_update[columns] <- sapply(churn_data_m_update[columns], as.numeric)



churn_data_m_update_coverted <- churn_data_m_update[,c("tenure", "MonthlyCharges", "TotalCharges")]
churn_data_m_update_coverted <- data.frame(scale(churn_data_m_update_coverted))



churn_data_m_update_cat <- churn_data_m_update[,-c(1,6,19,20)]
dummy<- data.frame(sapply(churn_data_m_update_cat,function(x) data.frame(model.matrix(~x-1,data =churn_data_m_update_cat))[,-1]))



churn_data_final_cleaned <- cbind(churn_data_m_update_coverted,dummy)


set.seed(1985)
training_index<-createDataPartition(churn_data_final_cleaned$tenure,p=.7, list=FALSE,times = 1)
df<-as.data.frame(churn_data_final_cleaned)
train_dataframe<-df[training_index,]
test_dataframe<-df[-training_index,]

#models

source("models.R")

nb_accuracy <- model.NaiveBayes(train_dataframe,test_dataframe) 
nb_accuracy$overall

logist_accuracy <- model.Logistic_cv(train_dataframe,test_dataframe)
logist_accuracy$overall

svm_accuracy <-model.svm(train_dataframe,test_dataframe)
svm_accuracy$overall

knn_accuracy <-model.knn(train_dataframe,test_dataframe)
knn_accuracy

res <- data.frame(nb_accuracy$overall,logist_accuracy$overall,svm_accuracy$overall)

# visual 10
# Model accuracy scores
y <- c(nb_accuracy$overall["Accuracy"]*100,logist_accuracy$overall["Accuracy"]*100,svm_accuracy$overall["Accuracy"]*100,knn_accuracy)
x <- c("NaiveBayes","Logistic","SVM","KNN")

counts <- data.frame(cbind(x,y))
ggplot(data=counts, aes(x=x, y=y)) +
  geom_bar(stat="identity")+
  labs(title="Model Accuracy bar graph")

devtools::install_github("sachsmc/plotROC")
install.packages("plotROC")
library(plotROC)


#shiny_plotROC()

