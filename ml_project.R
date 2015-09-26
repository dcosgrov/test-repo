
# Machine Learning Course Project
library(RCurl)
library(caret)
trainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
train <- read.csv(url(trainURL), na.strings=c("NA","#DIV/0!",""))


# SPEED UP PROCESSING USING PARALLELISM
install.packages("doParallel")
library(parallel, quietly=T)
library(doParallel, quietly=T)
cluster <- makeCluster(detectCores())
registerDoParallel(cluster)

# Identify, then, remove near zero covariates to efficiently build model
nsv <- nearZeroVar(train,saveMetrics = FALSE)
train <- train[,-nsv]

# Remove unique row ids/sequences and participant names since they are not predictors as well as remove timestamp columns since *WHEN*
# the activity was performed is not a predictor of proper
# technique
train <- train[,-(1:6)]

# Only keep columns with 1% or fewer NAs
train <- train[, colMeans(is.na(train)) <= .01]

# Partition training set into train and test
# Training = 60%, Testing = 40%
inTrain <- createDataPartition(y=train$classe, p=.60, list=FALSE)
newTraining <- train[inTrain,]
newTesting <- train[-inTrain,]

# remove original train object
rm(train)

# CART Approach
set.seed(12345)
# repeated k-fold cross validation (7fold, repeat 3 times)
train_control <- trainControl(method="cv", number=7, repeats=3)
cartModel <- train(classe ~., 
                   data=newTraining, 
                   trControl=train_control, 
                   method="rpart")

cartModel$results[1,]

#Better plot
install.packages("rattle")
install.packages("rpart.plot")
library(rattle)
fancyRpartPlot(cartModel$finalModel)

cartPredictions <- predict(cartModel, newdata=newTesting)
cartPredictions
confusionMatrix(cartPredictions,newTesting$classe)

# Accuracy is <50% !

# RANDOM FORESTS APPROACH
train_control <- trainControl(method="none",verboseIter = TRUE) 
rfGrid <- expand.grid(mtry=5)
rfModel <- train(classe ~., 
                 data=newTraining, 
                 method="parRF",
                 tuneGrid=rfGrid,
                 ntree=20,
                 trControl=train_control
                 )
predictions <- predict(rfModel$finalModel, newTesting)
confusionMatrix(predictions,newTesting$classe)

# Present table of predictions (clearly shows error predicitons)
table(predictions,newTesting$classe)

#NAIVE BAYES APPROACH
install.packages("klaR")
install.packages("MASS")
library("klaR")
library("MASS")
train_control <- trainControl(method="cv",
                              number=2,
                              verboseIter = TRUE) 

nbModel <- train(classe ~.,
                 data=newTraining,
                 method="nb",
                 trControl=train_control,
                 tuneGrid = data.frame(usekernel=TRUE, fL=0)
                 )

#Accuracy is only 74% on training data, so, skipping
nbModel$results

# BAGGING APPROACH
train_control <- trainControl(method="cv",
                              number=2,
                              verboseIter = TRUE) 
bagModel <- train(classe ~.,
                  data=newTraining,
                  method="treebag",
                  trControl=train_control)
bagModel$results

predictions <- predict(bagModel$finalModel, newTesting)

#98% accuracy
confusionMatrix(predictions,newTesting$classe)

# PREPARE FINAL TESTING DATA FOR EVALUATION
# NOTE THAT TESTING DATA IS TRANSFORMED THE SAME WAY THE TRAINING
# DATA WAS TRANSFORMED
finalTestURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
finalTest <- read.csv(url(finalTestURL), na.strings=c("NA","#DIV/0!",""))
finalNsv <- nearZeroVar(finalTest,saveMetrics = FALSE)
finalTest <- finalTest[,-nsv]
finalTest <- finalTest[,-(1:6)]
finalTest <- finalTest[, colMeans(is.na(finalTest)) <= .01]
rfFinalPredictions <- predict(rfModel$finalModel, finalTest)
bagFinalPredictions <- predict(rfModel$finalModel, finalTest)
cartFinalPredictions <- predict(cartModel, finalTest)

# Turn of parallelism
stopCluster(cluster)

# Answers
answers=c("B","A","B","A","A","E","D","B","A","A","B","C","B","A","E","E","A","B","B","B")

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)

