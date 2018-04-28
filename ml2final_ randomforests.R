rm(list = ls())
set.seed(100)
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}
installIfAbsentAndLoad(c("randomForest", "pROC", "verification", "rpart",'caret','proxy'))

#setwd("C:/Users/Kevin Deng/Desktop/Curricurlum/Spring/MachineLearning/TeamProject")
data <- read.csv('C:/Users/dpkuz/Google Drive/Machine Learing II/Final Team Project USED Cars/autos.csv', header = TRUE, stringsAsFactors = FALSE, na.strings=c(""," "))
summary(data)

# delete some useless columns
colnames(data)
deletecols <- c("name", "dateCrawled", "offerType", "abtest", "monthOfRegistration", "dateCreated", "nrOfPictures", "postalCode", "lastSeen")
data <- data[ , -which(colnames(data) %in% deletecols)]
colnames(data)

# delete the duplicated rows
data <- unique(data[ , ])

# Check the completeness of data
sapply(data, function(x) sum(is.na(x)))

# Filter out all the outliers
summary(data)
data <- data[data$yearOfRegistration <= 2017, ]
data <- data[data$yearOfRegistration >= 1950, ]
data <- data[data$price >= 100, ]
data <- data[data$price <= 150000, ]
data <- data[data$powerPS >= 10, ]
data <- data[data$powerPS <= 500,]
summary(data)
data[is.na(data)] <- "Not-declared"
sapply(data, function(x) sum(is.na(x)))

# Group the price data
data$price <- cut(data$price, breaks=c(0, 5000, 12000, 20000, Inf), labels=c("Junker","Medium", "High", "Luxury"))


#Creating Smaller Dataset to test
smaller_data <- data[c(seq(1:277860)),]
smaller_data$vehicleType <- as.factor(smaller_data$vehicleType)
smaller_data$gearbox <- as.factor(smaller_data$gearbox)
smaller_data$model <- as.factor(smaller_data$model)
smaller_data$fuelType <- as.factor(smaller_data$fuelType)
smaller_data$brand <- as.factor(smaller_data$brand)
smaller_data$notRepairedDamage <- as.factor(smaller_data$notRepairedDamage)
smaller_data <- subset( smaller_data, select = -model )


#Partition
partition <- createDataPartition(smaller_data$price, p = .75, list = FALSE)
train <- smaller_data[ partition,]
test  <- smaller_data[-partition,]


# MODEL BUILDING with Caret
#k_folds <- trainControl(method = 'cv', number = 10)
#rf <- train(price~., data = train, method="rf", trControl = k_folds)
#rf
#plot(rf)

#preds <- predict(rf, test)
#confusionMatrix(preds, test$price)


#mtry <- seq(3:7)
#Random Forest Package 
rf <- randomForest(formula = price ~ .,data=train,ntree=500, mtry=4, importance=TRUE,localImp=TRUE,replace=FALSE)
rf
importance(rf)[order(importance(rf)[,"MeanDecreaseAccuracy"], decreasing=T),]




plot(rf, main="Error Rates for Random Forest")
legend("topright", c("OOB","Junker","Medium", "High", "Luxury"), text.col=1:5, lty=1:5, col=1:5)
min.err <- min(rf$err.rate[,"OOB"])
min.err.idx <- which(rf$err.rate[,"OOB"]== min.err)
min.err.idx[1]
rf$err.rate[min.err.idx[1],]


#Build better model with number of trees equal to the minimum
better_rf <- randomForest(formula=price ~ .,data=train,ntree= min.err.idx[1], mtry=4,
                   importance=TRUE,localImp=TRUE,replace=FALSE)
better_rf
head(better_rf$votes)


#ROC Curve and AUCC
require(pROC)               #required for roc.plot
require(verification)       #required for roc.area
aucc <- roc.area(as.integer(train$price)-1,rf$votes[,2])
aucc$A
aucc$p.value
roc.plot(as.integer(train$price)-1,rf$votes[,2], main="", xlab = "False Positive Rate", ylab = "True Positive Rate")
legend("bottomright", bty="n",
       sprintf("Area Under the Curve (AUC) = %1.3f", aucc$A))
title(main="OOB ROC Curve Random Forest autos.csv")




#Evaluate using training set
prtrain <- predict(better_rf, newdata=train)
t <- table(train$price, prtrain,dnn=c("Actual", "Predicted"))
round(100* table(train$price, prtrain,dnn=c("% Actual", "% Predicted"))/length(prtrain))
train.errorrate <-(t[1,2]+t[1,3]+t[1,4]+t[2,1]+t[2,3]+t[2,4]+t[3,1]+t[3,2]+t[3,4]+t[4,1]+t[4,2]+t[4,3])/sum(t)
print(paste("The error rate when using the train set is: ", train.errorrate))


###Evaluate by scoring the test set
prtest <- predict(better_rf, newdata=test)
t <- table(test$price, prtest,dnn=c("Actual", "Predicted"))
round(100* table(test$price, prtest,dnn=c("% Actual", "% Predicted"))/length(prtest))
errorrate <-(t[1,2]+t[1,3]+t[1,4]+t[2,1]+t[2,3]+t[2,4]+t[3,1]+t[3,2]+t[3,4]+t[4,1]+t[4,2]+t[4,3])/sum(t)
print(paste("The error rate when using the test set is: ", errorrate))
print(paste("The percent correct for the test set is: ", 1-errorrate))


#using cuts at 1500, 4000, 15000
#> print(paste("The error rate when using the train set is: ", train.errorrate))
#[1] "The error rate when using the train set is:  0.226140491465808"
#> prtest <- predict(better_rf, newdata=test)
#> t <- table(test$price, prtest,dnn=c("Actual", "Predicted"))
#> round(100* table(test$price, prtest,dnn=c("% Actual", "% Predicted"))/length(prtest))
#% Predicted
#% Actual Junker Medium High Luxury
#Junker     22      5    0      0
#Medium      7     18    3      0
#High        0      6   28      1
#Luxury      0      0    3      7
#> errorrate <-(t[1,2]+t[1,3]+t[1,4]+t[2,1]+t[2,3]+t[2,4]+t[3,1]+t[3,2]+t[3,4]+t[4,1]+t[4,2]+t[4,3])/sum(t)
#> print(paste("The error rate when using the test set is: ", errorrate))
#[1] "The error rate when using the test set is:  0.260728733282467"
#> print(paste("The percent correct for the test set is: ", 1-errorrate))
#[1] "The percent correct for the test set is:  0.739271266717533"






















#using cuts at 4000, 12000, 20000
#> rf

#Call:
#  randomForest(formula = price ~ ., data = train, ntree = 500,      mtry = 4, importance = TRUE, localImp = TRUE, replace = FALSE) 
#Type of random forest: classification
#Number of trees: 500
#No. of variables tried at each split: 4

#OOB estimate of  error rate: 17.14%
#Confusion matrix:
#  Junker Medium  High Luxury class.error
#Junker 121874   5498   121    112  0.04491203
#Medium  14683  32751  2100    184  0.34126473
#High      713   7265 10859   1045  0.45382758
#Luxury    156    548  3286   7202  0.35650465
#> importance(rf)[order(importance(rf)[,"MeanDecreaseAccuracy"], decreasing=T),]
#Junker     Medium       High     Luxury MeanDecreaseAccuracy MeanDecreaseGini
#powerPS            105.381836 102.522744 143.671508 135.652366           183.147953        15766.635
#yearOfRegistration 109.086233 131.250351 154.415414 108.492841           174.552168        23137.505
#kilometer           44.085876  41.894434  49.744887  54.507186            84.784790         6780.338
#vehicleType         41.910929  30.461322  36.160920  46.458005            59.743928         5741.573
#notRepairedDamage   19.182534  59.557597  44.857158  33.437817            55.564288         1670.244
#gearbox             14.354419  10.626689  21.218105  20.411624            33.228946         1312.142
#fuelType            11.062186  17.025484  29.751059  32.013933            29.425666         1755.729
#brand                1.990101   3.021763   2.628887   4.029535             4.458702         7375.475
#> 
#  > 
#  > 
#  > 
#  > plot(rf, main="Error Rates for Random Forest")
#> legend("topright", c("OOB","Junker","Medium", "High", "Luxury"), text.col=1:5, lty=1:5, col=1:5)
#> min.err <- min(rf$err.rate[,"OOB"])
#> min.err.idx <- which(rf$err.rate[,"OOB"]== min.err)
#> min.err.idx[1]
#[1] 426
#> rf$err.rate[min.err.idx[1],]
#OOB     Junker     Medium       High     Luxury 
#0.17109171 0.04469261 0.34106360 0.45387788 0.35480701 
#> 
#  > 
#  > #Build better model with number of trees equal to the minimum
#  > better_rf <- randomForest(formula=price ~ .,data=train,ntree= min.err.idx[1], mtry=4,
#                              +                    importance=TRUE,localImp=TRUE,replace=FALSE)
#> better_rf

#Call:
#  randomForest(formula = price ~ ., data = train, ntree = min.err.idx[1],      mtry = 4, importance = TRUE, localImp = TRUE, replace = FALSE) 
#Type of random forest: classification
#Number of trees: 426
#No. of variables tried at each split: 4

#OOB estimate of  error rate: 16.97%
#Confusion matrix:
#  Junker Medium  High Luxury class.error
#Junker 121796   5571   126    112  0.04552329
#Medium  14282  33037  2201    198  0.33551229
#High      701   7042 11116   1023  0.44090132
#Luxury    154    565  3381   7092  0.36633310
#> head(better_rf$votes)
#Junker     Medium        High      Luxury
#2 0.01351351 0.18918919 0.560810811 0.236486486
#3 0.30813953 0.62790698 0.058139535 0.005813953
#4 0.98795181 0.01204819 0.000000000 0.000000000
#5 0.57419355 0.41935484 0.006451613 0.000000000
#6 0.98780488 0.01219512 0.000000000 0.000000000
#7 0.64102564 0.33974359 0.019230769 0.000000000
#> 
 # > 
#  > #ROC Curve and AUCC
#  > require(pROC)               #required for roc.plot
#> require(verification)       #required for roc.area
#> aucc <- roc.area(as.integer(train$price)-1,rf$votes[,2])
#> aucc$A
#[1] 1.268153
#> aucc$p.value
#[1] 0
#> roc.plot(as.integer(train$price)-1,rf$votes[,2], main="", xlab = "False Positive Rate", ylab = "True Positive Rate")
#Warning message:
#  In roc.plot.default(as.integer(train$price) - 1, rf$votes[, 2],  :
#                        Large amount of unique predictions used as thresholds. Consider specifying thresholds.
#                      > legend("bottomright", bty="n",
#                               +        sprintf("Area Under the Curve (AUC) = %1.3f", aucc$A))
#                      > title(main="OOB ROC Curve Random Forest autos.csv")
#                      > 
#                        > 
#                        > 
#                        > 
#                        > #Evaluate using training set
#                        > prtrain <- predict(better_rf, newdata=train)
#                      > t <- table(train$price, prtrain,dnn=c("Actual", "Predicted"))
#                      > round(100* table(train$price, prtrain,dnn=c("% Actual", "% Predicted"))/length(prtrain))
#                      % Predicted
#                      % Actual Junker Medium High Luxury
 #                     Junker     59      2    0      0
#                      Medium      6     17    1      0
##                      High        0      3    6      0
#                      Luxury      0      0    1      4
#                      > train.errorrate <-(t[1,2]+t[1,3]+t[1,4]+t[2,1]+t[2,3]+t[2,4]+t[3,1]+t[3,2]+t[3,4]+t[4,1]+t[4,2]+t[4,3])/sum(t)
#                      > print(paste("The error rate when using the train set is: ", train.errorrate))
#                      [1] "The error rate when using the train set is:  0.144205530789791"
#                      > 
#                        > 
#                        > ###Evaluate by scoring the test set
#                        > prtest <- predict(better_rf, newdata=test)
#                      > t <- table(test$price, prtest,dnn=c("Actual", "Predicted"))
#                      > round(100* table(test$price, prtest,dnn=c("% Actual", "% Predicted"))/length(prtest))
#                      % Predicted
#                      % Actual Junker Medium High Luxury
#                      Junker     58      3    0      0
#                      Medium      7     16    1      0
#                      High        0      3    5      1
#                      Luxury      0      0    2      3
#                      > errorrate <-(t[1,2]+t[1,3]+t[1,4]+t[2,1]+t[2,3]+t[2,4]+t[3,1]+t[3,2]+t[3,4]+t[4,1]+t[4,2]+t[4,3])/sum(t)
#                      > print(paste("The error rate when using the test set is: ", errorrate))
#                      [1] "The error rate when using the test set is:  0.169140405683601"
#                      > print(paste("The percent correct for the test set is: ", 1-errorrate))
#                      [1] "The percent correct for the test set is:  0.830859594316399"



