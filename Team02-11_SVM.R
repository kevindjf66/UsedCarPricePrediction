rm(list=ls())

installIfAbsentAndLoad  <-  function(neededVector) {
  if(length(neededVector) > 0) {
    for(thispackage in neededVector) {
      if(! require(thispackage, character.only = T)) {
        install.packages(thispackage)}
      require(thispackage, character.only = T)
    }
  }
}

needed <- c('e1071', 'ISLR', 'boot', 'MASS', 'splines', 'caret', 'dummies')      
installIfAbsentAndLoad(needed)


#Import the data
data <- read.csv('autos.csv', header = TRUE, stringsAsFactors = FALSE, na.strings=c(""," "))


# delete some useless columns
deletecols <- c("name", "dateCrawled", "offerType", "abtest", "monthOfRegistration", "dateCreated", 
                "nrOfPictures", "postalCode", "lastSeen", "brand", "notRepairedDamage")
data <- data[ , -which(colnames(data) %in% deletecols)]


# delete the duplicated rows
data <- unique(data[ , ])


# Check the completeness of data
sapply(data, function(x) sum(is.na(x)))


# Filter out all the outliers
#summary(data)
data <- data[data$yearOfRegistration <= 2017, ]
data <- data[data$yearOfRegistration >= 1950, ]
data <- data[data$price >= 100, ]
data <- data[data$price <= 150000, ]
data <- data[data$powerPS >= 10, ]
data <- data[data$powerPS <= 500,]
data[is.na(data)] <- "Not-declared"
sapply(data, function(x) sum(is.na(x)))


# Group the price data
summary(data$price)
sort(table(data$price),decreasing=TRUE)[1:5]
data$price <- cut(data$price, breaks=c(0, 5000, 12000, 20000, Inf),labels=c("L", "M", "H", "VH"))
summary(data$price)

# Turn to factors
data$seller <- as.factor(data$seller)
data$vehicleType <- as.factor(data$vehicleType)
data$gearbox <- as.factor(data$gearbox)
data$model <- as.factor(data$model)
data$fuelType <- as.factor(data$fuelType)
#data$brand <- as.factor(data$brand)
#data$notRepairedDamage <- as.factor(data$notRepairedDamage)


# Create correlation table
cor.data <- data
for (i in 1:ncol(cor.data)){
  cor.data[,i] <- as.numeric(cor.data[,i])
}
cortable <- cor(cor.data)


# PCA Analysis
prin_comp <- prcomp(new_data, scale. = T)
std_dev <- prin_comp$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)



set.seed(1)
# Create train and test data
training <- createDataPartition(y = data$price, p= 0.75, list=F)
train <- data[training,]; test <- data[-training,]

set.seed(1)
# Make subsets
train.new <- sample(nrow(train), 0.0889*nrow(train), replace = F)
train.s <- train[train.new,]
test.new <- sample(nrow(test), 1*nrow(test), replace=F)
test.s <- data[test.new,]


# Turn targets to numeric
train.s$price <- as.numeric(train.s$price)
test.s$price <- as.numeric(test.s$price)


###########################
##### Run the Models ######
###########################




#### Run SVM Linear Model ####
tune.lm <- tune(svm, as.factor(price) ~ ., data = train.s, kernel="linear", 
                ranges=list(cost=c(.25, .5, 1, 2, 4, 8, 18, 32, 64, 128)))
#save(tune.lm, file = "tune_lm.rda")
#load("tune_lm.rda")
bestlm <- tune.lm$best.model


lm.testpred <- as.numeric(predict(bestlm, newdata=test.s))
mean(lm.testpred == test.s$price)
mean(lm.testpred != test.s$price)

mytable <- table("actual" = test.s$price, "predicted" = lm.testpred)
round(mytable / sum(mytable) * 100,1)

aetable <- data.frame(c())
for (i in 1:nrow(mytable)) {
  a <- mytable[i,i] / sum(mytable[,i])
  tp <- mytable[i,i] / sum(mytable[i,])
  e <- 1 - a
  aetable <- rbind(aetable, c(a, tp, e))
}
colnames(aetable) = c('Accuracy', 'True Pos', 'Overall Error')

plot(tune.lm)


#### Run SVM Polynomial Model ####
tune.poly <- tune(svm, as.factor(price) ~ ., data=train.s, kernel="polynomial",
                 degree=2, ranges=list(cost=c(0.01, 0.1, 1,5, 10)))
bestpoly <- tune.poly$best.model

poly.trainpred <- as.numeric(predict(bestpoly, train.s))
poly.testpred <- as.numeric(predict(bestpoly, test.s))

mean(poly.trainpred == train.s$price)
mean(poly.testpred == test.s$price)


#### Best so far: linear
# > mean(lm.testpred == test.s$price)
# [1] 0.8424915
# > mean(lm.testpred != test.s$price)
# [1] 0.1575085

# predicted
# actual     1     2     3     4
# 1 21706  1391    86    33
# 2  1907  6231   658    46
# 3   155  1016  2870   223
# 4    43    16   328   762

# Accuracy  True Pos Overall Error
# 1 0.9115955 0.9349586    0.08840452
# 2 0.7200139 0.7047048    0.27998613
# 3 0.7280568 0.6730769    0.27194318
# 4 0.7161654 0.6631854    0.28383459










#####################
#### Other Stuff ####
#####################



# Run SVM Radial Model...
# tune.rad <- tune(svm, as.factor(price) ~ ., train.s, kernel="radial", 
#                  ranges=list(cost=c(0.01, 0.1, 1,5, 10)))
# bestrad <- tune.rad$best.model
# 
# rad.trainpred <- as.numeric(predict(bestrad, newdata=train.s))
# rad.testpred <- as.numeric(predict(bestrad, newdata=test.s))
# 
# mean(trainpred == train.s$price)
# mean(testpred == test.s$price)



# Test the best model on a random test sample
# test.new <- sample(nrow(test), 0.7*nrow(test), replace=F)
# test.data <- data[test.new,]
# test.data$price <- as.numeric(test.data$price)
# 
# final.testpred <- as.numeric(predict(bestlm, newdata=test.data))
# mean(final.testpred == test.data$price)

#0, 3500, 6500 -- .780
#0, 2500, 8000 -- .789
#0, 2000, 6000 -- .749
#0, 3500 -- .876
#0, 3000 -- .869
#0, 4000 -- .873
#0, 6000 -- .898
#0, 8000 -- .916