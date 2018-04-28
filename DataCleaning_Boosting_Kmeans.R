rm(list = ls())
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = TRUE) )
    { install.packages(thispackage)}
    require(thispackage, character.only = TRUE)
  }
}
needed = c('maboost', 'dummies')
require(dummies)
installIfAbsentAndLoad(needed)
setwd("C:/Users/Kevin Deng/Desktop/Curricurlum/Spring/MachineLearning/TeamProject")
data <- read.csv('autos.csv', header = TRUE, stringsAsFactors = FALSE, na.strings=c(""," "))
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

# Before PCA, make sure all of our variables should be numeric
summary(data)
new_data <- dummy.data.frame(data, names = c("vehicleType","gearbox",
                                             "model","fuelType",
                                             "brand","notRepairedDamage"))
set.seed(5072)
prin_comp <- prcomp(new_data, scale. = T)
clust2 <- kmeans(prin_comp$x, centers = 4)
new_data$price <- cut(new_data$price, breaks=c(0, 5000, 12000, 20000, Inf), labels=c(1, 2, 3, 4))
t <- table(new_data$price, clust2$cluster)
acc.rate <- (t[1,1]+t[2,2]+t[3,3]+t[4,4])/sum(t)

summary(new_data)
std_dev <- prin_comp$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
abline(h = 0.98)
abline(v = 275)

# Group the price data
data$price <- cut(data$price, breaks=c(0, 1450, 3500, 6332, 8000, Inf), labels=c("VL", "L", "M", "H","VH"))
new_data$price <- cut(new_data$price, breaks=c(0, 3500, 8000, Inf), labels=c("L", "M", "H"))
x.values <- data[,-1]
y.values <- data[,1]

# Split train and test
num = 0.1*nrow(new_data)
set.seed(5082)
train <- sample(num, 0.7*num)
test <- setdiff(c(1:num), train)

# boosting model
bm <-maboost(price~.,data=new_data[train,], iter=18, nu=1, breg="l2",
             type="sparse",bag.frac=1,random.feature=FALSE ,random.cost=FALSE, C50tree=FALSE, maxdepth=10,verbose=TRUE)
summary(bm)
# pred.bm = predict(bm, new_data[-train,], type="class")

print(pred.bm)