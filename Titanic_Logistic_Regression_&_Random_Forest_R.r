df.train <- read.csv('titanic_train.csv')
head(df.train)

str(df.train)

df.train$Sex <- as.factor(df.train$Sex)
df.train$Ticket <- as.factor(df.train$Ticket)
df.train$Cabin <- as.factor(df.train$Cabin)

str(df.train)

library(Amelia)

missmap(df.train, main='Missing Map', col = c('yellow','black'), legend =F)

library(ggplot2)

ggplot(df.train, aes(Survived)) + geom_bar()

ggplot(df.train, aes(Pclass)) + geom_bar(aes(fill = factor(Pclass)))

ggplot(df.train, aes(Sex)) + geom_bar(aes(fill = factor(Sex)))

ggplot(df.train, aes(Age)) + geom_histogram(bins=20, alpha=0.5, fill='blue')

ggplot(df.train, aes(SibSp)) + geom_bar()

ggplot(df.train, aes(Fare)) + geom_histogram(fill='green', color='black', alpha=0.5)

pl1 <- ggplot(df.train, aes(Pclass, Age))
pl2 <- pl1 + geom_boxplot(aes(group=Pclass, fill=factor(Pclass), alpha=0.4))
pl3 <- pl2 + scale_y_continuous(breaks = seq(min(0),max(80), by =2))
pl3

impute_age <- function(age, class, sex){
    out <- age
    for (i in 1:length(age)){
        if (is.na(age[i])){
            if(sex[i] == 1){
               if(class[i] == 1){
                    out[i] <- 35
               }else if(class[i] == 2){
                   out[i] <- 28
               }else{
                   out[i] <- 21.5
               }
            } else{
                if(class[i] == 1){
                    out[i] <- 40
               }else if(class[i] == 2){
                   out[i] <- 30
               }else{
                   out[i] <- 25
            }
        }
    }else{
            out[i] <- age[i]
        }
}
    return(out)
}

fixed.ages <- impute_age(df.train$Age, df.train$Pclass, df.train$Sex)

df.train$Age <- fixed.ages

missmap(df.train, main = 'Imputation Check', col =c('yellow','black'), legend=F)

str(df.train)

library(dplyr)
df.train <- select(df.train, -PassengerId,-Name,-Ticket,-Cabin)
head(df.train)

str(df.train)

df.train$Survived <- as.factor(df.train$Survived)
df.train$Pclass <- as.factor(df.train$Pclass)
df.train$SibSp <- as.factor(df.train$SibSp)

str(df.train)

log.model <- glm(Survived ~., family=binomial(link='logit'), data = df.train)

summary(log.model)

library(caTools)

set.seed(101)
split <- sample.split(df.train$Survived, SplitRatio=0.7)
final.train <- subset(df.train, split == T)
final.test <- subset(df.train, split == F)

final.log.model <- glm(Survived ~.,family=binomial(link='logit'), data=final.train)

summary(final.log.model)

fitted.probabilities <- predict(final.log.model, final.test, type='response')
fitted.results <- ifelse(fitted.probabilities>=0.5,1,0)

misClassError <- mean(fitted.results != final.test$Survived)
print(1-misClassError)

table(final.test$Survived, fitted.probabilities>=0.5)

library(randomForest)

rf.model <- randomForest(Survived ~., data = final.train)

rf.predictions <- predict(rf.model, final.test)

rf.misClassError <- mean(rf.predictions != final.test$Survived)
print(1-rf.misClassError)

table(final.test$Survived, rf.predictions)

randfor.model <- randomForest(Survived ~., data = df.train)

df.test <- read.csv('titanic_test.csv')
df.test2 <- df.test

head(df.test)

str(df.test)

df.test$Sex <- as.factor(df.test$Sex)
df.test$Ticket <- as.factor(df.test$Ticket)
df.test$Cabin <- as.factor(df.test$Cabin)
df.test$Pclass <- as.factor(df.test$Pclass)
df.test$SibSp <- as.factor(df.test$SibSp)

str(df.test)

PID <- df.test$PassengerID

missmap(df.test, main='Missing Map', col = c('yellow','black'), legend =F)

fixed.test_ages <- impute_age(df.test$Age, df.test$Pclass, df.test$Sex)

df.test$Age <- fixed.test_ages

missmap(df.test, main = 'Imputation Check', col =c('yellow','black'), legend=F)

df.test <- select(df.test, -PassengerId,-Name,-Ticket,-Cabin)

rf.pred <- predict(randfor.model, df.test)

lr.pred_fit <- predict(log.model, df.test)
lr.pred <- ifelse(lr.pred_fit>=0.5,1,0)

str(rf.pred)

str(lr.pred)

output.lr <- data.frame(PassengerID = df.test2$PassengerId, Survived = lr.pred)
output.rf <- data.frame(PassengerID = df.test2$PassengerId, Survived = rf.pred)

head(output.lr)

head(output.rf)

write.csv(output.lr, 'Titanic_LR.csv', row.names=F)
write.csv(output.rf, 'Titanic_RF.csv', row.names=F)


