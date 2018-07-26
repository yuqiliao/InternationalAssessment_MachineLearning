## Machine learning with ILSA
## Yuqi Liao
## 07/21/2018

### Ideas dump -----
# i could try to use the US restrict-use files to include more US variables!

### Set things up -----

install.packages("EdSurvey")
install.packages("modeest")
install.packages("randomForest")
install.packages("e1071")
install.packages("gbm")
library(gbm)
library(dplyr)
library(EdSurvey)
library(modeest)
library(caret)
library(randomForest)
library(e1071)




### Download/Read in -----
setwd("/Users/Yuqi/Desktop/Files/AIR/Machine learning/Data")
#downloadTIMSS(year=2015, root = "/Users/Yuqi/Desktop/Files/AIR/Machine learning/Data")
T15_USA <- readTIMSS("./TIMSS2015", countries = c("usa"), gradeLvl = "4")

### calculate missing rate and decide on which variables to use
# read in everything first
T15_USA_df <- getData(data = T15_USA, varnames = colnames(T15_USA),
                      omittedLevels = FALSE, addAttributes = TRUE)

# calculate missing rate
missing_rate <- colMeans(is.na(T15_USA_df)) 

# keep variabels that has missing rate < 0.1
missing_rate_less0.1 <- missing_rate[missing_rate < 0.1]
names(missing_rate_less0.1)

# keep only student variables for now
stuVarCols <- c(9, 99:186)
missing_rate_less0.1_stu <- missing_rate_less0.1[stuVarCols]
names_missing_rate_less0.1_stu <- names(missing_rate_less0.1_stu)[-2] #[-2] is to exclude "asbg01" (boy/girl) because "itsex" is included

# define variable names for Y
math_ach_lvl <- c("asmibm01", "asmibm02", "asmibm03", "asmibm04", "asmibm05")

# get the df using listwise deletion of the omitted levels
T15_USA_df_stu_omittedIncluded <- getData(data = T15_USA, varnames = c(names_missing_rate_less0.1_stu, math_ach_lvl),
                          omittedLevels = FALSE, addAttributes = TRUE)
T15_USA_df_stu <- getData(data = T15_USA, varnames = c(names_missing_rate_less0.1_stu, math_ach_lvl),
                                          omittedLevels = TRUE, addAttributes = TRUE)

### Define Y -----
### Y would be the majority vote of ASMIBM01-05
T15_USA_df_stu_clean <- T15_USA_df_stu %>% 
  #select(idcntry, idstud, asmibm01, asmibm02, asmibm03, asmibm04, asmibm05) %>% 
  #filter(idstud %in% c("10101", "10102", "10103", "10104")) %>% 
  # create new variable math_ach_lvl which is the mode of all asmibm01-05
  rowwise() %>% 
  summarize(math_ach_lvl = round((asmibm01 + asmibm02 + asmibm03 + asmibm04 +asmibm05)/5, digits = 0)) %>% 
  ungroup() %>% 
  bind_cols(T15_USA_df_stu) %>% 
  # create dummy version of math_ach_lvl
  mutate(math_ach_lvl_atabv4 = ifelse(math_ach_lvl %in% c(4,5), 1, 0),
         math_ach_lvl_atabv4 = as.factor(math_ach_lvl_atabv4)) %>% 
  select(-c("asmibm01", "asmibm02", "asmibm03", "asmibm04", "asmibm05", "math_ach_lvl"))

# identify vars that has more than 2 levels
level_morethan2 <- sapply(T15_USA_df_stu_clean, nlevels) > 2
level_morethan2[12:13] <- c(FALSE, FALSE) ##make it FALSE for two variables c("asbg06a", "asbg06b"), because they are better to stay as factors


T15_USA_df_stu_clean2 <- T15_USA_df_stu_clean[, level_morethan2] %>% 
  # transform some variables into numeric
  mutate_all(as.numeric) %>% 
  bind_cols(T15_USA_df_stu_clean[, !level_morethan2])

# check the missing rate for each variable, just in case
colMeans(is.na(T15_USA_df_stu_clean)) %>% sum()


### Create traing and test data sets -----
set.seed(123)
testIndex <- createDataPartition(T15_USA_df_stu_clean2$math_ach_lvl_atabv4, p = 0.7, list = FALSE)
test <- T15_USA_df_stu_clean2[testIndex, ]
training <- T15_USA_df_stu_clean2[-testIndex, ]


### Random Forest (there are missing values)-----
set.seed(123)
mtry <- sqrt(ncol(T15_USA_df_stu_clean2) - 1) 
#mtry <- 16 #could do fine tuning to find the optimal mtry value
rf <- train(math_ach_lvl_atabv4~., 
             data=training, 
             method="rf", 
             metric="Accuracy", 
             tuneGrid=expand.grid(.mtry=mtry), 
             trControl=trainControl(method="oob", number=25),
             #default to be 500
             ntree = 500)
print(rf)


# Apply the model to the test set
test_pred <- predict(rf, test)
# See the accuracy of the model
table(test_pred, test$math_ach_lvl_atabv4)/nrow(test)
# Use confusionMatrix
confusionMatrix(table(test_pred , test$math_ach_lvl_atabv4 ))


# Variable importance
# top 20 most important variables
varImp(rf) #"scale = TRUE" is the default
top20_plot <- plot(varImp(rf), top = 20)
# get the names of the top 20 variables
top20_varName <- as.character(top20_plot$panel.args[[1]]$y)


# Create new RF models with the top variables
data_top20 <- T15_USA_df_stu_clean %>% 
  select(CHRONIC_ABSENTEE,
         top20_varName)
set.seed(123)
testIndex <- createDataPartition(data_top20$CHRONIC_ABSENTEE, p = 0.8, list = FALSE)
training <- data_top20[-testIndex, ]
test <- data_top20[testIndex, ]

# Create model
mtry <- sqrt(ncol(data_top10) - 1) 
#mtry <- 16 #could do fine tuning to find the optimal mtry value
rf_CHRONIC_ABSENTEE_top10 <- train(CHRONIC_ABSENTEE~., 
                                   data=training, 
                                   method="rf", 
                                   metric="Accuracy", 
                                   tuneGrid=expand.grid(.mtry=mtry), 
                                   trControl=trainControl(method="oob", number=25),
                                   #default to be 500
                                   ntree = 1000)
print(rf_CHRONIC_ABSENTEE_top10)

# Apply the model to the test set
test_pred <- predict(rf_CHRONIC_ABSENTEE_top10, test)
# Use confusionMatrix
confusion_matrix_top10 <- confusionMatrix(table(test_pred , test$CHRONIC_ABSENTEE ))




############## GBM (Gradient Boosted Machines) /XGBoost ##############
#build a GBM model (Y = math_ach_lvl_atabv4)

# Create traing and test data sets
set.seed(123)
testIndex <- createDataPartition(T15_USA_df_stu_clean2$math_ach_lvl_atabv4, p = 0.7, list = FALSE)
test <- T15_USA_df_stu_clean2[testIndex, ]
training <- T15_USA_df_stu_clean2[-testIndex, ]

# Create model
#mtry <- sqrt(ncol(data) - 1) 
#mtry <- 16 #could do fine tuning to find the optimal mtry value
gbm <- train(math_ach_lvl_atabv4~., 
              data=training, 
              method="gbm", 
              #metric="Accuracy", 
              trControl=trainControl(method="cv", number=3))
#interaction.depth = 1,
#shrinkage = 0.1,
#n.minobsinnode = 10,
#n.trees = 50)
print(gbm)

# Apply the model to the test set
test_pred <- predict(gbm, test)
# See the accuracy of the model
table(test_pred, test$math_ach_lvl_atabv4)/nrow(test)
# Use confusionMatrix
confusion_matrix_gbm <- confusionMatrix(table(test_pred , test$math_ach_lvl_atabv4 ))



### Step 3
# Variable importance
# top 20 most important variables
varImp(object = gbm) #"scale = TRUE" is the default
gbm_top20_plot <- plot(varImp(gbm), top = 20)
# get the names of the top 20 variables
gbm_top20_varName <- as.character(gbm_top20_plot$panel.args[[1]]$y)



# Create new GBM models with the top variables
gbm_data_top20 <- data %>% 
  select(CHRONIC_ABSENTEE,
         gbm_top20_varName)
set.seed(123)
testIndex <- createDataPartition(gbm_data_top20$CHRONIC_ABSENTEE, p = 0.8, list = FALSE)
training <- gbm_data_top20[-testIndex, ]
test <- gbm_data_top20[testIndex, ]

# Create model
gbm_CHRONIC_ABSENTEE_top20 <- train(CHRONIC_ABSENTEE~ch_present+ch_lifecon, 
                                    data=training, 
                                    method="gbm", 
                                    #metric="Accuracy", 
                                    trControl=trainControl(method="cv", number=3))
print(gbm_CHRONIC_ABSENTEE_top20)

# Apply the model to the test set
gbm_test_pred_top20 <- predict(gbm_CHRONIC_ABSENTEE_top20, test)
# Use confusionMatrix
confusion_matrix_gbm_top20 <- confusionMatrix(table(gbm_test_pred_top20 , test$CHRONIC_ABSENTEE ))



############## Principal Components Analysis ##############
###        0. create the dataset (no NAs, no ID variables)
### Steps: 1. standardize and center
###        2. PCA
###        3. Visualize

# PCA requires that all variables are quantitative (no factors)
T15_USA_df_stu_pca <- T15_USA_df_stu_clean2 %>% 
  mutate_all(as.numeric)

### PCA with function prcomp(), from the "stats" package
pca1 = prcomp(T15_USA_df_stu_pca, center = TRUE, scale. = TRUE)
# sqrt of eigenvalues
pca1$sdev
# loadings
head(pca1$rotation)
# PCs (aka scores)
head(pca1$x)
# Summary
plot(pca1, type = "l")
summary(pca1)


### PCA with function PCA(), from the "FactoMineR" package
install.packages("FactoMineR")
library(FactoMineR)
# apply PCA
pca2 <- PCA(T15_USA_df_stu_pca, graph = FALSE)
# matrix with eigenvalues
pca2$eig
# PCs (aka scores)
head(pca2$ind$coord)

### PCA with function preProcess(), from the "caret" package
trans <- preProcess(T15_USA_df_stu_pca, 
                    method=c("pca"))
PC = predict(trans, T15_USA_df_stu_pca)



### Visualize

# create data frame with scores
scores = as.data.frame(pca1$x)
# Plot - Plot of observations
ggplot(data = scores, aes(x = PC1, y = PC2, label = rownames(scores))) +
  geom_hline(yintercept = 0, colour = "gray65") +
  geom_vline(xintercept = 0, colour = "gray65") +
  geom_text(colour = "tomato", alpha = 0.8, size = 4) +
  ggtitle("PCA plot")

# Plot - biplot 
biplot(pca1)

# Plot - ggbiplot 
library(devtools)
install_github("ggbiplot", "vqv")
library(ggbiplot)
ggbiplot(pca1, obs.scale = 1, var.scale = 1, 
         ellipse = TRUE, 
         circle = TRUE) + 
  scale_color_discrete(name = '') + 
  theme(legend.direction = 'horizontal', 
        legend.position = 'top')




############## LASSO & Elastic Net regressions ##############
## Reference: https://www.slideshare.net/ShangxuanZhang/ridge-regression-lasso-and-elastic-net
## http://www.sthda.com/english/articles/36-classification-methods-essentials/149-penalized-logistic-regression-essentials-in-r-ridge-lasso-and-elastic-net/

### Prepare for a data frame/matrix
# Inspect the data
glimpse(T15_USA_df_stu_clean2)

### Create traing and test data sets -----
set.seed(123)
testIndex <- createDataPartition(T15_USA_df_stu_clean2$math_ach_lvl_atabv4, p = 0.7, list = FALSE)
test <- T15_USA_df_stu_clean2[testIndex, ]
training <- T15_USA_df_stu_clean2[-testIndex, ]

# Use model.matrix() to create the matrix of predictors (required by glmnet())
x <- model.matrix(math_ach_lvl_atabv4~., training)[,-1] #[,-1] is to get rid of the intercept column
# Convert the outcome variable to a numerical variable (required by glmnet())
y <- ifelse(training$math_ach_lvl_atabv4 == 1, 1, 0)

### LASSO
library(glmnet)
# Find the best lambda using cross-validation
set.seed(123) 
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)
cv.lasso$lambda.min
cv.lasso$lambda.1se
coef(cv.lasso, cv.lasso$lambda.min)
coef(cv.lasso, cv.lasso$lambda.1se)



# Final model with lambda.min
lasso.model <- glmnet(x, y, alpha = 1, family = "binomial",
                      lambda = cv.lasso$lambda.min)
# Make prediction on test data
x.test <- model.matrix(math_ach_lvl_atabv4 ~., test)[,-1]
probabilities <- lasso.model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
observed.classes <- test$math_ach_lvl_atabv4
mean(predicted.classes == observed.classes)


# Final model with lambda.1se
lasso.model <- glmnet(x, y, alpha = 1, family = "binomial",
                      lambda = cv.lasso$lambda.1se)
# Make prediction on test data
x.test <- model.matrix(math_ach_lvl_atabv4 ~., test)[,-1]
probabilities <- lasso.model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
observed.classes <- test$math_ach_lvl_atabv4
mean(predicted.classes == observed.classes)


### Elastic Net model
elasticnet.model <- glmnet(x, y, alpha = 0.5, family = "binomial",
                           lambda = cv.lasso$lambda.1se)
probabilities <- elasticnet.model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy rate
observed.classes <- test$math_ach_lvl_atabv4
mean(predicted.classes == observed.classes)
elasticnet.model$beta



rownames(coef(cv.lasso, cv.lasso$lambda.1se))[2]




####### explore

featurePlot(x, y)
plotClassProbs(object = rf )
