## Machine learning with ILSA
## Yuqi Liao
## 07/21/2018

### Ideas dump -----
# i could try to use the US restrict-use files to include more US variables!

### Set things up -----

#install.packages("EdSurvey")
devtools::load_all("U:/ESSIN Task 14/NAEP R Program/Yuqi/edsurvey")

# install.packages("modeest")
# install.packages("randomForest")
# install.packages("e1071")
# install.packages("gbm")

library(gbm)
library(dplyr)
library(modeest)
library(caret)
library(randomForest)
library(e1071)


### 1. Download/Read in -----
#downloadTIMSS(year=2015, root = "/Users/Yuqi/Desktop/Files/AIR/Machine learning/Data")
#T15_USA <- readTIMSS("G:/Conference/2019/Data/TIMSS/TIMSS2015", countries = c("usa"), gradeLvl = "4")
P15_USA <- readPIRLS("./Data/PIRLS/P16_and_PL16_Data", countries = c("usa"))

# for Trang
datapath = "G:/01-EdSurvey (start 06-19)/PIRLS"
P15_USA <- readPIRLS(file.path(datapath,2016), countries='usa',verbose=TRUE)


### 2. Select relevant variables =======
# read in everything (in the student file and the teacher file) first
student_teacher_var <- tolower(union(P15_USA$fileFormat$variableName, P15_USA$fileFormatTeacher$variableName))
P15_USA_df <- getData(data = P15_USA, varnames = student_teacher_var,
                      omittedLevels = FALSE, addAttributes = TRUE)

# calculate missing rate

missing_rule <- function(x, omittedLevels) {
  return(sapply(x, function(i) {
    ifelse(is.na(i) | i %in% omittedLevels,1,0)
  }))
}

om <- getAttributes(P15_USA_df,"omittedLevels")
missing_rate <- sapply(colnames(P15_USA_df),
                       function(i) {
                         mean(missing_rule(P15_USA_df[,i],om))
                       })
# keep variabels that has missing rate < 0.1
missing_rate_less0.1 <- missing_rate[missing_rate < 0.1]
names(missing_rate_less0.1)

# exclude variables that are weights, ids (except for itsex), pvs, (and vars taht are derived from questionnaire items), as were done in the TIMSS paper
includeVar <- grep("^jk|wgt|asbg01|^id",names(missing_rate_less0.1), value = TRUE, invert = TRUE)
# stuVarCols <- c(9, 51:115) #9 is itsex, accordingly "asbg01" (boy/girl) is excluded
# tchVarCols <- c(290:416)
# stu_tch_VarCols <- c(stuVarCols, tchVarCols)
# 
# missing_rate_less0.1_stu_tch <- names(missing_rate_less0.1[stu_tch_VarCols])

# Remove plausible values from includeVar
pvvars <- getAttributes(P15_USA,'pvvars')
pvvars <- unlist(lapply(pvvars, function(p) p$varnames))
includeVar <- includeVar[!includeVar %in% pvvars]

# check codebook of included variables
codebook <- showCodebook(P15_USA)
codebook <- codebook[tolower(codebook$variableName) %in% includeVar,]

# after looking at the codebook, exclude some variables
excludeVar <- c("version","scope","itadmini")
includeVar <- includeVar[!includeVar %in% excludeVar]

# 3. Reformat outcome variable ====== 
# define variable names for Y
read_ach_lvl <- c("rrea")

# get the df using listwise deletion of the omitted levels
P15_USA_df_stu_tch <- getData(data = P15_USA, varnames = c(includeVar, read_ach_lvl,'tchwgt'),
                                          omittedLevels = FALSE, addAttributes = TRUE)

P15_USA_df_stu_tch <- P15_USA_df_stu_tch[,~grep("^jk",colnames(P15_USA_df_stu_tch),value=TRUE)]

### Define Y, process the dataset before modelling -----
### Y would be the majority vote of asribm01-05
P15_USA_df_stu_tch <- P15_USA_df_stu_tch %>% 
  mutate(asrrea01 = as.numeric(asrrea01),
         asrrea02 = as.numeric(asrrea02),
         asrrea03 = as.numeric(asrrea03),
         asrrea04 = as.numeric(asrrea04),
         asrrea05 = as.numeric(asrrea05))

al <- as.numeric(getAttributes(P15_USA,'achievementLevels'))
P15_USA_df_stu_tch_clean <- P15_USA_df_stu_tch %>% 
  # create new variable math_ach_lvl which is the mode of all asmibm01-05
  mutate(read_ach_lvl = round((asrrea01 + asrrea02 + asrrea03 + asrrea04 +asrrea05)/5, digits = 0)) %>% 
  # create dummy version of math_ach_lvl
  mutate(read_ach_lvl_atabv4 = ifelse(read_ach_lvl >= al[3], 1, 0),
         read_ach_lvl_atabv4 = as.factor(read_ach_lvl_atabv4)) %>% 
  select(-c(asrrea01, asrrea02, asrrea03, asrrea04, asrrea05, read_ach_lvl))


# Check whether Y is balanced or not:
table(P15_USA_df_stu_tch_clean$read_ach_lvl_atabv4)
# 0    1 
# 2093 2332
# identify vars that has more than 2 levels (to turn as numeric later)
level_morethan2 <- sapply(P15_USA_df_stu_tch_clean, nlevels) > 2
# note that later i'll have to go through all the variable levels to make sure it makes sense to convert vars that has more than 2 levels into numeric (e.g. in TIMSS G4, there are two vars c("asbg06a", "asbg06b") that asks students the origin of their parents, to which one of the level is "i don't know", so for those vars, it is better to have them stay as factors.


P15_USA_df_clean <- P15_USA_df_stu_tch_clean[, level_morethan2] %>% 
  # transform some variables into numeric
  mutate_all(as.numeric) %>% 
  bind_cols(P15_USA_df_stu_tch_clean[, !level_morethan2])

# check the missing rate for each variable, just in case
colMeans(is.na(P15_USA_df_clean)) %>% sum()

# Pre-process (e.g. rescale and center) Note that here all predictors should actullay be in likert scale, though some are in 2 levels, and some oar 3, or 4 levels. I think it's best to pre-process them anyway
#yl : not working for now
# P15_USA_df_clean_processed <- preProcess(P15_USA_df_clean, 
#                               method = c("center", "scale"))
            #"nzv"

nzv <- nearZeroVar(P15_USA_df_clean, freqCut = 95/5)
P15_USA_df_clean <- P15_USA_df_clean[,-c(nzv)]


### 4. Model building - data split -----
library(caret)
set.seed(123)
trainingIndex <- createDataPartition(P15_USA_df_clean$read_ach_lvl_atabv4, p = 0.8, list = FALSE)
training <- P15_USA_df_clean[trainingIndex, ]
test <- P15_USA_df_clean[-trainingIndex, ]




### Decision Tree (CART) -----
dtree_fit <- train(read_ach_lvl_atabv4 ~., 
                   data = training, 
                   method = "rpart")
                   #trControl=trainControl(method="oob", number=25),
                   #tuneLength = 10, 
                   #parms=list(split='information')

dtree_fit

# Apply the model to the test set
test_pred <- predict(dtree_fit, test)
# See the accuracy of the model
confusionMatrix(table(test_pred , test$read_ach_lvl_atabv4 ))
                   
### AdaBoost -----
adaboost <- train(read_ach_lvl_atabv4 ~., 
                   data = training, 
                   method = "adaboost")
#trControl=trainControl(method="oob", number=25),
#tuneLength = 10, 
#parms=list(split='information')

adaboost

# Apply the model to the test set
test_pred <- predict(adaboost, test)
# See the accuracy of the model
confusionMatrix(table(test_pred , test$read_ach_lvl_atabv4 ))



### Nerual network -----

### Random Forest -----
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
