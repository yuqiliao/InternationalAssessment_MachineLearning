## Machine learning with ILSA
## Yuqi Liao & Trang Nguyen
## 08/29/2018

### Ideas dump -----
# we could try to use the US restrict-use files to include more US variables!
# focus on the US dataset for now, but could extend the scope to analyze datasets for all countries and do a comparison
# could add the cluster analysis in the future
# could add interaction terms in the future

### Tentative research quesitons
# RQ1: Could machine learning algorithms be better than logistic regressions in predicting student outcome?
# RQ2: What variables could best predict student outcome (achieving the “high” reading proficiency level, or NOT achieving the proficiency level)?
# #RQ3: Among students with low proficiency levels, what are the characteristics?

## paraphrased below
# the analysis could show how ML could be applied in ILSA data. in particular, the goals are 
# 1. to create a model which could successfully identify/predict low-performing students using information from STUDENT, Teacher, and Principal questionanires (focus on prediction accuracy/recall or a specific metric)
# 2. identify variables that have the most predictive power (so the policy implication would be that school should focus more on those variables to help low-performing students)
# 3. within low-performing students, cluster them to see if there are some usual patterns


### Set things up -----

#install.packages("EdSurvey")
#install.packages("modeest")
#install.packages("randomForest")
#install.packages("e1071")
#install.packages("gbm")
devtools::load_all("U:/ESSIN Task 14/NAEP R Program/Yuqi/edsurvey")
library(gbm)
library(dplyr)
library(modeest)
library(caret)
library(randomForest)
library(e1071)
library(extraTrees)
library(rJava)
library(xgboost)

### Step 1. Read in and prepare data ==============
### 1a. Download/Read in/Create a data frame for modeling -----

### Download data
#downloadTIMSS(year=2015, root = "/Users/Yuqi/Desktop/Files/AIR/Machine learning/Data")
#T15_USA_G4 <- readTIMSS("G:/Conference/2019/Data/TIMSS/TIMSS2015", countries = c("usa"), gradeLvl = "4")
T15_USA_G8 <- readTIMSS("G:/Conference/2019/Data/TIMSS/TIMSS2015", countries = c("usa"), gradeLvl = "8")

### Read in (1st time) to find out missing rate
# calculate missing rate and decide on which variables to use
# read in everything (in the student file and the teacher file) first
#student_school_var_G4 <- tolower(union(T15_USA_G4$fileFormat$variableName, T15_USA_G4$fileFormatSchool$variableName))
student_school_var_G8 <- tolower(union(T15_USA_G8$fileFormat$variableName, T15_USA_G8$fileFormatSchool$variableName))

# T15_USA_G4_df <- getData(data = T15_USA_G4, varnames = student_school_var_G4,
#                       omittedLevels = FALSE, addAttributes = TRUE)
T15_USA_G8_df <- getData(data = T15_USA_G8, varnames = student_school_var_G8,
                         omittedLevels = FALSE, addAttributes = TRUE)

# Use G8 data as a start
#T15_USA_df <- T15_USA_G4_df
T15_USA_df <- T15_USA_G8_df


# 1b. Remove some irrelevant variables (i.e. weight, PVs, math and science items)
# for the derived scale variables, drop the item-level variables that the scale variables are derived from
  #Home Educational Resources/SCL
  #Students Sense of School Belonging/SCL:bsbg15a-g,
  #Student Bullying/SCL:bsbg16a-i
  #Students Like Learning Mathematics/SCL:bsbm17a-i
  #Engaging Teaching in Math Lessons/SCL:bsbm18a-j
  #Student Confident in Mathematics/SCL:bsbm19a-i
  #Students Value Mathematics/SCL:bsbm20a-i
  #Students Like Learning Science/SCL:bsbs21a-i
  #Engaging Teaching in Science Lessons/SCL:bsbs22-
  #Student Confident in Sciences/SCL
  #...
  #Instr Aff by Mat Res Shortage-Prncpl/SCL:bcbg13a-, b-
  #Instr Aff by Sci Res Shortage-Prncpl/SCL: bcbg13a-, c-
  #School Emph on Acad Success-Prncpl/SCL:bcbg14-
  #School Discipline Problems-Prncpl/SCL:bcbg15-
  
includeVar <- grep("bsbg04|bsdg06s|^bsbg15|^bsbg16|^bsbm17|^bsbm18|^bsbm19|^bsbm20|^bsbs21|^bsbs22|^bsbs23|^bsbs24|^bsbb22|^bsbb23|^bsbb24|^bsbe26|^bsbe27|^bsbe28|^bsbc30|^bsbc31|^bsbc32|^bsbp34|^bsbp35|^bsbp36|^bcbg13|^bcbg14|^bcbg15", names(T15_USA_df), value = TRUE, invert = TRUE)


# Remove weights, index version of the derived variables, and cognitive items
includeVar <- grep("^jk|wgt|^id|^ita|^bsdg|^bcdg|^m0|^s0|^si|^sr|^mi|^mr",includeVar, value = TRUE, invert = TRUE)
# add back 
includeVar <- c(includeVar, "bsdg06s", "bsdgedup", "bcdg03", "bcdg07hy")

# Remove plausible values from includeVar
pvvars <- getAttributes(T15_USA_df,'pvvars')
pvvars <- unlist(lapply(pvvars, function(p) p$varnames))
includeVar <- includeVar[!includeVar %in% pvvars]

# check codebook of included variables
codebook <- showCodebook(T15_USA_df)
codebook[tolower(codebook$variableName) %in% includeVar,] %>% View()


# after looking at the codebook, exclude some more variables
excludeVar <- c("ilreliab","version","bsbg01", "itlang", "bsdmlowp", "bsdslowp") #exclude "itlang" cuz the US only has one level
includeVar <- includeVar[!includeVar %in% excludeVar]


# 1c. calculate missing rate and remove variables that has more than 10% missing
#original method
missing_rate <- colMeans(is.na(T15_USA_df)) 

#method inspired by Trang
missing_rule <- function(x, omittedLevels) {
  return(sapply(x, function(i) {
    ifelse(is.na(i) | i %in% omittedLevels,1,0)
  }))
}
om <- getAttributes(T15_USA_df,"omittedLevels")

missing_rate_new <- colMeans(missing_rule(T15_USA_df,om))

summary(missing_rate)
summary(missing_rate_new) 

# keep variabels that has missing rate < 0.1
missing_rate_less0.1 <- missing_rate[missing_rate < 0.15]
names(missing_rate_less0.1)

includeVar <- includeVar[includeVar %in% names(missing_rate_less0.1)]



### Read in (2nd time) to have all right-hand-site variables needed for the model=====
# Reformat outcome variable # define variable names for Y
math_ach_lvl <- c("mmat")

# get the df using listwise deletion of the omitted levels
T15_USA_df_stu_sch <- getData(data = T15_USA_df, varnames = c(includeVar, math_ach_lvl),
                              omittedLevels = TRUE, addAttributes = TRUE)


# Step 2. Prepare data for model fitting ======
# 2a. Create dependent variable
# first get the average of bsmmat01-05, and create a new variable to document the proficiency level of that average
al <- as.numeric(getAttributes(T15_USA_df,'achievementLevels'))
T15_USA_df_stu_sch_clean <- T15_USA_df_stu_sch %>%
  mutate(bsmmat_mean = (bsmmat01 + bsmmat02 + bsmmat03 + bsmmat04 + bsmmat05)/5,
         math_ach_lvl_blwhigh = ifelse(bsmmat_mean < al[2], "low", "not_low"),
         math_ach_lvl_blwhigh = as.factor(math_ach_lvl_blwhigh)) %>% 
  select(-c(bsmmat01, bsmmat02, bsmmat03, bsmmat04, bsmmat05, bsmmat_mean))


#check the balance of Y ##YL: I expect it to not be balanced, because we are "predicting" the low-performing students
table(T15_USA_df_stu_sch_clean$math_ach_lvl_blwhigh)


# 2b. Scaling independent variables
# identify vars that has more than 2 levels (to turn as numeric later)
level_morethan2 <- sapply(T15_USA_df_stu_sch_clean, nlevels) > 2
# note that later i'll have to go through all the variable levels to make sure it makes sense to convert vars that has more than 2 levels into numeric (e.g. in TIMSS G8, there are two vars c("bsbg09a", "bsbg09b") that asks students the origin of their parents, to which one of the level is "i don't know", so for those vars, it is better to have them stay as factors.
level_morethan2["bsbg09a"] <- FALSE
level_morethan2["bsbg09b"] <- FALSE


T15_USA_df_clean <- T15_USA_df_stu_sch_clean[, level_morethan2] %>%
  # transform some variables into numeric
  mutate_all(as.double) %>%
  bind_cols(T15_USA_df_stu_sch_clean[, !level_morethan2])

# check the missing rate for each variable, just in case
colMeans(missing_rule(T15_USA_df_clean,om)) %>% sum()


# 2d. Prepocess data
# identify and remove columns with near zero variance (otherwise the models later won't run on this constant terms)
nzv <- nearZeroVar(T15_USA_df_clean, freqCut = 97/3)
T15_USA_df_clean <- T15_USA_df_clean[,-c(nzv)]

process_configuration <- preProcess(T15_USA_df_clean,
                                    method = c("center", "scale"))
processed <- predict(process_configuration, T15_USA_df_clean)

# Step 3. Feature engineering =====
# 3a. Correlational matrix
# Goal: Idenitfy pairs of highly correlated variables
is_factor <- sapply(processed, is.factor)


correlationMatrix <- cor(processed[, !is_factor])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.90)
processed_dbl_removeHighlyCorr <- names(processed[, !is_factor][highlyCorrelated])

processed <- processed[ , !(names(processed) %in% processed_dbl_removeHighlyCorr)]


### Step 4. Model building ===========
# 4a. data split -----
set.seed(123)
trainingIndex <- createDataPartition(processed$math_ach_lvl_blwhigh, p = 0.8, list = FALSE)
training <- processed[trainingIndex, ]
test <- processed[-trainingIndex, ]


### Decision Tree (CART) -----
dtree <- train(math_ach_lvl_blwhigh ~., 
                   data = training, 
                   method = "rpart",
                   metric = "Sens",
                   trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
                   tuneLength = 20)

dtree

# Apply the model to the test set
test_pred_dtree <- predict(dtree, test)
# See the accuracy of the model
result_dtree <- confusionMatrix(table(test_pred_dtree , test$math_ach_lvl_blwhigh ))




### Random Forest -----
# try using tuneLength #mtry = 83 is used in the end
rf <- train(math_ach_lvl_blwhigh~., 
             data=training, 
             method="rf", 
             metric="Accuracy", 
             #tuneGrid=expand.grid(.mtry=mtry), 
             trControl=trainControl(method="oob", number=25),
             #default to be 500
             ntree = 500,
             tuneLength = 5)
print(rf)

test_pred_rf <- predict(rf, test)
result_rf <- confusionMatrix(table(test_pred_rf , test$math_ach_lvl_blwhigh ))

# try to use Sens as the metric
rf_2 <- train(math_ach_lvl_blwhigh~., 
            data=training, 
            method="rf", 
            metric = "Sens",
            #tuneGrid=expand.grid(.mtry=mtry), 
            trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
            #default to be 500
            ntree = 1000,
            tuneLength = 10)
print(rf_2)

test_pred_rf_2 <- predict(rf_2, test)
result_rf_2 <- confusionMatrix(table(test_pred_rf_2 , test$math_ach_lvl_blwhigh ))


# # try using tuneGrid
# mtry <- sqrt(ncol(training) - 1)
# rfGrid <- expand.grid(mtry = c(5,  13 , 18, 25, 30))
# 
# rf2 <- train(math_ach_lvl_blwhigh~., 
#             data=training, 
#             method="rf", 
#             metric="Accuracy", 
#             trControl=trainControl(method="oob", number=25),
#             #default to be 500
#             ntree = 500,
#             #tuneLength = 10,
#             tuneGrid=rfGrid)
# print(rf2)
# 
# test_pred <- predict(rf2, test)
# confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh ))


### XGBoosting =====
# xgboostingGrid <- expand.grid(nrounds = 300,
#                               alpha = 10^seq(-3, 3, length = 20), 
#                               lambda = 10^seq(-3, 3, length = 20),
#                               eta = c(0.1, 0.2, 0.3))

xgboosting <- train(math_ach_lvl_blwhigh~., 
              data=training, 
              method="xgbLinear", 
              metric = "Sens",
              trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
              tuneLength = 5)
              #tuneGrid=xgboostingGrid,
              #verbose = TRUE)
print(xgboosting)

test_pred_xgboosting <- predict(xgboosting, test)
result_xgboosting <- confusionMatrix(table(test_pred_xgboosting , test$math_ach_lvl_blwhigh ))



# ### Random Forest by Randomization -----
# rfr <- train(math_ach_lvl_blwhigh~.,
#              data=training,
#              method="extraTrees",
#              metric="Accuracy",
#              #trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
#              #default to be 500
#              ntree = 500)
# print(rfr)
# 
# test_pred <- predict(rfr, test)
# confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh ))


### Nerual network -----
nnet <- train(math_ach_lvl_blwhigh~., 
                    data=training, 
                    method="nnet", 
                    metric = "Sens",
                    trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
                    tuneLength = 5)
print(nnet)

test_pred_nnet <- predict(nnet, test)
result_nnet <- confusionMatrix(table(test_pred_nnet , test$math_ach_lvl_blwhigh ))


### SVM -----
svmGrid <- expand.grid(C = c(0.1 , 0.3, 0.5, 0.7, 1))
svm <- train(math_ach_lvl_blwhigh~., 
              data=training, 
              method="svmLinear", 
              metric = "Sens",
              trControl=trainControl(method="cv", number=5,  classProbs=TRUE, summaryFunction = twoClassSummary),
              tuneGrid=svmGrid)
print(svm)

test_pred_svm <- predict(svm, test)
result_svm <- confusionMatrix(table(test_pred_svm , test$math_ach_lvl_blwhigh ))

# 
# # Variable importance
# # top 20 most important variables
# varImp(rf_2) #"scale = TRUE" is the default
# top20_plot <- plot(varImp(rf_2), top = 20)
# top20_plot
# 
# # get the names of the top 20 variables
# top20_varName <- as.character(top20_plot$panel.args[[1]]$y)
# 
# codebook %>% 
#   filter(variableName %in% top20_varName) %>% 
#   View()
# 
# # Create new RF models with the top variables



### Ridge -----
ridgeGrid <- expand.grid(alpha = 0, lambda = 10^seq(-3, 3, length = 100))
ridge <- train(math_ach_lvl_blwhigh~., 
               data=training, 
               method="glmnet", 
               metric = "Sens",
               trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
               tuneGrid=ridgeGrid)
print(ridge)

test_pred_ridge <- predict(ridge, test)
result_ridge <- confusionMatrix(table(test_pred_ridge , test$math_ach_lvl_blwhigh ))


### LASSO -----
lassoGrid <- expand.grid(alpha = 1, lambda = 10^seq(-3, 3, length = 100))
lasso <- train(math_ach_lvl_blwhigh~., 
             data=training, 
             method="glmnet", 
             metric = "Sens",
             trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
             tuneGrid=lassoGrid)
print(lasso)

test_pred_lasso <- predict(lasso, test)
result_lasso <- confusionMatrix(table(test_pred_lasso , test$math_ach_lvl_blwhigh ))

### Elastic net -----
#lassoGrid <- expand.grid(alpha = 1, lambda = 10^seq(-3, 3, length = 100))
elasticNet <- train(math_ach_lvl_blwhigh~., 
               data=training, 
               method="glmnet", 
               metric = "Sens",
               trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
               tuneLength = 10)
print(elasticNet)

test_pred_elasticNet <- predict(elasticNet, test)
result_elasticNet <- confusionMatrix(table(test_pred_elasticNet , test$math_ach_lvl_blwhigh ))

### compare ridge, lasso and elasticNet =====
models <- list(ridge = ridge, lasso = lasso, elasticNet = elasticNet)
resamples(models) %>% summary( metric = "Sens")

### compare all models? =====
#models <- list(ridge = ridge, lasso = lasso, elasticNet = elasticNet)
#resamples(models) %>% summary( metric = "Sens")


### Boosted Classification Trees ----- (YL: Trang: somehow this is taking forever to train, could you take a look?)
ada <- train(math_ach_lvl_blwhigh ~., 
                  data = training, 
                  method = "ada",
                  metric = "Sens",
                  trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
                  tuneLength = 5)


ada

# Apply the model to the test set
test_pred_ada <- predict(ada, test)
# See the accuracy of the model
result_ada <- confusionMatrix(table(test_pred_ada , test$math_ach_lvl_blwhigh))



### GLM ----- 
# use glm() no cross validation
glm = glm(math_ach_lvl_blwhigh ~., 
               data=training, 
               family=binomial)
test_pred_glm <- predict(glm, test, type="response")
test_pred_glm[test_pred_glm >= 0.5] = "not_low"
test_pred_glm[test_pred_glm < 0.5] = "low"

result_glm <- confusionMatrix(table(test_pred_glm , test$math_ach_lvl_blwhigh))


# use caret's glm with cross validation
glm_2 <- train(math_ach_lvl_blwhigh ~., 
             data = training, 
             method = "glm",
             family = binomial)
             #trControl = trainControl(method="cv", number=5))

glm_2

# Apply the model to the test set
test_pred_glm_2 <- predict(glm_2, test)
# See the accuracy of the model
result_glm_2 <- confusionMatrix(table(test_pred_glm_2 , test$math_ach_lvl_blwhigh))

##YL: glm and glm_2 are exactly the same, which is to be expected based on this post "https://www.kaggle.com/c/titanic/discussion/13582"



# use the only the traditional RHS variables
glm_3 <- train(math_ach_lvl_blwhigh ~ 	
                 itsex + bsdage + bsbg10a + bsbgher + bsdgedup + bsbgscm + bcbgdas + bcbg05a + bcbg05b, 
               data = training, 
               method = "glm",
               family = binomial)

glm_3

# Apply the model to the test set
test_pred_glm_3 <- predict(glm_3, test)
# See the accuracy of the model
result_glm_3 <- confusionMatrix(table(test_pred_glm_3 , test$math_ach_lvl_blwhigh))

##YL: glm and glm_2 are exactly the same, which is to be expected based on this post "https://www.kaggle.com/c/titanic/discussion/13582"



### Compare all models =====

allModels <- list(dtree, rf, rf_2, xgboosting, nnet, svm, ridge, lasso, elasticNet, 
                   #ada, 
                   glm, glm_2, glm_3)
allResults <- list(result_dtree, result_rf, result_rf_2, result_xgboosting, result_nnet, result_svm, result_ridge, result_lasso, result_elasticNet, 
                #result_ada, 
                result_glm, result_glm_2, result_glm_3)
allResults <- list(result_dtree, result_rf, result_rf_2)
FUN <- function(x){
  temp <- data.frame()
  return(x[["overall"]])
}

df <- data.frame()
for (i in 1:length(allResults)){
  temp <- allResults[[i]]$overall
  print(temp)
}
for (i in 1:length(allResults)){
  temp <- allResults[[i]]$byClass["Balanced Accuracy"]
  print(temp)
}



FUN(result_dtree)
FUN(allResults)
allResults[[1]]$overall
colnames(allResults[[1]]$table)

sapply(allResults, FUN = function(x){return(x$overall)})
sapply(allResults, "[[", "overall")
sapply(allResults,FUN)


### Variable Importance =====
sapply(allModels, varImp)
library(data.table)

for (i in 1:length(allModels)){
  varimp <- varImp(allModels[[i]])
  df <- as.data.frame(varimp$importance)
  df <- setDT(df, keep.rownames = TRUE)
  colnames(df) <- c("var", "imp")
}

varimp_result <- function(x){
  varimp <- varImp(x)
  df <- as.data.frame(varimp$importance)
  #df <- data.table::setDT(df, keep.rownames = TRUE)
  rname <- rownames(df)
  rownames(df) <- NULL
  df <- cbind(rname, df)
  #select only the first two columns (because the svm result has three columns somehow)
  if (ncol(df) > 2){
    df <- df[ , c(1,2)]
  }
  if (ncol(df) == 2){
    colnames(df) <- c("var", "imp")
  }
  df <- df %>% arrange(desc(imp))
  
  return(df)
}

varimp <- sapply(allModels, varimp_result)

# find top 20 vars
top_20 <- function(x){
  if (length(x$var) >= 20){
    top20 <- x$var[1:20]
  } else
    top20 <- x$var[1:length(x$var)]
}

top20_list <- sapply(varimp, top_20)

top20_list[[10]] <- NULL
top20_list[[11]] <- NULL
top20_list

setdiff(top20_list[[1]], top20_list[[2]])

#exist across all lists
Reduce(intersect, top20_list)
Reduce(union, top20_list)






varimp[[12]]$var[1:20]











varImp(rf_2) #"scale = TRUE" is the default
top20_plot <- plot(varImp(rf_2), top = 20)
top20_plot

# get the names of the top 20 variables
top20_varName <- as.character(top20_plot$panel.args[[1]]$y)

codebook %>%
  filter(variableName %in% top20_varName) %>%
  View()

# Create new RF models with the top variables





#
merge.light.edsurvey.data.frame(x, y)



### Unsupervised ML - Kmeans cluster -----
#reference: https://uc-r.github.io/kmeans_clustering
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

# need to convert all columns (including the factors) into numeric
low_performer <- T15_USA_df_clean[T15_USA_df_clean$math_ach_lvl_blwhigh == 1,]
low_performer_numeric <- low_performer %>% 
  mutate_all(as.numeric) %>%  #note that, by doing this, 2-level factors are turned into 1 & 2, instead of 0 & 1, may need to fix this later
  # de-select the label
  select(-math_ach_lvl_blwhigh)

# standardize these columns
process_configuration_2 <- preProcess(low_performer_numeric,
                                      method = c("center", "scale"))
low_performer_numeric_standardize <- predict(process_configuration_2, low_performer_numeric)
  
  

### try k = 2 as the start
k2 <- kmeans(low_performer_numeric, centers = 2, nstart = 25)
k2_standardize <- kmeans(low_performer_numeric_standardize, centers = 2, nstart = 25)

str(k2)

# visualize using fviz_cluster which performs PCA and plot the data points according to the first two principal components
fviz_cluster(k2, data = low_performer_numeric)
fviz_cluster(k2, data = low_performer_numeric_standardize)


# visualize using scatter plots to illustrate the clusters compared to the original variables
low_performer_numeric %>%
  as_tibble() %>%
  mutate(cluster = k2$cluster) %>%
  #just randomly pick the two continuous variabels to visualize
  #bsbgscm - Student Confident in Mathematics/SCL
  #bsbgher - Home Educational Resources/SCL
  ggplot(aes(bsbgscm, bsbgher, color = factor(cluster))) +
  geom_point()

### Determining Optimal number of Clusters

## Elbow method (there are two other methods mentioned in the reference)
set.seed(123)
# method = "wss" ---> compute total within-cluster sum of square
fviz_nbclust(low_performer_numeric, kmeans, method = "wss", k.max = 10) 


# Compute k-means clustering with k = 4
set.seed(123)
final <- kmeans(low_performer_numeric, 4, nstart = 25)
print(final)

#visualize
fviz_cluster(final, data = low_performer_numeric)

#summary - look at a few variables by cluster
low_performer_2 <- low_performer %>% #using the unscaled df here
  as_tibble() %>% 
  mutate(cluster = final$cluster) %>%
  group_by(cluster) %>% 
  mutate(n_eachCluster = n()) %>% 
  ungroup()

cluster_summary <- low_performer_2 %>% 
  group_by(cluster) %>% 
  summarise_all("mean") %>% 
  ungroup()





