## Machine learning with ILSA
## Yuqi Liao & Trang Nguyen
## 08/29/2018

### Ideas dump -----
# we could try to use the US restrict-use files to include more US variables!
# focus on the US dataset for now, but could extend the scope to analyze datasets for all countries and do a comparison

### Tentative research quesitons
# RQ1: Could machine learning algorithms be better than logistic regressions in predicting student outcome?
# RQ2: What variables could best predict student outcome (achieving the “high” reading proficiency level, or NOT achieving the proficiency level)?
# RQ3: Among students with low proficiency levels, what are the characteristics?

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

### Step 1. Read in and prepare data ==============
### 1a. Download/Read in/Create a data frame for modeling -----

### Download data
#downloadTIMSS(year=2015, root = "/Users/Yuqi/Desktop/Files/AIR/Machine learning/Data")
T15_USA_G4 <- readTIMSS("G:/Conference/2019/Data/TIMSS/TIMSS2015", countries = c("usa"), gradeLvl = "4")
T15_USA_G8 <- readTIMSS("G:/Conference/2019/Data/TIMSS/TIMSS2015", countries = c("usa"), gradeLvl = "8")

### Read in (1st time) to find out missing rate
# calculate missing rate and decide on which variables to use
# read in everything (in the student file and the teacher file) first
student_school_var_G4 <- tolower(union(T15_USA_G4$fileFormat$variableName, T15_USA_G4$fileFormatSchool$variableName))
student_school_var_G8 <- tolower(union(T15_USA_G8$fileFormat$variableName, T15_USA_G8$fileFormatSchool$variableName))

T15_USA_G4_df <- getData(data = T15_USA_G4, varnames = student_school_var_G4,
                      omittedLevels = FALSE, addAttributes = TRUE)
T15_USA_G8_df <- getData(data = T15_USA_G8, varnames = student_school_var_G8,
                         omittedLevels = FALSE, addAttributes = TRUE)

# Use G8 data as a start
T15_USA_df <- T15_USA_G8_df

# 1b. calculate missing rate
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
missing_rate_less0.1 <- missing_rate[missing_rate < 0.1]
names(missing_rate_less0.1)


# 1c. Remove some irrelevant variables (i.e. weight)
includeVar <- grep("^jk|wgt|^id|^ita|^bsdg|^bcdg",names(missing_rate_less0.1), value = TRUE, invert = TRUE)
# Remove plausible values from includeVar
pvvars <- getAttributes(T15_USA_df,'pvvars')
pvvars <- unlist(lapply(pvvars, function(p) p$varnames))
includeVar <- includeVar[!includeVar %in% pvvars]

# check codebook of included variables
codebook <- showCodebook(T15_USA_df)
codebook <- codebook[tolower(codebook$variableName) %in% includeVar,]

# after looking at the codebook, exclude some variables
excludeVar <- c("ilreliab","version","bsbg01", "itlang", "bsdmlowp", "bsdslowp") #exclude "itlang" cuz the US only has one level
includeVar <- includeVar[!includeVar %in% excludeVar]

### Read in (2nd time) to have all right-hand-site variables needed for the model=====
# Reformat outcome variable # define variable names for Y
math_ach_lvl <- c("mmat")

# get the df using listwise deletion of the omitted levels
T15_USA_df_stu_sch <- getData(data = T15_USA_df, varnames = c(includeVar, math_ach_lvl),
                              omittedLevels = TRUE, addAttributes = TRUE)

# Step 2. Feature engineering
# 2a. Correlational matrix
# Goal: Idenitfy pairs of highly correlated variables


# Step 3. Prepare data for model fitting
# 3a. Create dependent variable
# first get the average of bsmmat01-05, and create a new variable to document the proficiency level of that average
al <- as.numeric(getAttributes(T15_USA_df,'achievementLevels'))
T15_USA_df_stu_sch_clean <- T15_USA_df_stu_sch %>%
  mutate(bsmmat_mean = (bsmmat01 + bsmmat02 + bsmmat03 + bsmmat04 + bsmmat05)/5,
         math_ach_lvl_blwhigh = ifelse(bsmmat_mean < al[2], 1, 0),
         math_ach_lvl_blwhigh = as.factor(math_ach_lvl_blwhigh)) %>% 
  select(-c(bsmmat01, bsmmat02, bsmmat03, bsmmat04, bsmmat05, bsmmat_mean))

#check the balance of Y ##YL: I expect it to not be balanced, because we are "predicting" the low-performing students
table(T15_USA_df_stu_sch_clean$math_ach_lvl_blwhigh)


# 3b. Scaling independent variables
# identify vars that has more than 2 levels (to turn as numeric later)
level_morethan2 <- sapply(T15_USA_df_stu_sch_clean, nlevels) > 2
# note that later i'll have to go through all the variable levels to make sure it makes sense to convert vars that has more than 2 levels into numeric (e.g. in TIMSS G8, there are two vars c("bsbg09a", "bsbg09b") that asks students the origin of their parents, to which one of the level is "i don't know", so for those vars, it is better to have them stay as factors.
level_morethan2["bsbg09a"] <- FALSE
level_morethan2["bsbg09b"] <- FALSE


T15_USA_df_clean <- T15_USA_df_stu_sch_clean[, level_morethan2] %>%
  # transform some variables into numeric
  mutate_all(as.numeric) %>%
  bind_cols(T15_USA_df_stu_sch_clean[, !level_morethan2])


# check the missing rate for each variable, just in case
colMeans(missing_rule(T15_USA_df_clean,om)) %>% sum()



# 3d. Prepocess data
# identify and remove columns with near zero variance (otherwise the models later won't run on this constant terms)
nzv <- nearZeroVar(T15_USA_df_clean, freqCut = 95/5)
T15_USA_df_clean <- T15_USA_df_clean[,-c(nzv)]

process_configuration <- preProcess(T15_USA_df_clean,
                                    method = c("center", "scale"))
processed <- predict(process_configuration, T15_USA_df_clean)



### Step 4. Model building ===========
# 4a. data split -----
set.seed(123)
trainingIndex <- createDataPartition(processed$math_ach_lvl_blwhigh, p = 0.8, list = FALSE)
training <- processed[trainingIndex, ]
test <- processed[-trainingIndex, ]


### Decision Tree (CART) -----
dtree_fit <- train(math_ach_lvl_blwhigh ~., 
                   data = training, 
                   method = "rpart",
                   trControl=trainControl(method="cv", number=10),
                   tuneLength = 10)

dtree_fit

# Apply the model to the test set
test_pred <- predict(dtree_fit, test)
# See the accuracy of the model
confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh ))


### Random Forest -----
# try using tuneLength #mtry = 83 is used in the end
mtry <- sqrt(ncol(training) - 1) 
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

test_pred <- predict(rf, test)
confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh ))

# try using tuneGrid
mtry <- sqrt(ncol(training) - 1)
rfGrid <- expand.grid(mtry = c(5,  13 , 18, 25, 30))

rf2 <- train(math_ach_lvl_blwhigh~., 
            data=training, 
            method="rf", 
            metric="Accuracy", 
            trControl=trainControl(method="oob", number=25),
            #default to be 500
            ntree = 500,
            #tuneLength = 10,
            tuneGrid=rfGrid)
print(rf2)

test_pred <- predict(rf2, test)
confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh ))


### XGBoosting


### Random Forest by Randomization -----
# rf3 <- train(math_ach_lvl_blwhigh~., 
#              data=training, 
#              method="extraTrees", 
#              metric="Accuracy", 
#              trControl=trainControl(method="cv", number=25),
#              #default to be 500
#              ntree = 500,
#              tuneLength = 3)
# print(rf3)
# 
# test_pred <- predict(rf3, test)
# confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh ))
# 






# Variable importance
# top 20 most important variables
varImp(rf) #"scale = TRUE" is the default
top20_plot <- plot(varImp(rf), top = 20)
top20_plot

# get the names of the top 20 variables
top20_varName <- as.character(top20_plot$panel.args[[1]]$y)

codebook %>% 
  filter(variableName %in% top20_varName) %>% 
  View()

# Create new RF models with the top variables
data_top20 <- T15_USA_df_clean %>%
  select(math_ach_lvl_blwhigh,
         top20_varName)
set.seed(123)
testIndex <- createDataPartition(data_top20$math_ach_lvl_blwhigh, p = 0.8, list = FALSE)
training <- data_top20[trainingIndex, ]
test <- data_top20[-trainingIndex, ]

# Create model
mtry <- sqrt(ncol(data_top10) - 1)
#mtry <- 16 #could do fine tuning to find the optimal mtry value
rf_top20 <- train(math_ach_lvl_blwhigh~., 
            data=training, 
            method="rf", 
            metric="Accuracy", 
            #tuneGrid=expand.grid(.mtry=mtry), 
            trControl=trainControl(method="oob", number=25),
            #default to be 500
            ntree = 500,
            tuneLength = 5)
print(rf_top20)

# Apply the model to the test set
test_pred <- predict(rf_top20, test)
# Use confusionMatrix
confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh ))


# ### Random Forest - with weights 
# mtry <- sqrt(ncol(training) - 1) 
# rf_weight <- train(read_ach_lvl_atabv4~., 
#             data=training_weight, 
#             method="rf", 
#             metric="Accuracy", 
#             tuneGrid=expand.grid(.mtry=mtry), 
#             trControl=trainControl(method="oob", number=25),
#             #default to be 500
#             ntree = 500)
# print(rf_weight)
# 
# 
# # Apply the model to the test set
# test_pred_weight <- predict(rf_weight, test_weight)
# # Use confusionMatrix
# confusionMatrix(table(test_pred_weight , test_weight$read_ach_lvl_atabv4 ))
# 
# 
# # Variable importance
# # top 20 most important variables
# varImp(rf_weight) #"scale = TRUE" is the default
# top20_plot_weight <- plot(varImp(rf_weight), top = 20)
# # get the names of the top 20 variables
# top20_varName_weight <- as.character(top20_plot_weight$panel.args[[1]]$y)
# 
# 
# #find intersection
# intersect(top20_varName, top20_varName_weight)
# setdiff(top20_varName, top20_varName_weight)


### LASSO -----
lasso <- train(math_ach_lvl_blwhigh ~., 
             data = training, 
             method = "glmnet",
             alpha = 0.5,
             lambda = 1,
             family = "binomial")
             #trControl=trainControl(method="cv", number=10))
print(lasso)

test_pred <- predict(lasso, test)
confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh ))



### Boosted Classification Trees ----- (YL: Trang: somehow this is taking forever to train, could you take a look?)
ada <- train(math_ach_lvl_blwhigh ~., 
                  data = training, 
                  method = "ada",
                  iter = 500,
                  nu = 0.05,
                  maxdepth = 50
                  )
#trControl=trainControl(method="oob", number=25),
#tuneLength = 10, 
#parms=list(split='information')

ada

# Apply the model to the test set
test_pred <- predict(ada, test)
# See the accuracy of the model
confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh))



### Nerual network -----

### GBM (Gradient Boosted Machines) /XGBoost -----

### GLM ----- ##YL: ASK Trang, it's not working ;/
glm <- train(math_ach_lvl_blwhigh ~., 
             data = training, 
             method = "glmnet",
             alpha = 1,
             lambda = 1)
#trControl=trainControl(method="oob", number=25),
#tuneLength = 10, 
#parms=list(split='information')

glm

# Apply the model to the test set
test_pred <- predict(glm, test)
# See the accuracy of the model
confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh))


glm2 <- glm(math_ach_lvl_blwhigh ~., 
            data = training,
            family = binomial())

summary(glm2)
test_pred <- predict(glm2, test, type="response")
test_pred[test_pred >= 0.5] = 1
test_pred[test_pred < 0.5] = 0

confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh))



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





