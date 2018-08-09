## Machine learning with ILSA
## Yuqi Liao & Trang Nguyen
## 07/21/2018

### Ideas dump -----
# we could try to use the US restrict-use files to include more US variables!
# focus on the US dataset for now, but could extend the scope to analyze datasets for all countries and do a comparison

### Tentative research quesitons
# RQ1: Could machine learning algorithms be better than logistic regressions in predicting student outcome?
# RQ2: What variables could best predict student outcome (achieving the “high” reading proficiency level, or NOT achieving the proficiency level)?
# RQ3: Among students with low proficiency levels, what are the characteristics?

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


### Download/Read in/Create a data frame for modeling -----

### Download data
#downloadTIMSS(year=2015, root = "/Users/Yuqi/Desktop/Files/AIR/Machine learning/Data")
#T15_USA <- readTIMSS("G:/Conference/2019/Data/TIMSS/TIMSS2015", countries = c("usa"), gradeLvl = "4")
P15_USA <- readPIRLS("./Data/PIRLS/P16_and_PL16_Data", countries = c("usa"))

### Read in (1st time) to find out missing rate
# calculate missing rate and decide on which variables to use
# read in everything (in the student file and the teacher file) first
student_teacher_var <- tolower(union(P15_USA$fileFormat$variableName, P15_USA$fileFormatTeacher$variableName))
P15_USA_df <- getData(data = P15_USA, varnames = student_teacher_var,
                      omittedLevels = FALSE, addAttributes = TRUE)

# calculate missing rate
missing_rate <- colMeans(is.na(P15_USA_df)) 

# keep variabels that has missing rate < 0.1
missing_rate_less0.1 <- missing_rate[missing_rate < 0.1]
names(missing_rate_less0.1)

# exclude variables that are weights, ids (except for itsex), pvs, (and vars taht are derived from questionnaire items), as were done in the TIMSS paper
stuVarCols <- c(9, 51:115) #9 is itsex, accordingly "asbg01" (boy/girl) is excluded
tchVarCols <- c(290:416) 
stu_tch_VarCols <- c(stuVarCols, tchVarCols)
# variable names for Y (changed from using 5 plausible proficiency levels to using 5 reading PVs)
#read_ach_lvl <- c("asribm01", "asribm02", "asribm03", "asribm04", "asribm05")
read_pv <- "rrea"

# # try with weight
# techer_weight <- 577
# student_weight <- 576
# stu_tch_VarCols_plusWeight <-  c(stuVarCols, tchVarCols, techer_weight)
  
missing_rate_less0.1_stu_tch <- names(missing_rate_less0.1[stu_tch_VarCols])
#missing_rate_less0.1_stu_tch_weight <- names(missing_rate_less0.1[stu_tch_VarCols_plusWeight])


### Read in (2nd time) to have all right-hand-site variables needed for the model
# get the df using listwise deletion of the omitted levels
P15_USA_df_stu_tch <- getData(data = P15_USA, varnames = c(missing_rate_less0.1_stu_tch, read_pv),
                                          omittedLevels = TRUE, addAttributes = TRUE)
# P15_USA_df_stu_tch_weight <- getData(data = P15_USA, varnames = c(missing_rate_less0.1_stu_tch_weight, read_ach_lvl),
#                               omittedLevels = TRUE, addAttributes = TRUE)

### Define Y
# # Y would be the majority vote of asribm01-05
# P15_USA_df_stu_tch <- P15_USA_df_stu_tch %>%
#   mutate(asribm01 = as.numeric(asribm01),
#          asribm02 = as.numeric(asribm02),
#          asribm03 = as.numeric(asribm03),
#          asribm04 = as.numeric(asribm04),
#          asribm05 = as.numeric(asribm05))

# first get the average of asrrea01-05, and create a new variable to document the proficiency level of that average
al <- as.numeric(getAttributes(P15_USA,'achievementLevels'))
P15_USA_df_stu_tch <- P15_USA_df_stu_tch %>%
  mutate(asrrea_mean = (asrrea01 + asrrea02 + asrrea03 + asrrea04 + asrrea05)/5,
         #read_ach_lvl_atabv4 = ifelse(asrrea_mean >= al[3], 1, 0),
         #read_ach_lvl_atabv4 = as.factor(read_ach_lvl_atabv4),
         read_ach_lvl_blwhigh = ifelse(asrrea_mean < al[3], 1, 0),
         read_ach_lvl_blwhigh = as.factor(read_ach_lvl_blwhigh)) %>% 
  select(-c(asrrea01, asrrea02, asrrea03, asrrea04, asrrea05, asrrea_mean))
summary(P15_USA_df_stu_tch_1$read_ach_lvl_blwhigh)



# P15_USA_df_stu_tch_clean <- P15_USA_df_stu_tch %>% 
#   # create new variable math_ach_lvl which is the mode of all asmibm01-05
#   rowwise() %>% 
#   summarize(read_ach_lvl = round((asribm01 + asribm02 + asribm03 + asribm04 +asribm05)/5, digits = 0)) %>% 
#   ungroup() %>% 
#   bind_cols(P15_USA_df_stu_tch) %>% 
#   # create dummy version of math_ach_lvl
#   mutate(read_ach_lvl_atabv4 = ifelse(read_ach_lvl %in% c(4,5), 1, 0),
#          read_ach_lvl_atabv4 = as.factor(read_ach_lvl_atabv4)) %>% 
#   select(-c(asribm01, asribm02, asribm03, asribm04, asribm05, read_ach_lvl))

# ### try with weight
# P15_USA_df_stu_tch_weight <- P15_USA_df_stu_tch_weight %>% 
#   mutate(asribm01 = as.numeric(asribm01),
#          asribm02 = as.numeric(asribm02),
#          asribm03 = as.numeric(asribm03),
#          asribm04 = as.numeric(asribm04),
#          asribm05 = as.numeric(asribm05))
# 
# P15_USA_df_stu_tch_weight_clean <- P15_USA_df_stu_tch_weight %>% 
#   # create new variable math_ach_lvl which is the mode of all asmibm01-05
#   rowwise() %>% 
#   summarize(read_ach_lvl = round((asribm01 + asribm02 + asribm03 + asribm04 +asribm05)/5, digits = 0)) %>% 
#   ungroup() %>% 
#   bind_cols(P15_USA_df_stu_tch_weight) %>% 
#   # create dummy version of math_ach_lvl
#   mutate(read_ach_lvl_atabv4 = ifelse(read_ach_lvl %in% c(4,5), 1, 0),
#          read_ach_lvl_atabv4 = as.factor(read_ach_lvl_atabv4)) %>% 
#   select(-c(asribm01, asribm02, asribm03, asribm04, asribm05, read_ach_lvl))
# 
# P15_USA_df_stu_tch_weight_clean <- P15_USA_df_stu_tch_weight_clean %>% 
#   select(-(contains("jk.tchwgt")))



# identify vars that has more than 2 levels (to turn as numeric later)
level_morethan2 <- sapply(P15_USA_df_stu_tch, nlevels) > 2
# note that later i'll have to go through all the variable levels to make sure it makes sense to convert vars that has more than 2 levels into numeric (e.g. in TIMSS G4, there are two vars c("asbg06a", "asbg06b") that asks students the origin of their parents, to which one of the level is "i don't know", so for those vars, it is better to have them stay as factors.

P15_USA_df_clean <- P15_USA_df_stu_tch[, level_morethan2] %>%
  # transform some variables into numeric
  mutate_all(as.numeric) %>%
  bind_cols(P15_USA_df_stu_tch[, !level_morethan2])


# P15_USA_df_stu_tch_weight_clean_numeric <- P15_USA_df_stu_tch_weight_clean %>% 
#   mutate_at(.funs = funs(as.numeric), .vars = colnames(P15_USA_df_stu_tch_weight_clean)[!colnames(P15_USA_df_stu_tch_weight_clean) %in% c("read_ach_lvl_atabv4")])



### Model building - preProcess -----

# # check the missing rate for each variable, just in case
# colMeans(is.na(P15_USA_df_stu_tch_weight_clean)) %>% sum()

# Pre-process (e.g. rescale and center) Note that here all predictors should actullay be in likert scale, though some are in 2 levels, and some oar 3, or 4 levels. I think it's best to pre-process them anyway
#yl : not working for now

# identify and remove columns with near zero variance (otherwise the models later won't run on this constant terms)
nzv <- nearZeroVar(P15_USA_df_clean, freqCut = 95/5)
P15_USA_df_clean <- P15_USA_df_clean[,-c(nzv)]

process_configuration <- preProcess(P15_USA_df_clean,
                                    method = c("center", "scale"))
processed <- predict(process_configuration, P15_USA_df_clean)


# # try with weight
# process_configuration <- preProcess(P15_USA_df_stu_tch_weight_clean_numeric,
#                                     method = c("nzv","center", "scale"))
# processed_wight <- predict(process_configuration, P15_USA_df_stu_tch_weight_clean_numeric)



### Model building - data split -----
set.seed(123)
trainingIndex <- createDataPartition(processed$read_ach_lvl_blwhigh, p = 0.8, list = FALSE)
training <- processed[trainingIndex, ]
test <- processed[-trainingIndex, ]

# # try with weights
# training_weight <- P15_USA_df_stu_tch_weight_clean_numeric[trainingIndex, ]
# test_weight <- P15_USA_df_stu_tch_weight_clean_numeric[-trainingIndex, ]



### Decision Tree (CART) -----
dtree_fit <- train(read_ach_lvl_blwhigh ~., 
                   data = training, 
                   method = "rpart")
                   #trControl=trainControl(method="oob", number=25),
                   #tuneLength = 10, 
                   #parms=list(split='information')

dtree_fit

# Apply the model to the test set
test_pred <- predict(dtree_fit, test)
# See the accuracy of the model
confusionMatrix(table(test_pred , test$read_ach_lvl_blwhigh ))

# ### Decision Tree (CART) - with weight
# dtree_fit_weight <- train(read_ach_lvl_atabv4 ~., 
#                    data = training_weight, 
#                    method = "rpart",
#                    weights = tchwgt)
# #trControl=trainControl(method="oob", number=25),
# #tuneLength = 10, 
# #parms=list(split='information')
# 
# dtree_fit_weight
# 
# # Apply the model to the test set
# test_pred <- predict(dtree_fit_weight, test_weight)
# # See the accuracy of the model
# confusionMatrix(table(test_pred , test_weight$read_ach_lvl_atabv4 ))




### Random Forest -----
mtry <- sqrt(ncol(training) - 1) 
rf <- train(read_ach_lvl_blwhigh~., 
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
# Use confusionMatrix
confusionMatrix(table(test_pred , test$read_ach_lvl_blwhigh ))


# Variable importance
# top 20 most important variables
varImp(rf) #"scale = TRUE" is the default
top20_plot <- plot(varImp(rf), top = 20)
# get the names of the top 20 variables
top20_varName <- as.character(top20_plot$panel.args[[1]]$y)


# # Create new RF models with the top variables
# data_top20 <- T15_USA_df_stu_clean %>% 
#   select(CHRONIC_ABSENTEE,
#          top20_varName)
# set.seed(123)
# testIndex <- createDataPartition(data_top20$CHRONIC_ABSENTEE, p = 0.8, list = FALSE)
# training <- data_top20[-testIndex, ]
# test <- data_top20[testIndex, ]
# 
# # Create model
# mtry <- sqrt(ncol(data_top10) - 1) 
# #mtry <- 16 #could do fine tuning to find the optimal mtry value
# rf_CHRONIC_ABSENTEE_top10 <- train(CHRONIC_ABSENTEE~., 
#                                    data=training, 
#                                    method="rf", 
#                                    metric="Accuracy", 
#                                    tuneGrid=expand.grid(.mtry=mtry), 
#                                    trControl=trainControl(method="oob", number=25),
#                                    #default to be 500
#                                    ntree = 1000)
# print(rf_CHRONIC_ABSENTEE_top10)
# 
# # Apply the model to the test set
# test_pred <- predict(rf_CHRONIC_ABSENTEE_top10, test)
# # Use confusionMatrix
# confusion_matrix_top10 <- confusionMatrix(table(test_pred , test$CHRONIC_ABSENTEE ))


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




### AdaBoost ----- (YL: Trang: somehow this is taking forever to train, could you take a look?)
adaboost <- train(read_ach_lvl_blwhigh ~., 
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

### GBM (Gradient Boosted Machines) /XGBoost -----

### GLM -----

### Unsupervised ML - Kmeans cluster -----
#reference: https://uc-r.github.io/kmeans_clustering
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

# need to convert all columns (including the factors) into numeric
low_performer <- P15_USA_df_clean[P15_USA_df_clean$read_ach_lvl_blwhigh == 1,]
low_performer_numeric <- low_performer %>% 
  mutate_all(as.numeric) %>%  #note that, by doing this, 2-level factors are turned into 1 & 2, instead of 0 & 1, may need to fix this later
  # de-select the label
  select(-read_ach_lvl_blwhigh)

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
  #atbg01 - GEN\YEARS BEEN TEACHING
  #atbr07 - READ\TIME SPENT READING INSTR
  ggplot(aes(atbr07, atbg01, color = factor(cluster))) +
  geom_point()

### Determining Optimal number of Clusters

## Elbow method (there are two other methods mentioned in the reference)
set.seed(123)
# method = "wss" ---> compute total within-cluster sum of square
fviz_nbclust(low_performer_numeric, kmeans, method = "wss") 


# Compute k-means clustering with k = 6
set.seed(123)
final <- kmeans(low_performer_numeric, 6, nstart = 25)
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





