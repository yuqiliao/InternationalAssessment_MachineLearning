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


### Download/Read in/Create a data frame for modeling -----

### Download data
#downloadTIMSS(year=2015, root = "/Users/Yuqi/Desktop/Files/AIR/Machine learning/Data")
T15_USA_G4 <- readTIMSS("G:/Conference/2019/Data/TIMSS/TIMSS2015", countries = c("usa"), gradeLvl = "4")
T15_USA_G8 <- readTIMSS("G:/Conference/2019/Data/TIMSS/TIMSS2015", countries = c("usa"), gradeLvl = "8")
#P15_USA <- readPIRLS("./Data/PIRLS/P16_and_PL16_Data", countries = c("usa"))

### Read in (1st time) to find out missing rate
# calculate missing rate and decide on which variables to use
# read in everything (in the student file and the teacher file) first
#student_teacher_var_G4 <- tolower(union(T15_USA_G4$fileFormat$variableName, T15_USA_G4$fileFormatTeacher$variableName))
student_school_var_G4 <- tolower(union(T15_USA_G4$fileFormat$variableName, T15_USA_G4$fileFormatSchool$variableName))
#student_teacher_var_G8 <- tolower(union(T15_USA_G8$fileFormat$variableName, T15_USA_G8$fileFormatTeacher$variableName))
student_school_var_G8 <- tolower(union(T15_USA_G8$fileFormat$variableName, T15_USA_G8$fileFormatSchool$variableName))
##YL: discuss with Trang: in the TIMSS paper, it merges G4 student and teacher data together (for Korea), which make the n of rows increase from 4334 to 4771 (because for some students there are more than one teachers). The paper then "This study kept the first observation of the duplicates, which resulted in the original number of 4,334 observations with a total of 586 variables." I think it is a bit problematic. Do you agree? In the same logic, I think it's best not to merge student with teacher variables. I then merge student with school variables, so the number of rows does not increase (becuase each student only has one school principle)

T15_USA_G4_df <- getData(data = T15_USA_G4, varnames = student_school_var_G4,
                      omittedLevels = FALSE, addAttributes = TRUE)
T15_USA_G8_df <- getData(data = T15_USA_G8, varnames = student_school_var_G8,
                         omittedLevels = FALSE, addAttributes = TRUE)

# Use G8 data as a start
T15_USA_df <- T15_USA_G8_df

# calculate missing rate
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
summary(missing_rate_new) ##YL: confirm with Trang #though they seem to be the same, i think my code is correct, it should be just that there the omittedlevels do not exist in the data frame, it may exist elsewhere in the teacher data or in the assessment data? 

##YL: also I think Trang's code below is not correct?
missing_rate <- sapply(colnames(P15_USA_df),
                       function(i) {
                         mean(missing_rule(P15_USA_df[,i],om))
                       })
missing_rate <- sapply(colnames(P15_USA_df),
                       function(i) {
                         mean(missing_rule(P15_USA_df[,i],om))
                       })


# keep variabels that has missing rate < 0.1
missing_rate_less0.1 <- missing_rate[missing_rate < 0.1]
names(missing_rate_less0.1)

# exclude variables that are weights, ids (except for itsex), pvs, (and vars taht are derived from questionnaire items), as were done in the TIMSS paper
# stuVarCols <- c(9, 51:115) #9 is itsex, accordingly "asbg01" (boy/girl) is excluded
# tchVarCols <- c(290:416) 
# stu_tch_VarCols <- c(stuVarCols, tchVarCols)
# # variable names for Y (changed from using 5 plausible proficiency levels to using 5 reading PVs)
# #read_ach_lvl <- c("asribm01", "asribm02", "asribm03", "asribm04", "asribm05")
# read_pv <- "rrea"
# # try with weight
# techer_weight <- 577
# student_weight <- 576
# stu_tch_VarCols_plusWeight <-  c(stuVarCols, tchVarCols, techer_weight)


includeVar <- grep("^jk|wgt|^id|^ita|^bsdg|^bcdg",names(missing_rate_less0.1), value = TRUE, invert = TRUE) ##for a series of derived variables that have both scale and index versions, "^bsdg|^bcdg" is to drop the index version (cuz otherwise they are highly colinear)
#YL: for G8, a lot of background quesitons have both sci and math versions, should I be concerned that they are highly correlated with each other?

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



# first get the average of bsmmat01-05, and create a new variable to document the proficiency level of that average
al <- as.numeric(getAttributes(T15_USA_df,'achievementLevels'))
T15_USA_df_stu_sch_clean <- T15_USA_df_stu_sch %>%
  mutate(bsmmat_mean = (bsmmat01 + bsmmat02 + bsmmat03 + bsmmat04 + bsmmat05)/5,
         math_ach_lvl_blwhigh = ifelse(bsmmat_mean < al[2], 1, 0),
         math_ach_lvl_blwhigh = as.factor(math_ach_lvl_blwhigh)) %>% 
  select(-c(bsmmat01, bsmmat02, bsmmat03, bsmmat04, bsmmat05, bsmmat_mean))

#check the balance of Y ##YL: I expect it to not be balanced, because we are "predicting" the low-performing students
table(T15_USA_df_stu_sch_clean$math_ach_lvl_blwhigh)



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
colMeans(is.na(T15_USA_df_clean)) %>% sum()



### Model building - preProcess -----

# # check the missing rate for each variable, just in case
# colMeans(is.na(P15_USA_df_stu_tch_weight_clean)) %>% sum()

# Pre-process (e.g. rescale and center) Note that here all predictors should actullay be in likert scale, though some are in 2 levels, and some oar 3, or 4 levels. I think it's best to pre-process them anyway
#yl : not working for now

# identify and remove columns with near zero variance (otherwise the models later won't run on this constant terms)
nzv <- nearZeroVar(T15_USA_df_clean, freqCut = 95/5)
T15_USA_df_clean <- T15_USA_df_clean[,-c(nzv)]

process_configuration <- preProcess(T15_USA_df_clean,
                                    method = c("center", "scale"))
processed <- predict(process_configuration, T15_USA_df_clean)





### Model building - data split -----
set.seed(123)
trainingIndex <- createDataPartition(processed$math_ach_lvl_blwhigh, p = 0.8, list = FALSE)
training <- processed[trainingIndex, ]
test <- processed[-trainingIndex, ]


### Decision Tree (CART) -----
dtree_fit <- train(math_ach_lvl_blwhigh ~., 
                   data = training, 
                   method = "rpart")
                   #trControl=trainControl(method="oob", number=25),
                   #tuneLength = 10, 
                   #parms=list(split='information')

dtree_fit

# Apply the model to the test set
test_pred <- predict(dtree_fit, test)
# See the accuracy of the model
confusionMatrix(table(test_pred , test$math_ach_lvl_blwhigh ))

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





