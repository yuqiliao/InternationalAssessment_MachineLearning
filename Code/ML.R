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
#load("G:/Conference/2019/Git/InternationalAssessment_MachineLearning/all_objects.RData")

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
library(haven)
library(onehot)
library(data.table)


### Step 1. Read in and prepare data ==============
### 1a. Download/Read in/Create a data frame for modeling

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
  #Home Educational Resources/SCL: bsbg04, bsdg06s, bsdgedup
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

includeVar <- grep("bsbg04|bsdg06s|bsdgedup|^bsbg15|^bsbg16|^bsbm17|^bsbm18|^bsbm19|^bsbm20|^bsbs21|^bsbs22|^bsbs23|^bsbs24|^bsbb22|^bsbb23|^bsbb24|^bsbe26|^bsbe27|^bsbe28|^bsbc30|^bsbc31|^bsbc32|^bsbp34|^bsbp35|^bsbp36|^bcbg13|^bcbg14|^bcbg15", names(T15_USA_df), value = TRUE, invert = TRUE)


# Remove weights, index version of the derived variables, and cognitive items
includeVar <- grep("^jk|wgt|^id|^ita|^bsdg|^bcdg|^m0|^s0|^si|^sr|^mi|^mr",includeVar, value = TRUE, invert = TRUE) #1098 vars excluded

#add back student id for merging with US PUF file later
#add back bcdg03 and bcdg07hy because they were removed by the regular expression above but they do not seem to be part of the root variables for scale variables
includeVar <- c(includeVar, "bcdg03", "bcdg07hy", "idstud") 

# Remove plausible values from includeVar
pvvars <- getAttributes(T15_USA_df,'pvvars')
pvvars <- unlist(lapply(pvvars, function(p) p$varnames))
includeVar <- includeVar[!includeVar %in% pvvars] #90 vars excluded

# check codebook of included variables
codebook <- showCodebook(T15_USA_df)
codebook[tolower(codebook$variableName) %in% includeVar,] %>% View()


# after looking at the codebook, exclude some more variables
# exclude "itlang" cuz the US only has one level, "bsbg01" cuz "itsex" should cover it
# delete (bsbg07a, bsbg07b, bsbg08) based on the observation of the US PUF file later
excludeVar <- c("ilreliab","version","bsbg01", "itlang", "bsdmlowp", "bsdslowp", "bsbg07a", "bsbg07b", "bsbg08") 
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



### Read in (2nd time) to have all right-hand-site variables needed for the model
# Reformat outcome variable # define variable names for Y
math_ach_lvl <- c("mmat")

# get the df using listwise deletion of the omitted levels
T15_USA_df_stu_sch <- getData(data = T15_USA_df, varnames = c(includeVar, math_ach_lvl),
                              omittedLevels = TRUE, addAttributes = TRUE)



### Read in the US PUF for G8

US_PUF_stu_sch <- read_sav(file = "G:/Conference/2019/Git/InternationalAssessment_MachineLearning/Data/TIMSS 2015 G8 Data - PUF/TIMSS_2015_G8_PUBLIC/T15_G8_Stu_Sch_USPUF.SAV") #by default, user-defined missings (e.g. 8, 9, 999, etc.) will be converted to NA

names(US_PUF_stu_sch) <- tolower(names(US_PUF_stu_sch))

#get only the US national variables plus 'idstud'
to_keep <- grep("idstud|^bsx|^bsn|msrace2|^bcx|^bcn|pctfrpl|pubpriv",names(US_PUF_stu_sch), value = TRUE, invert = FALSE)
#get rid of 'bsxg03bt' (because it is an open-ended string variables)
to_keep <- to_keep[to_keep != 'bsxg03bt']

US_PUF_vars <- US_PUF_stu_sch %>% select(to_keep)

#check if there are any variables that has "not administered" levels (then delete them)
sapply(US_PUF_vars, function(x){summary(as.factor(x))})

# there are five US version of the variables "bsng07a", "bsng07b", "bsng08", "bcng07ba", "bcng07bb". decided to get rid of ("bcng07ba", "bcng07bb") because the international version makes more sense. later, i should drop the international version of the "bsng07a", "bsng07b", "bsng08", (bsbg07a, bsbg07b, bsbg08, respectively) because the US version provides more granular information
# get rid of ("bcng07ba", "bcng07bb")
to_keep <- to_keep[!(to_keep %in% c("bcng07ba", "bcng07bb"))]

#get rid of the variable that has too many missing values
missing_rate_uspuf <- colMeans(is.na(US_PUF_vars)) #since  `read_sav` already turned user-defined missings into NA, so this should be safe

missing_rate_uspuf_less0.15 <- missing_rate_uspuf[missing_rate_uspuf < 0.15]
names(missing_rate_uspuf_less0.15)

to_keep_2 <- to_keep[to_keep %in% names(missing_rate_uspuf_less0.15)]

US_PUF_vars <- US_PUF_stu_sch %>% select(to_keep_2)

# change variable type before merge
temp <- US_PUF_vars %>% 
  select(-idstud) %>% 
  haven::as_factor() %>% 
  #drop non-used levels, namely "Not administered", "Omitted or invalid" 
  droplevels()


US_PUF_vars$idstud <- as.integer(US_PUF_vars$idstud)

US_PUF_vars <- US_PUF_vars %>% 
  select(idstud) %>% 
  bind_cols(temp)

# merge US_PUF_vars with T15_USA_df_stu_sch

T15_USA_df_stu_sch_merged <- T15_USA_df_stu_sch %>% 
  left_join(US_PUF_vars, by = "idstud") %>% 
  #get rid of NA rows
  na.omit() %>% 
  select(-idstud)






# Step 2. Prepare data for model fitting ======
# 2a. Create dependent variable
# first get the average of bsmmat01-05, and create a new variable to document the proficiency level of that average
al <- as.numeric(getAttributes(T15_USA_df,'achievementLevels'))
T15_USA_df_stu_sch_clean <- T15_USA_df_stu_sch_merged %>%
  #create first depedent variable - low math score
  mutate(bsmmat_mean = (bsmmat01 + bsmmat02 + bsmmat03 + bsmmat04 + bsmmat05)/5,
         math_ach_lvl_blwhigh = ifelse(bsmmat_mean < al[2], "low", "not_low"),
         math_ach_lvl_blwhigh = factor(math_ach_lvl_blwhigh, levels = c("not_low", "low"))) %>% 
  #create second dependent variable - ever absent last month
  mutate(absent_ever = ifelse(bsxg13b %in% c("None"), "No", "Yes"),
         absent_ever = factor(absent_ever, levels = c("No", "Yes")),
         # absent more than 2 days last month (http://education.ohio.gov/Topics/Chronic-Absenteeism)
         absent_2more = ifelse(bsxg13b %in% c("None", "1 or 2 days"), "No", "Yes"),
         absent_2more = factor(absent_2more, levels = c("No", "Yes")),
         # ever repeated grade in either primary school or middle school
         repeat_ever = ifelse(bsxg14a == "Yes" | bsxg14b == "Yes", "Yes", "No"),
         repeat_ever = factor(repeat_ever, levels = c("No", "Yes"))
         ) %>% 
  select(-c(bsmmat01, bsmmat02, bsmmat03, bsmmat04, bsmmat05))


# depending on which model to run, i will need to remove a few variables

#low-math: exclude bsmmat_mean, math_ach_lvl_blwhigh, absent_ever, absent_2more, repeat_ever
table(T15_USA_df_stu_sch_clean$math_ach_lvl_blwhigh)
#absent_ever: exclude math_ach_lvl_blwhigh, absent_2more, repeat_ever
table(T15_USA_df_stu_sch_clean$absent_ever)
#absent_2more: exclude math_ach_lvl_blwhigh, absent_ever, repeat_ever
table(T15_USA_df_stu_sch_clean$absent_2more)
#repeat_ever: exclude math_ach_lvl_blwhigh, absent_ever, absent_2more
table(T15_USA_df_stu_sch_clean$repeat_ever)




### 2b. Scaling independent variables
## identify vars that are ordinal (to turn as double/continuous)

# 1. turn all interger into double (so there are only double and factor types)
is_int <- sapply(T15_USA_df_stu_sch_clean, is.integer)
T15_USA_df_stu_sch_clean <- T15_USA_df_stu_sch_clean[, is_int] %>% 
  mutate_all(as.double) %>% 
  bind_cols(T15_USA_df_stu_sch_clean[,!is_int])

# 2. for factors that has more than 2 levels and that are ordinal, turn them as double
level_morethan2 <- sapply(T15_USA_df_stu_sch_clean, nlevels) > 2
# some of the these variables are not ordinal (thus should not be converted to numerics)
level_morethan2_nonOrdinal <- c(#NAT\HIGHEST LVL OF EDU OF MOTHER/FATHER (from US PUF)
                                "bsng07a", "bsng07b",
                                #GEN\MOTHER/FATHER BORN IN <COUNTRY>
                                "bsbg09a", "bsbg09b",
                                #*NAT\DERIVED RACE-COLLAPSED* (from US PUF)
                                "msrace2",
                                #NAT\TYPE OF SCHOOL (from US PUF)
                                "bcxg07")
level_morethan2[names(level_morethan2) %in% level_morethan2_nonOrdinal] <- FALSE
# turn all ordinal variables into continous
T15_USA_df_stu_sch_clean <- T15_USA_df_stu_sch_clean[, level_morethan2] %>%
  # transform some variables into numeric
  mutate_all(as.double) %>%
  bind_cols(T15_USA_df_stu_sch_clean[, !level_morethan2]) 


# 3. for factors that has more than 2 levels but are not ordinal, dummy them out
# dummy out var in `level_morethan2_nonOrdinal`
#for the international variables `bsbg09a` and `bsbg09b` in `level_morethan2_nonOrdinal`, they have two classes, which will confuses the onehot() function. ##YL: check with Trang to make sure the 'lfactor' classes in other vars won't affect modelling
class(T15_USA_df_stu_sch_clean$bsbg09a) <- "factor"
class(T15_USA_df_stu_sch_clean$bsbg09b) <- "factor"

#one hot encoding
temp <- onehot(T15_USA_df_stu_sch_clean[, level_morethan2_nonOrdinal])
temp2 <- temp %>% 
  predict(T15_USA_df_stu_sch_clean) %>% 
  as.data.frame() %>% 
  #de-select reference categories
  select(-c(`bsng07a=I don't know`, `bsng07b=I don't know`, `bsbg09a=I DON'T KNOW`, `bsbg09b=I DON'T KNOW`, `msrace2=Other`, `bcxg07=Other`)) 
#turn 0 and 1 into No and Yes to be consistent with other two-level factors
temp2 <- as.data.frame(sapply(temp2, function(x){x <- ifelse(x == 1, "Yes", "No")
                                                 x <- factor(x, levels = c("No", "Yes"))
                                                 #x <- relevel(x, ref = "Yes")
                                                 })) ##YL: no matter what i try to make the order "yes" and "no", it won't work. i then decide to change evertying into the order or No and Yes so issue fixed. but need to check with Trang at somepoint to figure out what is not working

names(temp2) <- make.names(names(temp2), unique = TRUE)

#merge it back with the remaining columns
T15_USA_df_stu_sch_clean <- temp2 %>% 
  bind_cols(T15_USA_df_stu_sch_clean[, !(names(T15_USA_df_stu_sch_clean) %in% level_morethan2_nonOrdinal) ] ) 


# 4. check all independent variables should either be double (ordinal), or factor (dummy)
#drop two variables because they only have 1 level #bcxg21a & bcxg20a
T15_USA_df_stu_sch_clean <- T15_USA_df_stu_sch_clean[ , !(names(T15_USA_df_stu_sch_clean) %in% c("bcxg21a", "bcxg20a"))]

is_factor <- sapply(T15_USA_df_stu_sch_clean, is.factor)
sum(sapply( T15_USA_df_stu_sch_clean[ , is_factor], nlevels) < 2) 

#no more factors that has more than 2 levels
sum( sapply( T15_USA_df_stu_sch_clean[ , is_factor], nlevels) > 2)

#make sure all factor variables (2-level variables) have the level "No" "Yes"
ncol(T15_USA_df_stu_sch_clean[ , is_factor])

# make the following vars FALSE because their levels are not yes/no
is_factor[names(is_factor) %in% c("itsex", "pubpriv", "math_ach_lvl_blwhigh")] <- FALSE

is_factor_cols <- T15_USA_df_stu_sch_clean[ , is_factor]
sapply(is_factor_cols, levels)

is_factor_cols <- as.data.frame(sapply( is_factor_cols, function(x){x <- ifelse(x %in% c("YES", "Yes"), "Yes", "No")
                                                            x <- factor(x, levels = c("No", "Yes"))
                                                                    }))
T15_USA_df_stu_sch_clean <- is_factor_cols %>% 
  bind_cols(T15_USA_df_stu_sch_clean[, !is_factor ] ) 


# check the missing rate for each variable, just in case
colMeans(missing_rule(T15_USA_df_stu_sch_clean,om)) %>% sum()


# 2d. Prepocess data
# identify and remove columns with near zero variance (otherwise the models later won't run on this constant terms)
nzv <- nearZeroVar(T15_USA_df_stu_sch_clean, freqCut = 99/1) #set the threshold high, rationale being as long as it is not absolute zero, it should be brought to the model
nzv
#T15_USA_df_clean <- T15_USA_df_stu_sch_clean[,-c(nzv)]

process_configuration <- preProcess(T15_USA_df_stu_sch_clean,
                                    method = c("center", "scale"))
processed <- predict(process_configuration, T15_USA_df_stu_sch_clean)

# Step 3. Feature engineering =====
# 3a. Correlational matrix
# Goal: Idenitfy pairs of highly correlated variables
is_double <- sapply(processed, is.double)


correlationMatrix <- cor(processed[, is_double])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.95)  
processed_dbl_removeHighlyCorr <- names(processed[, is_double][highlyCorrelated])

processed <- processed[ , !(names(processed) %in% processed_dbl_removeHighlyCorr)]


### Step 4. Model building ===========
# 4a. data split

#low-math: exclude bsmmat_mean, math_ach_lvl_blwhigh, absent_ever, absent_2more, repeat_ever
low_math <- processed[ , !names(processed) %in% c("bsmmat_mean", "absent_ever", "absent_2more", "repeat_ever")]
#absent_ever: exclude math_ach_lvl_blwhigh, absent_2more, repeat_ever, bsxg13b
absent_e <- processed[ , !names(processed) %in% c("math_ach_lvl_blwhigh", "absent_2more", "repeat_ever", "bsxg13b", "bsbg11" )]
#absent_2more: exclude math_ach_lvl_blwhigh, absent_ever, repeat_ever, bsxg13b
absent_2 <- processed[ , !names(processed) %in% c("math_ach_lvl_blwhigh", "absent_ever", "repeat_ever", "bsxg13b", "bsbg11")]
#repeat_ever: exclude math_ach_lvl_blwhigh, absent_ever, absent_2more, bsxg14a
repeat_e <- processed[ , !names(processed) %in% c("math_ach_lvl_blwhigh", "absent_ever", "absent_2more", "bsxg14a")]



#
set.seed(817)

trainingIndex_low_math <- createDataPartition(low_math$math_ach_lvl_blwhigh, p = 0.8, list = FALSE)
training_low_math <- low_math[trainingIndex_low_math, ]
test_low_math <- low_math[-trainingIndex_low_math, ]

trainingIndex_absent_e <- createDataPartition(absent_e$absent_ever, p = 0.8, list = FALSE)
training_absent_e <- absent_e[trainingIndex_absent_e, ]
test_absent_e <- absent_e[-trainingIndex_absent_e, ]

trainingIndex_absent_2 <- createDataPartition(absent_2$absent_2more, p = 0.8, list = FALSE)
training_absent_2 <- absent_2[trainingIndex_absent_2, ]
test_absent_2 <- absent_2[-trainingIndex_absent_2, ]

trainingIndex_repeat_e <- createDataPartition(repeat_e$repeat_ever, p = 0.8, list = FALSE)
training_repeat_e <- repeat_e[trainingIndex_repeat_e, ]
test_repeat_e <- repeat_e[-trainingIndex_repeat_e, ]




### Decision Tree (CART) -----
#rpart is having difficulties with non-standard variable names, so using the non-formula method instead
# 
# rpart_low_math <- train(y = training_low_math$math_ach_lvl_blwhigh,
#                x = training_low_math[ , names(training_low_math) != "math_ach_lvl_blwhigh"],
#                method = "rpart",
#                metric = "Accuracy",
#                trControl=trainControl(method="cv", number=5),
#                tuneLength = 20)
# 
# rpart_low_math
# 
# # Apply the model to the test set & see result
# rpart_low_math_result <- confusionMatrix(table(predict(rpart_low_math, test_low_math) , test_low_math$math_ach_lvl_blwhigh), positive = "low")
# 
# 
# rpart_absent_e <- train(y = training_absent_e$absent_ever,
#                         x = training_absent_e[ , names(training_absent_e) != "absent_ever"],
#                         method = "rpart",
#                         metric = "Accuracy",
#                         trControl=trainControl(method="cv", number=5),
#                         tuneLength = 20)
# 
# rpart_absent_e
# 
# rpart_absent_e_result <- confusionMatrix(table(predict(rpart_absent_e, test_absent_e) , test_absent_e$absent_ever), positive = "Yes")




### write a function
model <- function(y, trainingData, testData, methodName, tunelgth = 5, positive, ...){
  set.seed(817)
  
  if(methodName %in% c("rpart")){
    m <- caret::train(y = trainingData[ , y],
                      x = trainingData[ , names(trainingData) != y],
                      method = methodName,
                      metric = "Accuracy",
                      trControl=trainControl(method="cv", number=5),
                      tuneLength = tunelgth)
  } else if(methodName %in% c("glm")){
    m <- caret::train(form =  formula(paste0(y , " ~.")),
                      data = trainingData,
                      method = methodName,
                      #metric = "Accuracy",
                      #trControl=trainControl(method="cv", number=5),
                      #tuneLength = tunelgth,
                      family = binomial) 
    } else if(methodName %in% c("xgbLinear")){
      m <- caret::train(form =  formula(paste0(y , " ~.")),
                        data = trainingData,
                        method = methodName,
                        metric = "Accuracy",
                        trControl=trainControl(method="cv", number=5),
                        tuneLength = 5)
      
    } else {
    m <- caret::train(form =  formula(paste0(y , " ~.")),
                      data = trainingData,
                      method = methodName,
                      metric = "Accuracy",
                      trControl=trainControl(method="cv", number=5),
                      tuneLength = tunelgth) }
  
  #output model result
  assign(paste0(methodName, "_", y), m, envir = .GlobalEnv)
  #get prediction and output confusion matrix
  p <- predict(m, testData)
  
  cm <- confusionMatrix(table(p , testData[ , y] ), positive = positive)
  assign(paste0(methodName, "_", y, "_cm"), cm, envir = .GlobalEnv)
  
}



# # experiment
# model(y = "math_ach_lvl_blwhigh",
#       trainingData = training_low_math,
#       testData = test_low_math,
#       methodName = "rpart",
#       tunelgth = 20,
#       positive = "low")
# 
# model(y = "absent_ever",
#       trainingData = training_absent_e,
#       testData = test_absent_e,
#       methodName = "glm",
#       positive = "Yes")

###loop over
list <- c("rpart", "rf", "nnet", "xgbLinear", "glmnet", "glm")
#"svmLinear"

#low_math 
for (m in list){
  print(m)
  model(y = "math_ach_lvl_blwhigh",
        trainingData = training_low_math,
        testData = test_low_math,
        methodName = m,
        tunelgth = 20,
        positive = "low")
}
Sys.time()

#absent_ever
for (m in list){
  print(m)
  model(y = "absent_ever",
        trainingData = training_absent_e,
        testData = test_absent_e,
        methodName = m,
        tunelgth = 10,
        positive = "Yes")
}
Sys.time()

#absent_2more
for (m in list){
  print(m)
  model(y = "absent_2more",
        trainingData = training_absent_2,
        testData = test_absent_2,
        methodName = m,
        tunelgth = 10,
        positive = "Yes")
}
Sys.time()

#repeat_ever
for (m in list){
  print(m)
  model(y = "repeat_ever",
        trainingData = training_repeat_e,
        testData = test_repeat_e,
        methodName = m,
        tunelgth = 10,
        positive = "Yes")
}
Sys.time()












### Random Forest -----
# try using tuneLength #mtry = 83 is used in the end
rf <- train(math_ach_lvl_blwhigh~. -bsmmat_mean -absent_ever -absent_2more -repeat_ever, 
             data=training, 
             method="rf", 
             metric="Accuracy", 
             #tuneGrid=expand.grid(.mtry=mtry), 
             trControl=trainControl(method="oob", number=25),
             #default to be 500
             ntree = 500,
             tuneLength = 2)
print(rf)

test_pred_rf <- predict(rf, test)
result_rf <- confusionMatrix(table(test_pred_rf , test$math_ach_lvl_blwhigh ), positive = "low")

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
# ada <- train(math_ach_lvl_blwhigh ~., 
#                   data = training, 
#                   method = "ada",
#                   metric = "Sens",
#                   trControl=trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction = twoClassSummary),
#                   tuneLength = 5)
# 
# 
# ada

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

allModels <- list(rpart_math_ach_lvl_blwhigh, 
                  rf_math_ach_lvl_blwhigh,
                  nnet_math_ach_lvl_blwhigh,
                  xgbLinear_math_ach_lvl_blwhigh,
                  #svmLinear_math_ach_lvl_blwhigh,
                  glmnet_math_ach_lvl_blwhigh,
                  glm_math_ach_lvl_blwhigh)
allModels <- list(rpart_absent_ever, 
                  rf_absent_ever,
                  nnet_absent_ever,
                  xgbLinear_absent_ever,
                  glmnet_absent_ever,
                  glm_absent_ever)
allModels <- list(rpart_absent_2more, 
                  rf_absent_2more,
                  nnet_absent_2more,
                  xgbLinear_absent_2more,
                  glmnet_absent_2more,
                  glm_absent_2more)
allModels <- list(rpart_repeat_ever, 
                  rf_repeat_ever,
                  nnet_repeat_ever,
                  xgbLinear_repeat_ever,
                  glmnet_repeat_ever,
                  glm_repeat_ever)
                  
allResults <- list(rpart_math_ach_lvl_blwhigh_cm, 
                   rf_math_ach_lvl_blwhigh_cm,
                   nnet_math_ach_lvl_blwhigh_cm,
                   xgbLinear_math_ach_lvl_blwhigh_cm,
                   #svmLinear_math_ach_lvl_blwhigh_cm,
                   glmnet_math_ach_lvl_blwhigh_cm,
                   glm_math_ach_lvl_blwhigh_cm)
allResults <- list(rpart_absent_ever_cm, 
                   rf_absent_ever_cm,
                   nnet_absent_ever_cm,
                   xgbLinear_absent_ever_cm,
                   glmnet_absent_ever_cm,
                   glm_absent_ever_cm)
allResults <- list(rpart_absent_2more_cm, 
                   rf_absent_2more_cm,
                   nnet_absent_2more_cm,
                   xgbLinear_absent_2more_cm,
                   glmnet_absent_2more_cm,
                   glm_absent_2more_cm)
allResults <- list(rpart_repeat_ever_cm, 
                   rf_repeat_ever_cm,
                   nnet_repeat_ever_cm,
                   xgbLinear_repeat_ever_cm,
                   glmnet_repeat_ever_cm,
                   glm_repeat_ever_cm)


for (i in 1:length(allResults)){
  temp <- allResults[[i]]$overall["Accuracy"]
  print(temp)
}
for (i in 1:length(allResults)){
  temp <- allResults[[i]]$byClass["Balanced Accuracy"]
  print(temp)
}




### Variable Importance =====
# sapply(allModels, varImp)
# library(data.table)
# 
# for (i in 1:length(allModels)){
#   varimp <- varImp(allModels[[i]])
#   df <- as.data.frame(varimp$importance)
#   df <- setDT(df, keep.rownames = TRUE)
#   colnames(df) <- c("var", "imp")
# }

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
  df <- df %>% dplyr::arrange(desc(imp))
  
  return(df)
}

varimp <- lapply(allModels, varimp_result)

# find top 20 vars
top_20 <- function(x){
  if (length(x$var) >= 20){
    top20 <- x$var[1:20]
  } else
    top20 <- x$var[1:length(x$var)]
}

top20_list <- sapply(varimp, top_20)


top20_list

setdiff(top20_list[,1], top20_list[,2])

#exist across all lists
top20_list[,1] %>% 
  intersect(top20_list[,2]) %>% 
  intersect(top20_list[,3]) %>% 
  intersect(top20_list[,4]) %>% 
  intersect(top20_list[,5]) %>% 
  intersect(top20_list[,6])

#Reduce(intersect, top20_list)
#Reduce(union, top20_list)
variablelist <- top20_list[,1] %>% 
  union(top20_list[,2]) %>% 
  union(top20_list[,3]) %>% 
  union(top20_list[,4]) %>% 
  union(top20_list[,5]) %>% 
  union(top20_list[,6])

variableFreq <- sapply(variablelist, function(v) {
  return(sum(sapply(1:6, function(i) v %in% top20_list[,i])))
})
variableTable <- data.frame(variableName=variablelist,
                       freq=variableFreq
                       )










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





