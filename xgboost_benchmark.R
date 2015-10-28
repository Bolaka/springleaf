# clear the memory!
rm(list=ls(all=TRUE))
gc()

library(caret)
library(readr)
library(xgboost)
library(plyr)

set.seed(786)

# set the work directory
setwd("~/Backup/datasets/Springleaf")

cat("reading the train and test data\n")
# train <- read_csv("train.csv")
# save(train,file="train.Rda")
load("train.Rda")

# test  <- read_csv("test.csv")
# save(test,file="test.Rda")

load("test.Rda")

outliersTolerance <- 5
varianceTolerance <- 0.009

# na_count <- sapply(train, function(x) sum(is.na(x)))
# na_count <- data.frame(na_count)
# na_count$cols <- rownames(na_count)
# total_len <- length(rownames(na_count))
# sparse_cols <- na_count$cols[na_count$na_count > total_len/2]
# train <- train[ , -which(names(train) %in% sparse_cols)]
# test <- test[ , -which(names(test) %in% sparse_cols)]

# # parse dates
# date_features <- c('VAR_0073', 'VAR_0075', 'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159', 'VAR_0166', 'VAR_0167', 
#                    'VAR_0168', 'VAR_0169', 'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179', 'VAR_0204', 'VAR_0217', 
#                    'VAR_0531' )
# for (date_f in date_features) {
# #     print(date_f)
#     if (date_f == 'VAR_0531') {
#         train[[paste(date_f, 'year', sep="_")]] <- as.numeric(substr(train[[date_f]],1,4))
#         train[[paste(date_f, 'month', sep="_")]] <- as.numeric(substr(train[[date_f]],5,6))
#         
#         test[[paste(date_f, 'year', sep="_")]] <- as.numeric(substr(test[[date_f]],1,4))
#         test[[paste(date_f, 'month', sep="_")]] <- as.numeric(substr(test[[date_f]],5,6))
#         
#         train[[date_f]] <- NULL
#         test[[date_f]] <- NULL
#     } else {
#         train[[date_f]] <- as.numeric(as.Date(train[[date_f]],format="%d%b%y:%H:%M:%S"))
#         test[[date_f]] <- as.numeric(as.Date(test[[date_f]],format="%d%b%y:%H:%M:%S"))
#     }
# }
# 
# # reordering columns so that 'target' is the last one
# temp <- train[['target']]
# train[['target']] <- NULL
# train[['target']] <- temp
# 
# temp <- test[['target']]
# test[['target']] <- NULL
# test[['target']] <- temp

feature.names <- names(train)[2:ncol(train)-1]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
c <- 0
for (f in feature.names) {
#     print(class(train[[f]]))
    # levels
    levelsZ <- unique(c(train[[f]], test[[f]]))
    levelsX <- unique(train[[f]])
    anyNA <- any(is.na(levelsX))
    
    if (length(levelsX) == 1 | (length(levelsX) == 2 & anyNA == TRUE)) {
        train[[f]] <- NULL
        test[[f]] <- NULL
        c <- c + 1
    }

    # numerically label categories for now
    # TODO deal properly later...
    if (class(train[[f]])=="character") {
        # check for zero variance
        if (f == 'VAR_0044') {
            train[[f]] <- NULL
            test[[f]] <- NULL
            c <- c + 1
        } else {
            train[[f]] <- as.integer(factor(train[[f]], levels=levelsZ))
            test[[f]]  <- as.integer(factor(test[[f]],  levels=levelsZ))
            
            # check class after conversion??
        }
    } 
    # remove outliers from numerical
    else if (class(train[[f]])=="integer") {
#         print(f)
#         train[[f]] <- log1p(train[[f]])
#         test[[f]] <- log1p(test[[f]])
        x <- train[[f]]
        y <- test[[f]]
        z <- c(x, y)
        varZ <- var(z,na.rm = T)
        
        if (is.na(varZ) | varZ < varianceTolerance) {
            train[[f]] <- NULL
            test[[f]] <- NULL
            c <- c + 1
        } else {
            meanZ <- mean(z,na.rm = T)
            stdZ <- sd(z,na.rm = T)
            if (any(x > (meanZ + outliersTolerance*stdZ) | x < (meanZ - outliersTolerance*stdZ), na.rm = T)) {
                cat(f, 'train ')
                train[[f]] <- replace(x, x > (meanZ + outliersTolerance*stdZ), (meanZ + outliersTolerance*stdZ)) # 
                train[[f]] <- replace(x, x < (meanZ - outliersTolerance*stdZ), (meanZ - outliersTolerance*stdZ)) # 
            } 
            if (any(y > (meanZ + outliersTolerance*stdZ) | y < (meanZ - outliersTolerance*stdZ), na.rm = T)) {
#                 cat(f, 'test\n')
                test[[f]] <- replace(y, y > (meanZ + outliersTolerance*stdZ), (meanZ + outliersTolerance*stdZ)) # 
                test[[f]] <- replace(y, y < (meanZ - outliersTolerance*stdZ), (meanZ - outliersTolerance*stdZ)) # 
            }
        }
    }
    
    if ( any(is.na(train[[f]])) | any(is.na(test[[f]]))) {
        x <- train[[f]]
        y <- test[[f]]
        medianX <- median(x,na.rm = T)
        train[[f]] <- replace(x, is.na(x), medianX)
        test[[f]] <- replace(y, is.na(y), medianX)
        # median !
    }
    levelsX <- unique(train[[f]])
    if (length(levelsX) == 1) {
        print(f)
        train[[f]] <- NULL
        test[[f]] <- NULL
        c <- c + 1
    }
}
cat('\ndropped ', c, ' columns!')

feature.names <- names(train)[2:ncol(train)-1]
rm(levelsX, levelsZ, x, y, z, meanZ, stdZ)

# 2:35 pm
# princ <- prcomp(train, scale. = TRUE)

# folds <- createFolds(factor(train$target), k = 2, list = TRUE, returnTrain = TRUE)
# train <- train[sample(nrow(train), 40000),]

cat("Making train and validation matrices\n")
inTraining <- createDataPartition(factor(train$target), p = 0.80, list = FALSE)
# validation  <- train[-inTraining,]
# train <- train[ inTraining,]

gc()

# # check stratification...
# table(train$target)
# table(validation$target)

# as DMatrix
dtrain <- xgb.DMatrix(data.matrix(train[inTraining, feature.names]), 
                      label=train[inTraining, 'target']) # , missing = NaN
dval <- xgb.DMatrix(data.matrix(train[-inTraining,feature.names]), 
                    label=train[-inTraining, 'target']) # , missing = NaN

# # save to speed up things
# xgb.DMatrix.save(dtrain, "dtrain.buffer")
# xgb.DMatrix.save(dval, "dval.buffer")

# # to load it in, simply call xgb.DMatrix
# dtrain <- xgb.DMatrix("dtrain.buffer")
# dval <- xgb.DMatrix("dval.buffer")

# sum of positives in training
lbls <- getinfo(dtrain, 'label')
sum_pos = sum(lbls)
sum_neg = length(lbls) - sum_pos

watchlist <- list(eval = dval, train = dtrain)

param <- list(  objective           = "binary:logistic", 
                # booster = "gblinear",
                eta                 = 0.05, # changed from default of 0.001
                max_depth           = 7, # changed from default of 14
                subsample           = 0.7, # changed from default of 0.6
                colsample_bytree    = 0.7, # changed from default of 0.6
                eval_metric         = "auc"
#                 scale_pos_weight    = (sum_neg/sum_pos)
#                 max_delta_step      = 3
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 500, # changed from 50
                    verbose             = 2, 
                    early.stop.round    = 10,
                    watchlist           = watchlist,
                    maximize            = TRUE)
# [99]    eval-auc:0.779890	train-auc:0.964216 | 10 max_depth
# [99]    eval-auc:0.780063	train-auc:0.848016 | 7 max_depth
# [99]    eval-auc:0.779585	train-auc:0.845527 | ignore missing x
# [99]    eval-auc:0.779825	train-auc:0.848777 | dates - v1
# [99]    eval-auc:0.779729	train-auc:0.848780 | remove features with variance < 0.1
# [99]    eval-auc:0.777182	train-auc:0.817733 | 6 max_depth x
# [99]    eval-auc:0.778...  train-auc:0.848... | remove features with variance < 0.1 + 212 features
# [99]    eval-auc:0.779273	train-auc:0.848371 | median!
# [99]    eval-auc:0.779822	train-auc:0.848737 | median + remove features with variance < 0.05 - 175 features v
# [99]    eval-auc:0.779857	train-auc:0.849072 | without dates parsing and remove single levels at top x
# [99]    eval-auc:0.779800	train-auc:0.887914 | max_depth = 8 x
# [99]    eval-auc:0.779799	train-auc:0.848162 | replace outliers ny NaN too x
# [99]    eval-auc:0.779721	train-auc:0.847087 | 0.005 x
# [199]    eval-auc:0.785711	train-auc:0.889540 | 0.009, w/o dates parsing, 200
# [242]    eval-auc:0.786171	train-auc:0.901761
# Stopping. Best iteration: 233

# importance_matrix <- xgb.importance(feature.names, model = clf)
# xgb.plot.importance(importance_matrix, numberOfClusters = 4)

# cat("training a XGBoost classifier\n")
# clf <- xgboost(data        = data.matrix(train[,feature.names]),
#                label       = train$target,
#                nrounds     = 20,
#                objective   = "binary:logistic",
#                eval_metric = "auc",
#                verbose = 1)

cat("making predictions in batches due to 8GB memory limitation\n")

submission <- data.frame(ID=test$ID)
submission$target <- NA 
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
    submission[rows, "target"] <- predict(clf, data.matrix(test[rows,feature.names]))
}
    
# # benchmark = 0.76178
# 0.76867 (local) = 0.77042 (public LB)
cat("saving the submission file\n")
write_csv(submission, "xgboost_submission_200.csv")
