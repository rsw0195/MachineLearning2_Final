rm(list=ls())
options(warn=-1)   # Supress warning messages
###############################################
############# Functions #######################
###############################################


installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}

needed <- c("glmulti", "leaps","caret", "kernlab","doMC",
            "e1071",'AppliedPredictiveModeling',"mlbench",
            'earth',"doParallel","snow","ROCR","bst","plyr",
            "rpart","MASS","rattle","randomForest", "pROC", 
            "verification", "rpart","doMC","ada","e1071",
            "randomForest","gbm","xgboost","MASS",'tree','kLaR',
            'xgboost')  
installIfAbsentAndLoad(needed)

confusion <- function(true, pred, title = NULL, type) {
  if (is.null(title)) {
    conf.table <- table(true, pred, dnn = c('True', 'Pred'))
  } else {
    conf.table <- table(true, pred, dnn = c('', title))
  }
  
  acc <- (conf.table[1] + conf.table[4])/sum(conf.table)
  err <- (conf.table[2] + conf.table[3])/sum(conf.table)
  precision <- conf.table[2,2]/sum(conf.table[,2])
  type1 <- conf.table[1,2]/sum(conf.table[1,])
  type2 <- conf.table[2,1]/sum(conf.table[2,])
  power <- 1 - type2
  true_pos <- conf.table[2,2]/sum(conf.table[2,])
  true_neg <- conf.table[1,1]/sum(conf.table[1,])
  
  if (type == 1) {
    return(c(type2, type1, true_pos, true_neg, acc))
  } else {
    print(paste('Overall Accuracy:', acc))
    print(paste('Overall Error Rate:', err))
    print(paste('Type 1 Error Rate:', type1))
    print(paste('Type 2 Error Rate:', type2))
    print(paste('Power:', power))
    print(paste('Precision:', precision))
    return(conf.table)
  }
}

#####################################################################################
###################### Machine Learning 2 Final Project #############################
############################ Email SPAM Filtering ###################################
#####################################################################################
############## Team 03-01: Rebecca Wood, Eric Fung & Michael McKenna ################
#####################################################################################


## Enable Multiproccessing
cores=detectCores()
registerDoMC(cores=cores)
registerDoParallel(cores=detectCores())

## Read in Data
emails <- read.csv("email.csv", header=FALSE, sep=";")
names(emails)[58]<-'y'

## Data Transformation: Predictor to Factor ##
emails$y <- as.factor(emails$y)

## Data Partitioning: Create Training and Test Sets ##
train <- createDataPartition(emails$y,p = .8, list = FALSE, times = 1)
train.data <- emails[train, ]
test.data <- emails[-train, ]

####################################
####### Models in This Script ######
####################################

    # 1) Naive Bayes
    # 2) GLM - Bayesian
    # 3) GAM - Boosted
    # 4) SVM - Linear, Radial, Polynomial
    # 5) Classification Trees - Information, Gini
    # 6) Random Forest
    # 7) Boosted Trees - ada, gbm, xgboost

##########################################################
##########################################################
################ MODEL 1: Naive Bayes ####################
##########################################################
##########################################################

Models <- matrix(ncol=3,nrow=12)
row.names(Models)<-c("NaiveBayes","GLM", "GAM", "SVMLinear",
                     "SVMRadial","SVMPolynomial","TreeInformation",
                     "TreeGini","RandomForest","AdaBoost","GbmBoost",
                     "XgBoost")
colnames(Models)<-c("Accuracy","Type1","Type2")

##########################################################
################ Training the Model ######################
##########################################################

t.control <- trainControl(method = "repeatedcv", 
                          number = 10, repeats = 3)

BayesModel = train(y ~., data = train.data, method='nb',
              trControl=t.control,
              preProcess = c("pca","center", "scale"),
                                 tuneLength = 10)
          
          ###################################
          ########  TRAINING DETAILS ########
          ## Repeated 10-fold CV (3 times) ##
          ## Data is centered & scaled ######
          ## Processed predictors with PCA ##
          ###################################

       
########################################################
############## Training Set Prediction #################
########################################################

Bayes.train <- predict(BayesModel, newdata = train.data, 
                       type="raw")
confusionMatrix(Bayes.train, train.data$y)

##########################################################
################# Test Set Prediction ####################
##########################################################

  Bayes.test <-predict(BayesModel, newdata=test.data, 
                       type='raw')
  conf.table <- table(test.data$y, Bayes.test, 
                      dnn = c('True', 'Pred'))
  
  Models[1,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
  Models[1,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
  Models[1,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100
  confusionMatrix(Bayes.test,test.data$y)
  
  ## 87% Accuracy
  ## 15% Type 1
  ## 10% Type 2

#################################################################
#################################################################
#################### MODEL 2: Bayes GLM  ########################
#################################################################
#################################################################

    ######################################################
    ################ Training the Model ##################
    ######################################################
  
    t.control <- trainControl(method = "repeatedcv", 
                            number = 10, repeats = 3)
    set.seed(1)
    glmModel <- train(y ~., data = train.data, method="bayesglm",
                        trControl=t.control,
                        preProcess = c("pca","center", "scale"),
                        tuneLength = 10)
  
              ###################################
              ########  TRAINING DETAILS ########
              ## Repeated 10-fold CV (3 times) ##
              ## Data is centered & scaled ######
              ## Processed predictors with PCA ##
              ###################################
              
########################################################
############## Training Set Prediction #################
########################################################
  
glm.train <- predict(glmModel, newdata = train.data, 
                     type="raw")
confusionMatrix(glm.train, train.data$y)
  
##########################################################
################# Test Set Prediction ####################
##########################################################
  
  glm.test <- predict(glmModel, newdata = test.data)
  conf.table <- table(test.data$y, glm.test, 
                      dnn = c('True', 'Pred'))

  Models[2,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
  Models[2,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
  Models[2,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100
  
  confusionMatrix(glm.test,test.data$y)
  
  # 92% Accuracy
  # 5%  Type 1
  # 11% Type 2
  
#################################################################
#################################################################
#################### MODEL 3: GAM Boost  ########################
#################################################################
#################################################################
  
  ######################################################
  ################ Training the Model ##################
  ######################################################
  
    set.seed(1)
    t.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3)
    
    gamModel = train(y ~., data = train.data,method='gamboost',
                trControl=t.control,
                preProcess = c("pca","center", "scale"),
                tuneLength = 10)
  
            ###################################
            ########  TRAINING DETAILS ########
            ## Repeated 10-fold CV (3 times) ##
            ## Data is centered & scaled ######
            ## Processed predictors with PCA ##
            ###################################
  
########################################################
############## Training Set Prediction #################
########################################################
  
gam.train <- predict(gamModel, newdata=train.data, 
                     type="raw")
confusionMatrix(gam.train, train.data$y)
  
##########################################################
################# Test Set Prediction ####################
##########################################################
    
  gam.test <- predict(gamModel, newdata=test.data)
  conf.table <- table(test.data$y, gam.test, 
                        dnn = c('True', 'Pred'))
    
  Models[3,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
  Models[3,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
  Models[3,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100
  
  confusionMatrix(gam.test,test.data$y)
    
#################################################################
#################################################################
################### MODEL 4: Linear SVM  ########################
#################################################################
#################################################################
  
  ######################################################
  ################ Training the Model ##################
  ######################################################
  
    t.control <- trainControl(method = "repeatedcv", 
                            number = 10, repeats = 3)
   #tune.grid <- expand.grid(C=seq(.01,3,.01))
  
    tune.grid <- expand.grid(C=c(.8,.9,1))
                             #1.1,1.2,1.3,1.4))
  
    set.seed(1)
    SVMlinearModel <- train(y ~., data = train.data, 
                            method = "svmLinear",trControl=t.control,
                            preProcess = c("pca","center", "scale"),
                            tuneGrid = tune.grid,
                            tuneLength = 10)
          
          ###################################
          ########  TRAINING DETAILS ########
          ## Repeated 10-fold CV (3 times) ##
          ## Data is centered & scaled ######
          ## Processed predictors with PCA ##
          ## Tuned for best values of C #####
          ###################################

###############################################
####### PLOT THE COST VS ACCURACY##############
###############################################
    
plot(SVMlinearModel)
    
########################################################
############## Training Set Prediction #################
########################################################
    
linearSVM.train <- predict(SVMlinearModel, newdata=train.data, 
                           type="raw")
confusionMatrix(linearSVM.train, train.data$y)
    
##########################################################
################# Test Set Prediction ####################
##########################################################
    
linearSVM.test <- predict(SVMlinearModel, newdata=test.data)
conf.table <- table(test.data$y, linearSVM.test, 
                        dnn = c('True', 'Pred'))
    
    Models[4,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
    Models[4,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
    Models[4,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100
    
    confusionMatrix(linearSVM.test,test.data$y)

#################################################################
#################################################################
################### MODEL 5: Radial SVM  ########################
#################################################################
#################################################################
    
    ######################################################
    ################ Training the Model ##################
    ######################################################
    
      t.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3)
          
      tune.radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 
                                           0.03, 0.04,0.05, 0.06,
                                           0.07,0.08, 0.09, 0.1, 
                                           0.25, 0.5, 0.75,0.9),
                                 
                                    C = c(0,0.01, 0.05, 0.1, 0.25, 
                                          0.5, 0.75,1, 1.5, 2,5)
      )
  
  ## Optimal Parameters (From  ~ 4 Hours of Training)
  tune.radial <- expand.grid(sigma = .01, C=5)
  set.seed(1)
  SVMradialModel <- train(y ~., data = train.data, method = "svmRadial",
                         trControl=t.control,
                         preProcess = c("pca","center", "scale"),
                         tuneGrid = tune.radial,
                         tuneLength = 10)
  
              ###################################
              ########  TRAINING DETAILS ########
              ## Repeated 10-fold CV (3 times) ##
              ## Data is centered & scaled ######
              ## Processed predictors with PCA ##
              ## Tuned for best values of C #####
              ## Tuned for best values if sigma #
              ###################################

###############################################
####### PLOT THE COST VS ACCURACY #############
###############################################
  
  ### Will only Plot after training sigma/C values
  plot(SVMradialModel)   
  
########################################################
############## Training Set Prediction #################
########################################################
  
  radialSVM.train <- predict(SVMradialModel, newdata=train.data, 
                             type="raw")
  confusionMatrix(radialSVM.train, train.data$y)
  
##########################################################
################# Test Set Prediction ####################
##########################################################
  
  radialSVM.test <- predict(SVMradialModel, newdata=test.data)
  conf.table <- table(test.data$y, radialSVM.test, 
                      dnn = c('True', 'Pred'))
  
  Models[5,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
  Models[5,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
  Models[5,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100
  
  confusionMatrix(radialSVM.test,test.data$y)
  
 
#################################################################
#################################################################
################### MODEL 6: Polynomial SVM  ####################
#################################################################
#################################################################
  
  ######################################################
  ################ Training the Model ##################
  ######################################################
  
  t.control <- trainControl(method = "repeatedcv", 
                            number = 10, repeats = 3)
  
  tune.poly <- expand.grid(degree = c(2, 3, 4),
                            scale = c(0, .01, .02, .03, .07, .1, 
                                      .25, .5, .75, .9),
                                C = c(.01, .05, .1, .25, .5, 
                                      .75, 1, 1.5, 2, 5)
  )
  set.seed(1)
  #Best degree: 2
  #Best C: 5
  #Best scale: 01
  tune.poly <- expand.grid(degree = 2,
                           scale = .01,
                           C = seq(5,7,.1))
  
  SVMpolynomialModel <- train(y ~., data = train.data, method = "svmPoly",
                              trControl=t.control,
                              preProcess = c("pca","center", "scale"),
                              tuneGrid = tune.poly,
                              tuneLength = 10)
  
          ###################################
          ########  TRAINING DETAILS ########
          ## Repeated 10-fold CV (3 times) ##
          ## Data is centered & scaled ######
          ## Processed predictors with PCA ##
          ## Tuned for best values of C #####
          ## Tuned for best values of scale #
          ## Tuned for best degree value ####
          ###################################
  
###############################################
####### PLOT THE COST VS ACCURACY #############
###############################################
  
  ## WILL ONLY PLOT BEFORE RUNNING TRAINING MODEL 
  ## WITH OPTIMAL PARAMETERS (need tuning values)
  plot( SVMpolynomialModel)
  
########################################################
############## Training Set Prediction #################
########################################################
  
PolynomialSVM.train <- predict(SVMpolynomialModel,newdata=train.data,
                               type="raw")
confusionMatrix(PolynomialSVM.train, train.data$y)
  
##########################################################
################# Test Set Prediction ####################
##########################################################

  PolynomialSVM.test <- predict(SVMpolynomialModel, newdata=test.data)
  conf.table <- table(test.data$y, PolynomialSVM.test, 
                      dnn = c('True', 'Pred'))
  
  Models[6,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
  Models[6,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
  Models[6,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100
  
  confusionMatrix(PolynomialSVM.test,test.data$y)
  
#################################################################
#################################################################
############## MODELS 7-8: Classification Trees  ################
#################################################################
#################################################################
  
  ######################################################
  ################ Training the Models #################
  ###################################################### 

# Using both splitting parameters, find maximum accuracy
# on the training data by varying values for minbucket & minsplit.
# From this, the bestcp value can be obtained.

  parameters <- c("information", "gini")
  maxacc <- rep(0,2)
  minbucket <-rep(0,2)
  minsplit <-rep(0,2)
  bestcp <-rep(0,2)

  for(x in 1:length(parameters)){
  
    bucket<-seq(1,30,2)
    split<-seq(2,40,2)
    mat1<-expand.grid(bucket,split)
    mat1[3]<-rep(0,nrow(mat1))
    acc <- rep(0,nrow(mat1))
    colnames(mat1)<-c('bucket','split','cp.min')
  
    for(i in 1:nrow(mat1)){
      mymodel<-rpart(y~.,data=train.data,method="class",
                  parms=list(split=parameters[x]),
                  control=rpart.control(cp=0,
                                      minbucket=mat1[i,1],
                                      minsplit=mat1[i,2],
                                      byrow=TRUE, nrow=2))
  
      pred <- predict(mymodel, train.data, type="class")
      conf.table<-table(train.data$y, pred,
                    dnn=c("Actual", "Predicted"))
      acc[i] <- (conf.table[1] + conf.table[4])/sum(conf.table)
  
      xerr<-mymodel$cptable[,"xerror"]
      minxerr<-which.min(xerr)              
      mat1[i,3]<-mymodel$cptable[minxerr,"CP"]
  }
  
    maxacc[x] <- max(acc)
    mindata <- mat1[which.max(acc),]
    minbucket[x] <-mindata[1]
    minsplit[x] <-mindata[2]
    bestcp[x] <-mindata[3]
  
}
  ### Using Minbucket & Minsplit Data:
  ### Build New Models for Gini & Information
  ### Evaluate the Test Set Data

  mymodel <- list(0,0)
  mymodel.test <- list(0,0)
  conf.tables <- list(0,0)

  for(x in 1:length(parameters)){
    mymodel[[x]] <- rpart(y ~., data=train.data, method="class",
                  parms=list(split=parameters[x]),control=rpart.control
                  (cp=bestcp[x], minbucket=minbucket[x], minsplit=minsplit[x]))
    
    ###################################################
    ######### PRUNE CLASSIFICATION TREES ##############
    ###################################################
    
    mymodel.prune<-prune(mymodel[[x]],cp=bestcp[x])

##########################################################
################# Test Set Prediction ####################
##########################################################
    
    mymodel.test[[x]]<- predict(mymodel.prune, newdata=test.data, type="class")

    conf.tables[[x]] <- table(test.data$y, mymodel.test[[x]], dnn=c("Actual", "Predicted"))
    conf.table <- conf.tables[[x]]
 
    Models[(x+6),1] <- (conf.table[1] + conf.table[4])/sum(conf.table) * 100
    Models[(x+6),2] <- conf.table[1,2]/sum(conf.table[1,]) * 100
    Models[(x+6),3] <- conf.table[2,1]/sum(conf.table[2,]) * 100
    
    print(confusionMatrix(mymodel.test[[x]],test.data$y))
   
  }
  
for(x in 1:2){
  Models[(x+6),1] <- Models[(x+6),1] * 100
  Models[(x+6),2] <-Models[(x+6),2]* 100
  Models[(x+6),3] <-Models[(x+6),3] * 100
}

####################################
## Inspecting MissClassified Data ##
####################################

misclassified <- which(as.vector(mymodel.test[[1]]) != as.vector(test.data$y))
misclassified2 <- which(as.vector(mymodel.test[[2]]) != as.vector(test.data$y))
intersection <- intersect(misclassified,misclassified2)
new.predict1 <- predict(mymodel[1], newdata=test.data, type="prob")
new.predict2 <- predict(mymodel[2], newdata=test.data, type="prob")
new.predict1[[1]][misclassified,]
new.predict2[[1]][misclassified2,]
length(misclassified)
length(intersection)

#################################################################
#################################################################
################## MODEL 9: Random Forest #######################
#################################################################
#################################################################

######################################################
################ Training the Model ##################
###################################################### 

          #############################
          ## Find Optimal mtry value ##
          #############################

  min.err<-rep(0,ncol(train.data))
  min.err.idx<- rep(0,ncol(train.data))
  rferr<-matrix(ncol=3,nrow=ncol(train.data))

  for(i in 1:ncol(train.data)){
    f1 <- randomForest(y~.,data=train.data, ntree=500, mtry=i, 
                     importance=TRUE,localImp=TRUE,na.action=na.roughfix,
                     replace=FALSE) 
  
    min.err[i] <-min(f1$err.rate[,"OOB"])
    min.err.idx[i]<- which.min(f1$err.rate[,"OOB"])
    rferr[i,]<-f1$err.rate[min.err.idx[i],]
}

  ntree <- min.err.idx[which.min(min.err)]
  mtry <- which.min(min.err)

##########################################################
########## REBUILD TREE USING PARAMETERS #################
##########################################################
  
set.seed(1)
f2 <- randomForest(y ~ ., data=train.data, ntree=ntree, 
                   mtry=mtry, importance=TRUE, localImp=TRUE,
                   na.action=na.roughfix, replace=FALSE)

########################################################
############## Training Set Prediction #################
########################################################

  rftrain <- predict(f2, newdata=train.data)
  confusionMatrix(rftrain, train.data$y)

##########################################################
################# Test Set Prediction ####################
##########################################################

rftest <- predict(f2, newdata=test.data)
conf.table <- table(test.data$y, rftest, dnn=c("Actual", "Predicted"))

Models[9,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
Models[9,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
Models[9,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100

confusionMatrix(rftest,test.data$y)

#################################################################
#################################################################
################## MODEL 10: AdaBoost Tree ######################
#################################################################
#################################################################

  
  ######################################################
  ################ Training the Model ##################
  ###################################################### 


    ###############################################
    ## Use Optimal Data From Classification Tree ##
    ###############################################

    cp <- unlist(bestcp)[1]
    minsplit <- unlist(minsplit)[1]
          
    #############################
    ## Find Optimal nu value ####
    #############################

    nums <- c(.70,.71,.72,.75,.80,.90, 1)
    accur <- rep(0,length(nums))

    for(x in 1:length(nums)){
        set.seed(1)
        bm <- ada(formula=y ~ ., data=train.data, 
                  iter=ntree, nu=nums[x], bag.frac=0.5,
                  control=rpart.control(maxdepth=30,cp=cp, 
                              minsplit=minsplit, xval=10))

        trainpred <- predict(bm, train.data)
        cmat <- confusionMatrix(trainpred, train.data$y)
        accur[x]<-cmat$overall[1]
}

  bestnu<-nums[which.max(accur)]

#############################################################
############# REBUILD TREE USING PARAMETERS #################
#############################################################
  
adaModel <- ada(formula=y ~ ., data=train.data, iter=ntree, 
            nu=bestnu, bag.frac=0.5,control=rpart.control(
            maxdepth=30, cp=cp, minsplit=minsplit, xval=10))

########################################################
############## Training Set Prediction #################
########################################################

ada.train <- predict(adaModel, train.data)
confusionMatrix(ada.train, train.data$y)
  
##########################################################
################# Test Set Prediction ####################
##########################################################
  
  ada.test <- predict(adaModel, newdata=test.data)
  conf.table <- table(test.data$y, testpred, 
                      dnn=c("Actual", "Predicted"))

  Models[10,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
  Models[10,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
  Models[10,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100

  confusionMatrix(testpred, test.data$y)
  
#################################################################
#################################################################
################## MODEL 11: Gbm Boosted Tree ###################
#################################################################
#################################################################
  
  
  ######################################################
  ################ Training the Model ##################
  ###################################################### 
  
  newtrain<-train.data
  newtest<-test.data
  newtrain$y<-as.numeric(as.character(newtrain$y))
  newtest$y<-as.numeric(as.character(newtest$y))

  ######################################
  #### Find Optimal Shrinkage value ####
  ######################################
  
  shrinkage <- seq(.01,1,.01)
  acc <- rep(0,length(shrinkage))
  bestbound <- rep(0,length(shrinkage))
  boundary <- seq(.1, .999, .005)

  for(x in 1:length(shrinkage)){
      set.seed(1)
      gbmmodel<-gbm(y~.,distribution = 'bernoulli', 
              data=newtrain, n.trees=ntree, 
              shrinkage=shrinkage[x], bag.fraction = 0.5)
      
      gbpred <- predict(gbmmodel,newdata=newtrain,
                      n.trees=ntree,type='response')
      acc1<-rep(0,length(boundary))
      for(i in 1:length(boundary)){
        gbpred2<-gbpred
        gbpred2 <- ifelse(gbpred2 < boundary[i], 0, 1)
        cmat<-confusionMatrix(gbpred2, newtrain$y)
        acc1[i]<-cmat$overall[1]
      }
    bestbound[x] <- boundary[which.max(acc1)]
    acc[x]<-max(acc1)
  }
  
  bestshrink <- shrinkage[which.max(acc)]
  bestboundary <- bestbound[which.max(acc)]

###############################################
####### Plot Shrinkage vs. Accuracy ###########
###############################################
  
  plot(shrinkage, acc)
  
  ### Choose general area with less dispersion...
  bestshrink <- .55
  bestboundary <- bestbound[55]
    
##########################################################
########## REBUILD TREE USING PARAMETERS #################
##########################################################

  set.seed(1)
  gbmmodel<-gbm(y~.,distribution = 'bernoulli', 
              data=newtrain, n.trees=ntree, 
              shrinkage=.55, bag.fraction = 0.5)
  
#########################################################
############## Training Set Prediction ##################
#########################################################
  
  gbm.train <- predict(gbmmodel, newtrain, n.trees=500, 
                       type = 'response')
  
  gbpred <- gbm.train
  gbpred <- ifelse(gbpred < bestboundary, 0, 1)
  confusionMatrix(gbpred, newtrain$y)

##########################################################
################# Test Set Prediction ####################
##########################################################

  gbtest <- predict(gbmmodel, newtest, n.trees=500, 
                    type='response')
  gbtest2 <- gbtest
  gbtest2 <- ifelse(gbtest2 < bestboundary, 0, 1)
  conf.table <- table(newtest$y, gbtest2, 
                      dnn=c("Actual", "Predicted"))
  
  Models[11,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
  Models[11,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
  Models[11,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100
  
  confusionMatrix(gbtest2, newtest$y)

#################################################################
#################################################################
################## MODEL 12: Xg Boosted Tree ####################
#################################################################
#################################################################

  
  ######################################################
  ################ Training the Model ##################
  ###################################################### 
  
  set.seed(1)
  xgdata<-newtrain
  xgdata1<-newtest

  xg<-xgboost(params = list(objective='binary:logistic',max_depth=30),
            data=data.matrix(xgdata[1:57]),label = xgdata$y,
            nrounds=100,nfold=10)
  
  #########################################################
  ############## Training Set Prediction ##################
  #########################################################
  
  xgtrain <- predict(xg, newdata=data.matrix(xgdata),
                     type='response')
  
  nums<-seq(.01,.5,.005)
  acc<-rep(0,length(nums))
  for(i in 1:length(nums)){
    xgtrain2<-xgtrain
    xgtrain2<- ifelse(xgtrain2 < nums[i], 0, 1)
    conf.table <- table(newtrain$y, xgtrain2, dnn=c("Actual", "Predicted"))
    acc[i] <- (conf.table[1] + conf.table[4])/sum(conf.table)
  }
  
  decisionbound<-nums[which.max(acc2)]
  xgtrain<-ifelse(xgtrain < decisionbound, 0, 1)
  confusionMatrix(xgtrain, newtrain$y)
  
  ##########################################################
  ################# Test Set Prediction ####################
  ##########################################################

  xgtest <- predict(xg, newdata=data.matrix(xgdata1),
                    type='response')
  xgtest<-ifelse(xgtest < decisionbound, 0, 1)
  conf.table <- table(xgdata1$y, xgtest, 
                      dnn=c("Actual", "Predicted"))

  Models[12,1] <- ((conf.table[1] + conf.table[4])/sum(conf.table)) * 100
  Models[12,2] <- (conf.table[1,2]/sum(conf.table[1,])) * 100
  Models[12,3] <- (conf.table[2,1]/sum(conf.table[2,])) * 100
  
  confusionMatrix(xgtest, newtest$y)
  
###############################################################################
############# Model Summary - Accuracy, Type 1 & Type 2 Errors ################
###############################################################################

print(Models)
  
  ### Output ###
  
  # Models         Accuracy    Type1     Type2
  
  #NaiveBayes      86.50707 15.260323 10.773481
  #GLM             92.38303  5.026930 11.602210
  #GAM             92.05658  5.026930 12.430939
  #SVMLinear       92.27421  4.667864 12.430939
  #SVMRadial       93.03591  4.847397 10.220994
  #SVMPolynomial   93.25354  4.129264 10.773481
  #TreeInformation 92.27421  6.463196  9.668508
  #TreeGini        92.05658  5.745063 11.325967
  #RandomForest    95.10337  2.692998  8.287293
  #AdaBoost        95.64744  3.411131  5.801105
  #GbmBoost        94.12405  4.308797  8.287293
  #XgBoost         95.53863  4.308797  4.696133
  
 