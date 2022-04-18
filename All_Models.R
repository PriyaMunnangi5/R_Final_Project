#install.packages("corrplot")
#install.packages("ggfortify")
#install.packages("doParallel")
#install.packages("registerdoParallel")
#install.packages("ggcorrplot")
library("ggplot2")
library("e1071")
library(dplyr)
library(reshape2)
library(corrplot)
library(caret)
library(pROC)
library(gridExtra)
library(grid)
library(ggfortify)
library(purrr)
library(doParallel) 
registerDoParallel()
require(foreach)
require(iterators)
require(parallel)
library("corrplot")
library("ggfortify")
library("doParallel")
# library(regis)
library(ggcorrplot)
library(rpart)
library(rpart.plot)
library(mlbench)
library(randomForest)
library(stringr)



#Reading file
BD_raw <- read.csv("~/Desktop/R/Breast_Data.csv", sep=",")


#Descriptive statistics
str(BD_raw)

colnames(BD_raw[])


#ID column is not significant. So, removing it:
BD_clean <- BD_raw[,-c(0:1)]

#Removing the last column because it is an empty column:
BD_clean <- BD_clean[,-32]

str(BD_clean)
#Factoring the M/B column as category:
BD_clean$diagnosis <- as.factor(BD_clean$diagnosis)

head(BD_clean)


#checking for missing values:
sapply(BD_clean, function(x) sum(is.na(x)))
#there were no missing values


#summary of data:
summary(BD_clean)

#Diagnosis is categorical.
#All feature values are recoded with four significant digits.
#Missing attribute values: none
#Class distribution: 357 benign, 212 malignant



#univariate plots:

## Create a frequency table
freq.tablefordiag <- table(BD_clean$diagnosis)
mygraphcolors <- terrain.colors(2) 
# Create a pie chart 
prob.tablefordiag <- prop.table(freq.tablefordiag)*100
prop.dataframe.on.diagnosis <- as.data.frame(prob.tablefordiag)
labels.forpiechart <- sprintf("%s - %3.1f%s", prop.dataframe.on.diagnosis[,1], prob.tablefordiag, "%")

pie(prob.tablefordiag,
    labels=labels.forpiechart,  
    clockwise=TRUE,
    col=c("lightblue","dodgerblue"),
    border="gainsboro",
    radius=0.8,
    cex=0.8, 
    main="Distribution of cancer diagnosis")
legend(1, .4, legend=prop.dataframe.on.diagnosis[,1], cex = 0.7, fill = c("lightblue","dodgerblue") )


#M= Malignant (indicates prescence of cancer cells); B= Benign (indicates abscence)
#357 observations which account for 62.7% of all observations indicating the absence of cancer cells, 212 which account for 37.3% of all observations shows the presence of cancerous cell.




#Break up columns into groups, according to their suffix designation 
#(_mean, _se,and __worst) to perform visualisation plots off.
mean.of.BD_raw <- BD_raw[ ,c("diagnosis", "radius_mean", "texture_mean","perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave.points_mean", "symmetry_mean", "fractal_dimension_mean" )]

se.of.BD_raw <- BD_raw[ ,c("diagnosis", "radius_se", "texture_se","perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave.points_se", "symmetry_se", "fractal_dimension_se" )]

worst.of.BD_raw <- BD_raw[ ,c("diagnosis", "radius_worst", "texture_worst","perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave.points_worst", "symmetry_worst", "fractal_dimension_worst" )]

#Plot histograms of "_mean" variables group by diagnosis
ggplot(data = melt(mean.of.BD_raw, id.var = "diagnosis"),
       mapping = aes(x = value)) +
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) +
  scale_fill_manual(values = c("red", "blue"))+
  facet_wrap(~variable, scales ='free_x')+
   labs(title = "Distribution of Mean variables")
  


#Plot histograms of "_se" variables group by diagnosis
ggplot(data = melt(se.of.BD_raw, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + 
  facet_wrap(~variable, scales = 'free_x')+
    labs(title = "Distribution of Standard Error variables")+
  scale_fill_manual(values = c("red", "blue"))

#Plot histograms of "_worst" variables group by diagnosis
ggplot(data = melt(worst.of.BD_raw, id.var = "diagnosis"), mapping = aes(x = value)) + 
  geom_histogram(bins = 10, aes(fill=diagnosis), alpha=0.5) + facet_wrap(~variable, scales = 'free_x')+
labs(title = "Distribution of Worst measurd variables")+
  scale_fill_manual(values = c("red", "blue"))

#Most of the features are normally distributed.
#Comparison of radius distribution by malignancy shows that there is 
#no perfect separation between any of the features; we do have fairly good 
#separations for concave.points_worst, concavity_worst, perimeter_worst, area_mean, 
#perimeter_mean. We do have as well tight superposition for some of the values, 
#like symmetry_se, smoothness_se .

dev.off()


#bivariate

#collinearity
collinearityofBD_clean <- cor(BD_clean[,2:31])
corrplot(collinearityofBD_clean, order = "hclust", tl.cex = 0.7,
         main = "Correlation Coefficent",
         mar=c(0,0,1,0))


hi.cor.var.of.BD_clean <- colnames(BD_clean)[findCorrelation(collinearityofBD_clean, cutoff = 0.9, verbose = TRUE)]
#these variables are highly correlated:
hi.cor.var.of.BD_clean

#ggcorrplot(collinearityofBD_clean,type = "upper",lab = TRUE)

bc_data_cor <- BD_clean[, which(!colnames(BD_clean) %in% hi.cor.var.of.BD_clean)]

#Data Cleaning
cancer.pca <- prcomp(BD_clean[, 2:31], center=TRUE, scale=TRUE)
plot(cancer.pca, type="l", main='')
grid(nx = 10, ny = 14)
title(main = "Principal components weight", sub = NULL, xlab = "Components")
box()

summary(cancer.pca)

pca_var <- cancer.pca$sdev^2
pve_df <- pca_var / sum(pca_var)
cum_pve <- cumsum(pve_df)
pve_table <- tibble(comp = seq(1:ncol(BD_clean %>% select(-diagnosis))), pve_df, cum_pve)

ggplot(pve_table, aes(x = comp, y = cum_pve)) + 
  geom_point() + 
  labs(x = "Component",y = 'Cummulative Variation',
       title = 'Component Analysis for all variables')+
  geom_abline(intercept = 0.95, color = "red", slope = 0)

##PCA including highly correlated values
cancer.pca2 <- prcomp(bc_data_cor, center=TRUE, scale=TRUE)
summary(cancer.pca2)


pca_var2 <- cancer.pca2$sdev^2
pve_df2 <- pca_var2 / sum(pca_var2)
cum_pve2 <- cumsum(pve_df2)
pve_table2 <- tibble(comp = seq(1:ncol(bc_data_cor)), pve_df2, cum_pve2)

ggplot(pve_table2, aes(x = comp, y = cum_pve2)) + 
  geom_point() + 
  labs(x = "Component",y = 'Cummulative Variation',
       title = 'Component Analysis for Principal Components')+
  geom_abline(intercept = 0.95, color = "red", slope = 0)


pca_df <- as.data.frame(cancer.pca2$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=BD_clean$diagnosis)) + geom_point(alpha=0.5)

autoplot(cancer.pca2, data = BD_clean,  colour = 'diagnosis',
         loadings = FALSE, loadings.label = TRUE, loadings.colour = "blue")


df_pcs <- cbind(as_tibble(BD_clean$diagnosis), as_tibble(cancer.pca2$x))

GGally::ggpairs(df_pcs, columns = 2:4, ggplot2::aes(color = value))


#Split data set in train 70% and test 30%
set.seed(1234)
df <- cbind(diagnosis = BD_clean$diagnosis, bc_data_cor)
train_indx <- createDataPartition(df$diagnosis, p = 0.7, list = FALSE)

train_set <- df[train_indx,]
test_set <- df[-train_indx,]

nrow(train_set)

nrow(test_set)

fitControl <- trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)
#decision tree

model_dt <-rpart(diagnosis~., data=train_set, method='class')

summary(model_dt)

rpart.plot(model_dt)

pred_dt <- predict(object = model_dt,  
                            newdata = test_set,   
                            type = "class")

cm_dt <- confusionMatrix(data = pred_dt,reference = test_set$diagnosis)
cm_dt

pred_dt_prob <- predict(object = model_dt,  
                   newdata = test_set,   
                   type = "prob")

library(pROC)
auc <- auc(test_set$diagnosis,pred_dt_prob[,2])
plot(roc(test_set$diagnosis,pred_dt_prob[,2]),
     print.auc = TRUE,
     main = "ROC Measure for Decision Tree")

#random forest
model_rf <- train(diagnosis~.,
                  data = train_set,
                  method="rf",
                  metric="ROC",
                  #tuneLength=10,
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)

model_rf


# plot feature importance
plot(varImp(model_rf), top = 10, main = "Random forest")

#confusion matrix
pred_rf <- predict(model_rf, test_set)
cm_rf <- confusionMatrix(pred_rf, test_set$diagnosis, positive = "M")
cm_rf

#Random Forest with PCA
model_pca_rf <- train(diagnosis~.,
                      data = train_set,
                      method="ranger",
                      metric="ROC",
                      #tuneLength=10,
                      preProcess = c('center', 'scale', 'pca'),
                      trControl=fitControl)

model_pca_rf
pred_pca_rf <- predict(model_pca_rf, test_set)

#confusion matrix for PCA
cm_pca_rf <- confusionMatrix(pred_pca_rf, test_set$diagnosis, positive = "M")
cm_pca_rf

#GBM

model_gbm <- train(diagnosis~.,
                      data = train_set,
                      method="gbm",
                      metric="ROC",
                      #tuneLength=10,
                      preProcess = c('center', 'scale'),
                      trControl=fitControl)
model_gbm
pred_gbm <- predict(model_gbm, test_set)
cm_gbm <- confusionMatrix(pred_gbm, test_set$diagnosis, positive = "M")
cm_gbm


#model evaluation

model_list <- list(RF=model_rf, PCA_RF=model_pca_rf, 
                   GBM = model_gbm)
resamples <- resamples(model_list)


bwplot(resamples, metric = "ROC",main = "ROC Measure")

cm_list <- list(RF=cm_rf, PCA_RF=cm_pca_rf,GBM = cm_gbm,DT=cm_dt)

cm_list
results <- sapply(cm_list, function(x) x$byClass)

results

results_max <- apply(results, 1, which.max)
output_report <- data.frame(metric=names(results_max), 
                            best_model=colnames(results)[results_max],
                            value=mapply(function(x,y) {results[x,y]}, 
                                         names(results_max), 
                                         results_max))
rownames(output_report) <- NULL
output_report

