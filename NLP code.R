#Step 1:  import libraries and csv file
# activate libraries
library (readr) 
library (tidyverse)  
library (party)
library (Hmisc)
library (devtools)
library (dplyr)

news <- read.csv (file.choose (), header = T, sep = ";")
as_tibble (news)
view(news)

#Step 2: import libraries for data cleaning and exploration
library (stringr) # for data manipulation    
library (ggplot2) # for data visualization  
library (GGally)   #for data visulaitsation
library (tm) # for text mining
library (SnowballC) # for text stemming
library (wordcloud) # word-cloud generator 
library (RColorBrewer)  # color palettes
library (knitr)

#split the variables into categorical variables and numeric variables
cat_var <- names(news)[which (sapply (news, is.character))]
str (news)
summary (news)

#Step 3: data exploration for text using wordcloud and barplot
#convert data frame into corpus
df_title <- data.frame (doc_id=row.names(news),
                        text=news$Text)
docs <- Corpus(DataframeSource(df_title))
tdm <- TermDocumentMatrix(docs)

#Text transformation and cleaning
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, "", x))
docs <- tm_map (docs, toSpace, "/")
docs <- tm_map (docs, toSpace, "@")
docs <- tm_map (docs, toSpace, "\\|")
docs <- tm_map (docs, toSpace, "\\$")
docs <- tm_map (docs, toSpace, "?")
docs <- tm_map (docs, toSpace, "!")

# Convert the text to lower case
docs <- tm_map (docs, content_transformer(tolower))

# Remove numbers
docs <- tm_map (docs, removeNumbers)

# Remove English common stop words
docs <- tm_map (docs, removeWords, stopwords("english"))

# Remove punctuation
docs <- tm_map (docs, removePunctuation)

# Eliminate extra white spaces
docs <- tm_map (docs, stripWhitespace)

# Text stemming
docs <- tm_map(docs, stemDocument)


dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)

#Visualizing with Word cloud
set.seed(1234)
wordcloud (words = d$word, freq = d$freq, min.freq = 1,
           max.words=200, random.order=FALSE, rot.per=0.35, 
           colors=brewer.pal(8, "Dark2"))
findFreqTerms(dtm, lowfreq = 100)

#Calculating the frequency of the words
barplot (d[1:10,]$freq, las = 2, names.arg = d[1:10,]$word,
         col ="blue", main ="Most frequent words",
         ylab = "Word frequencies")

# function to set plot height and width
fig <- function (width, heigth){
  options (repr.plot.width = width, repr.plot.height = heigth)
}

#Step 4: data exploration for tags using wordcloud and barplot
#convert data frame into corpus
df_title <- data.frame (doc_id=row.names(news),
                        text=news$Text_Tag)
tags <- Corpus(DataframeSource(df_title))
inspect (tags)
tdm_tag <- TermDocumentMatrix(tags)

#Text transformation
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
tags <- tm_map (tags, toSpace, ",")

dtm_tag <- TermDocumentMatrix(tags)
m <- as.matrix(dtm_tag)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)

head(d, 10)

set.seed(1024)

wordcloud (words = d$word, freq = d$freq, min.freq = 1,
           max.words=200, random.order=FALSE, rot.per=0.35, 
           colors=brewer.pal(8, "Dark2"))

findFreqTerms(dtm_tag, lowfreq = 100)

barplot (d[1:30,]$freq, las = 2, names.arg = d[1:30,]$word,
         col ="blue", main ="Most frequent tags",
         ylab = "tag frequencies")

# function to set plot height and width
fig <- function (width, heigth){
  options (repr.plot.width = width, repr.plot.height = heigth)
}

#Step 5: Let’s see if the dataset is well balanced.
fig (12, 8)
common_theme <- theme (plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data = news, aes(x = factor(Label), 
                        y = prop.table(stat(count)), fill = factor (Label),
                        label = scales::percent (prop.table (stat(count))))) +
  geom_bar (position = "dodge") + 
  geom_text (stat = 'count',
             position = position_dodge(.9), 
             vjust = -0.5, 
             size = 3) + 
  scale_x_discrete (labels = c("FAKE", "REAL"))+
  scale_y_continuous (labels = scales::percent)+
  labs (x = 'Label', y = 'Percentage') +
  ggtitle ("Distribution of Labels") +
  common_theme

sum (news$Label == 'REAL')
sum (news$Label == 'FAKE')
#the results indicate that the data is imbalanced

#Step 6: model building preparation
docs_mat <- t(as.matrix(dtm))
dim(docs_mat)

tags_mat <- t(as.matrix(dtm_tag))
dim(tags_mat)

data <- cbind(docs_mat,tags_mat)
data <- cbind(news[,3],data)
dim(data)

#export combine dataset
write.csv(data,'final_data.csv')

#Step 7: building model using machine learning packages
library (moments)
library (gridExtra)
library (caret) # for sampling     M3
library (caTools) # for train/test split     M3
library (gridExtra)  #M3
library (pscl)   #M3
library (MLmetrics)
library (Label)
library (h2o)
library (testthat)
library (Rtsne) # for tsne plotting
library (xgboost) # for xgboost model

library (blorr)
library (magrittr)

library (ROCR)
library (pROC)

data <- data.frame(data)
# make Label a factor
data$V1 <- factor (data$V1)

str(data)  ## check data type 
data <- as.data.frame(lapply(data,as.numeric))
str(data)  ## check data type

# make Label a factor
data$V1 <- factor (data$V1)
str(data)  ## check data type            
              
#splitting data into test and train
intrain = createDataPartition (y = data$V1, p = .75, list = FALSE) 
train = data [intrain,]
test = data [-intrain,]
dim (train)

# write.csv(train,'train.csv')
# write.csv(test,'test.csv')
library (ROSE) 
library (rpart) 
library (Rborist) 
library (e1071) 

#Step 8: balance data using downsampling and upsampling
# downsampling
set.seed(9560)
down_train <- downSample(x = train[,-1],
                         y = train$V1)
table(down_train$Class)

# upsampling
set.seed(9560)
up_train <- upSample(x = train[, -1],
                     y = train$V1)
table(up_train$V1)

#Step 9: Model training and validation in different data balancing methods
################downsample###########################
#xgboost
# Convert Label 
labels <- down_train$Class

y <- recode(labels, '1' = 0, "2" = 1)
set.seed(42)
xgb <- xgboost(data = data.matrix(down_train[,-ncol(down_train)]), 
               label = y,
               eta = 0.1,
               gamma = 0.1,
               max_depth = 10, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.6,
               verbose = 0,
               nthread = 7,
)
xgb_pred <- predict(xgb, data.matrix(test[,-1]))

labels <- test$V1
labels <- recode(labels, '1' = 0, "2" = 1)
test$V1 <- labels
roc.curve(test$V1, xgb_pred, plotit = TRUE)
xgb_pred <- ifelse(xgb_pred > 0.5, 1, 0)
table (test$V1, xgb_pred)   # confusionMatrix 

important_variables = xgb.importance(model = xgb, feature_names = colnames(down_train[,-ncol(down_train)]))
important_variables
dim(important_variables)


################upsample###########################
#xgboost
# Convert Label labels from factor to numeric

labels <- up_train$Class

y <- recode(labels, '1' = 0, "2" = 1)
set.seed(42)
xgb.up <- xgboost(data = data.matrix(up_train[,-ncol(up_train)]), 
               label = y,
               eta = 0.1,
               gamma = 0.1,
               max_depth = 10, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.6,
               verbose = 0,
               nthread = 7,
)
xgb_pred.up <- predict(xgb.up, data.matrix(test[,-1]))

labels <- test$V1
labels <- recode(labels, '1' = 0, "2" = 1)
test$V1 <- labels
roc.curve(test$V1, xgb_pred.up, plotit = TRUE)
xgb_pred.up <- ifelse(xgb_pred.up > 0.5, 1, 0)
table (test$V1, xgb_pred.up)   # confusionMatrix 

important_variables.up = xgb.importance(model = xgb, feature_names = colnames(down_train[,-ncol(down_train)]))
important_variables.up
dim(important_variables.up)


################ NO sample###########################
#xgboost
# Convert Label labels from factor to numeric

labels <- train$V1

y <- recode(labels, '1' = 0, "2" = 1)
set.seed(42)
xgb.train <- xgboost(data = data.matrix(train[,-1]), 
                  label = y,
                  eta = 0.1,
                  gamma = 0.1,
                  max_depth = 10, 
                  nrounds = 300, 
                  objective = "binary:logistic",
                  colsample_bytree = 0.6,
                  verbose = 0,
                  nthread = 7,
)
xgb_pred.train <- predict(xgb.train, data.matrix(test[,-1]))

labels <- test$V1
labels <- recode(labels, '1' = 0,'2' = 1)
test$V1 <- labels

# roc.curve(test$V1, xgb_pred.train, plotit = TRUE)
xgb_pred.train <- ifelse(xgb_pred.train > 0.5, 1, 0)
table (test$V1, xgb_pred.train)   # confusionMatrix 

important_variables.train = xgb.importance(model = xgb, feature_names = colnames(down_train[,-ncol(down_train)]))
important_variables.train
dim(important_variables.train)

pred <- prediction(xgb_pred.train,test$V1) 
perf <- performance(pred,"tpr","fpr")
auc <- performance(pred,'auc')
auc = unlist(slot(auc,"y.values"))
plot(perf,
     xlim=c(0,1), ylim=c(0,1),col='red', 
     main=paste("ROC curve (", "AUC = ",auc,")"),
     lwd = 2, cex.main=1.3, cex.lab=1.2, cex.axis=1.2, font=1.2)
abline(0,1)

#Step 10: training error
xgb_pred.train.error <- predict(xgb.train, data.matrix(train[,-1]))

labels <- train$V1
labels <- recode(labels, '1' = 0,'2' = 1)
train$V1 <- labels

pred <- prediction(xgb_pred.train.error,train$V1) 
perf <- performance(pred,"tpr","fpr")
auc <- performance(pred,'auc')
auc = unlist(slot(auc,"y.values"))
plot(perf,
     xlim=c(0,1), ylim=c(0,1),col='red', 
     main=paste("ROC curve (", "AUC = ",auc,")"),
     lwd = 2, cex.main=1.3, cex.lab=1.2, cex.axis=1.2, font=1.2)
abline(0,1)
