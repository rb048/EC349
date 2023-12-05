library(jsonlite)
library(tidyverse)
library(ggplot2)
library(tidytext)
library(caret)
library(topicmodels)

cat("\014")
rm(list=ls())
gc()

setwd("C:/Users/User/Desktop/EC349/Workings")

#review_data  <- stream_in(file("yelp_academic_dataset_review.json"))
#user_data <- stream_in(file("yelp_academic_dataset_user.json"))
business_data <- stream_in(file("yelp_academic_dataset_business.json"))
#checkin_data  <- stream_in(file("yelp_academic_dataset_checkin.json"))
#tip_data  <- stream_in(file("yelp_academic_dataset_tip.json"))

load("yelp_review_small.Rda")
load("yelp_user_small.Rda")


business_data <- as_tibble(business_data)
review_data_small <- as_tibble(review_data_small)
user_data_small <- as_tibble(user_data_small)

user_data_small
business_data
review_data_small

user_summary <- summary(user_data_small)           
business_summary <- summary(business_data)
review_summary <- summary(review_data_small)
#summaries to identify redundant variables

user_summary
business_summary
review_summary

ggplot(review_data_small, aes(x = stars)) + 
  geom_histogram(bins = 30, col= "white")
#observe frequency distribution of stars given to each review

ggplot(user_data_small, aes(x=average_stars)) +
  geom_density(kernel="gaussian")
#observe frequency distributions of average stars given by each user

ggplot(business_data, aes(x=stars, y=review_count)) + 
  geom_point()
#number of reviews received based on how many stars they have

length(unique(business_data$state)) 
#no. of states across the datasets

ggplot(business_data, aes(x = state)) + 
  geom_bar() + scale_y_log10()
#no. of businesses in each state

ggplot(business_data, aes(x=state,y=stars)) +
  geom_col()
#total number of stars achieved by each state
  
avg_stars_by_state <- business_data %>%
  group_by(state) %>%
  summarize(avg_stars=mean(stars))
ggplot(avg_stars_by_state, aes(x=state,y=avg_stars)) +
  geom_col()
#average number of stars achieved by each state

business_status_avg <- business_data %>%
  group_by(is_open) %>%
  summarize(avg_stars=mean(stars))
print(business_status_avg)
#average stars based on operating status of business

#ggplot(user_data, aes(x=yelping_since, y=average_stars))+      #did not include this graph since it was too computationally expensive
#  geom_line()

user_data_small <- user_data_small %>%
  mutate(num_friends = ifelse(friends == "None", 0, 1+str_count(friends, ",")))

ggplot(user_data_small, aes(x=num_friends)) + 
  geom_density(kernel="gaussian")

ggplot(user_data_small, aes(x=average_stars, y=1+num_friends)) +
  geom_point() + scale_y_log10()

summarise(user_data_small, sum(num_friends))

review_words <- review_data_small %>%
  mutate(text = tolower(text)) %>%
  mutate(text = str_replace_all(text, "[[:punct:]]", " ")) %>%
  mutate(text = str_replace_all(text, "[[:digit:]]", " ")) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by="word")
#breaks down reviews into their individual words, removes uppercase, punctuation, numbers, and common stop words

word_counts <- review_words %>%
  count(stars, word, sort = TRUE)
#counts the occurences of each word by the star rating of the review for use in tf_idf analysis

word_counts <- word_counts %>%
  filter(n >= 20) %>%
  ungroup()
#removes words occuring under 20 times

reviews_tfidf <- word_counts %>%
  bind_tf_idf(word, stars,n)
nrow(filter(reviews_tfidf,tf_idf>0))
#tf_idf statistic computed, and number of words with a score>0 printed for reference

reviews_tfidf %>%
  group_by(stars) %>%
  slice_max(tf_idf, n = 15) %>%
  ungroup() %>%
  ggplot(aes(tf_idf, fct_reorder(word, tf_idf), fill = stars)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~stars, ncol = 2, scales = "free") +
  labs(x = "tf-idf", y = NULL)
#visualisation of 15 most impactful words on each star rating

reviews_tfidf <- reviews_tfidf %>%
  mutate(norm_relevance=(tf_idf-min(tf_idf))/max(tf_idf)-min(tf_idf))
nrow(filter(reviews_tfidf,tf_idf>0))
#min-max normalisation of the tf_idf score

word_scores <- reviews_tfidf %>%
  group_by(word) %>%
  summarise(score=sum(norm_relevance*stars*n)/sum(n))
word_scores
#weighted average 'sentiment' score created

review_words <- review_words %>%
  left_join(word_scores, by="word")


review_words <- review_words %>%
  group_by(review_id) %>%
  summarise(total_score=sum(score, na.rm=TRUE))
#calculates a score for each review based on the number of hits with impactful words and their scores

nrow(filter(review_words,total_score>0))

business_vars <- select(business_data, business_id,stars, is_open, state, review_count)
business_vars <- rename(business_vars, business_review_count = "review_count")
user_vars <- select(user_data_small, user_id, average_stars,review_count)
review_vars <- select(review_data_small, review_id, user_id, business_id)
final_data <- left_join(review_words,review_vars, by="review_id")
final_data <- left_join(final_data,business_vars, by="business_id")
final_data <- left_join(final_data,user_vars, by = "user_id")
final_data <- rename(final_data, business_stars = "stars", user_stars = "average_stars")
review_stars <- select(review_data_small,review_id,stars)
final_data <- left_join(final_data,review_stars, by = "review_id")
final_data <- select(final_data, stars, user_stars, business_stars, total_score, review_count, is_open, state, business_review_count)
#preparation of the final dataset

final_data
sum(is.na(final_data))
final_data <- na.omit(final_data)
final_data$stars<-factor(final_data$stars)
final_data$is_open<-factor(final_data$is_open)
final_data$state<-factor(final_data$state)
final_data$total_score<-as.numeric(final_data$total_score)
#remove blanks for compatibility with random forest
rm(review_stars,review_vars,review_words,reviews_tfidf,user_vars,word_counts,word_scores)
#clear memory

set.seed(1)
test_rows <- sample(1:nrow(final_data), 10000)
train <- final_data[-test_rows, ]
test <- final_data[test_rows, ]
#split data into testing and training

library(randomForest)
gc()
train
model <- randomForest(stars~., data=train, ntree=50)
predictions <- predict(model, test)
mean(model[["err.rate"]])

confusionMatrix(predictions, test$stars)

