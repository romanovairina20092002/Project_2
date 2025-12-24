#load libraries 
library(tm)
library(textstem)
library(cld3)
library (tidyverse)
library(syuzhet)
library (tidytext)
library (RColorBrewer)
library (tokenizers)
library(wordcloud)
library (dplyr)
library(ldatuning)
library(topicmodels)
library(LDAvis)
# Understanding data
hotels <-read.csv("HotelsData.csv")
str(hotels)
head(hotels)
sum(!complete.cases(hotels))
#% of English reviews
hotels$language <- cld3::detect_language(hotels$Text.1 )
total_texts <- nrow(hotels)
english_texts <- hotels %>% filter(language == "en") %>% nrow()
english_percentage <- (english_texts / total_texts) * 100
print (english_percentage) 
#80% of reviews are written in English, so the sample will consists only reviews in English 

# Creating a random sample of English reviews
set.seed(689) 
text <- hotels %>% filter(language == "en") %>% sample_n(2000)

#Split positive and negative reviews based on Review.score
text <- text %>% 
  mutate (sentiment = case_when(
    Review.score >=4 ~ "positive",
    Review.score <=2 ~ "negative", 
    TRUE ~ NA_character_
  )
  ) %>%
  filter(!is.na(sentiment)) #Neutral reviews are excluded
#Creating positive and negative data frames
pos_reviews <- text %>% filter(sentiment == "positive")
neg_reviews <- text %>% filter(sentiment == "negative")
#Convert review column of dataframe to character vector
corpus_pos <- iconv(pos_reviews$Text.1, from = '', to = 'UTF-8')
corpus_neg <- iconv(neg_reviews$Text.1, from = '', to = 'UTF-8')
#Corpus of Text
docs_pos <- VCorpus(VectorSource(corpus_pos))
print(docs_pos[[1]]$content)
docs_neg <- VCorpus(VectorSource(corpus_neg))
print(docs_neg[[1]]$content)

# Clean corpus
clean_corpus <- function(corp) {
  corp %>%
    tm_map(content_transformer(tolower)) %>%       # Сonvert to lower case
    tm_map(removePunctuation) %>%                  # Remove punctuation 
    tm_map(removeNumbers) %>%                      # Remove numbers
    tm_map(removeWords, stopwords("english")) %>%  # Remove stopwords
    tm_map(stripWhitespace) %>%                    # Remove extra spaces
    tm_map(content_transformer(lemmatize_strings)) #Lemmatization
}
docs_pos <- clean_corpus(docs_pos)
print(docs_pos[[1]]$content)
docs_neg <- clean_corpus(docs_neg)
print(docs_neg[[1]]$content)

#Document Term Matrix (Positive reviews)
dtm_pos <- DocumentTermMatrix(docs_pos,
                              control = list(wordLengths=c(3,Inf)))
# Drop terms which occur in less than 1 percent of the documents
dtms_pos <- removeSparseTerms(dtm_pos, 0.99)
#Removes documents with zero word counts (i.e., empty documents)
nonempty    <- rowSums(as.matrix(dtms_pos)) > 0
dtms_pos   <- dtms_pos[nonempty, ]
#TF_IDF
dtm_tfidf_pos <- weightTfIdf(dtms_pos)
tfidf_matrix_pos <- as.matrix(dtm_tfidf_pos)
tfidf_scores_pos <- colMeans(tfidf_matrix_pos)
tfidf_threshold_pos <- quantile(tfidf_scores_pos, 0.75)  # keep top 25% informative terms
selected_terms_pos <- names(tfidf_scores_pos[tfidf_scores_pos >= tfidf_threshold_pos])

# Filter DTM to include only selected TF-IDF terms
dtm_lda_pos <- dtms_pos[, selected_terms_pos]
dtm_lda_pos <- dtm_lda_pos[rowSums(as.matrix(dtm_lda_pos)) > 0, ]

# Top TF-IDF terms
tfidf_terms_pos <- sort(tfidf_scores_pos, decreasing = TRUE) [1:50]
print(tfidf_terms_pos)

#Overall emotional analysis
sentiment_scores <- get_nrc_sentiment(text$Text.1)
head(sentiment_scores)
sentiment_summary <- colSums(sentiment_scores)
sentiment_frame <- data.frame(
  Emotion = names (sentiment_summary), 
  Score =sentiment_summary
)
ggplot(sentiment_frame, aes(x = reorder(Emotion, -Score), y = Score)) +
  geom_bar(stat = "identity") +
  labs(title = "Overall Emotional Tone in Reviews", x = "Emotion", y = "Score")
#World cloud 
wordcloud(names(tfidf_terms_pos), tfidf_terms_pos, max.words = 50,rot.per=0.15, 
          random.order = FALSE, scale=c(2,0.9),
          random.color = FALSE, colors=brewer.pal(8,"Dark2"))
title("Positive Reviews")

#Document Term Matrix (negative_reviews)
dtm_neg <- DocumentTermMatrix(docs_neg,
                              control = list(wordLengths=c(3,Inf)))
# Drop terms which occur in less than 1 percent of the documents
dtms_neg <- removeSparseTerms(dtm_neg, 0.99)
#Removes documents with zero word counts (i.e., empty documents)
nonempty_neg    <- rowSums(as.matrix(dtms_neg)) > 0
dtms_neg   <- dtms_neg[nonempty_neg, ]
#TF_IDF
dtm_tfidf_neg <- weightTfIdf(dtms_neg)
tfidf_matrix_neg <- as.matrix(dtm_tfidf_neg)
tfidf_scores_neg <- colMeans(tfidf_matrix_neg)
tfidf_threshold_neg <- quantile(tfidf_scores_neg, 0.75)  # keep top 25% informative terms
selected_terms_neg <- names(tfidf_scores_neg[tfidf_scores_neg >= tfidf_threshold_neg])

# Filter DTM to include only selected TF-IDF terms
dtm_lda_neg <- dtms_neg[, selected_terms_neg]
dtm_lda_neg <- dtm_lda_neg[rowSums(as.matrix(dtm_lda_neg)) > 0, ]

# Top TF-IDF terms
tfidf_terms_neg <- sort(tfidf_scores_neg, decreasing = TRUE) [1:50]
print(tfidf_terms_neg)
#World cloud 
wordcloud(names(tfidf_terms_neg), tfidf_terms_neg , max.words = 100, rot.per=0.15, 
          random.order = FALSE, scale=c(2,0.5),
          random.color = FALSE, colors=brewer.pal(8,"Dark2"))
title("Negative Reviews")
#Topic Modeling (Positive review)
result_pos <- FindTopicsNumber(
  dtm_lda_pos,
  topics = seq(from = 5, to = 20, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(result_pos)
ldaOut_pos <-LDA(dtm_lda_pos,13, method="Gibbs", 
                 control=list(iter=3000,seed=689))
phi_pos <- posterior(ldaOut_pos)$terms %>% as.matrix #matrix, with each row containing the distribution over terms for a topic
theta_pos <- posterior(ldaOut_pos)$topics %>% as.matrix #matrix, with each row containing the probability distribution over topics for a document
ldaOut.terms_pos <- as.matrix(terms(ldaOut_pos, 10))
ldaOut.terms_pos
#Topic Modeling (Negative review)
result_neg <- FindTopicsNumber(
  dtm_lda_neg,
  topics = seq(from = 5, to = 20, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
  method = "Gibbs",
  control = list(seed = 689),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(result_neg)
best_k_neg <- result_neg %>%
  filter(CaoJuan2009 == min(CaoJuan2009)) %>%
  pull(topics)
ldaOut_neg <-LDA(dtm_lda_neg,15, method="Gibbs", 
                 control=list(iter=3000,seed=1000))
phi_neg <- posterior(ldaOut_neg)$terms %>% 
  as.matrix 
theta_neg <- posterior(ldaOut_neg)$topics %>% as.matrix 
ldaOut.terms_neg <- as.matrix(terms(ldaOut_neg, 10))
ldaOut.terms_neg