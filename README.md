## Analysing Hotel Reviews to Identify Factors Affecting Customer Satisfaction

## Project Overview 
This project analyses hotel customer reviews to identify key factors driving customer satisfaction and dissatisfaction

## Data
- File: `HotelsData.csv`
- Variables:
  - `Review.score` — review rating (1–5)
  - `Text.1` — review text

## Methodology
1. **Language detection & sampling**
   - Detect review language using `cld3`
   - Filter English reviews
   - Sample 2,000 reviews with fixed seed for reproducibility

2. **Sentiment-based segmentation**
   - Positive reviews: `Review.score >= 4`
   - Negative reviews: `Review.score <= 2`
   - Neutral reviews excluded

3. **Text preprocessing**
   - Lowercasing, punctuation/number removal, stopword removal, whitespace stripping
   - Lemmatization (`textstem`)
   - Separate corpora for positive and negative reviews

4. **Feature engineering (TF–IDF)**
   - Build Document-Term Matrices (DTM)
   - Remove sparse terms and empty documents
   - Compute TF–IDF and retain top informative terms (top 25%)

5. **Emotion analysis**
   - NRC lexicon-based emotion scoring (`syuzhet`)
   - Visualisation of overall emotional tone

6. **Topic modeling (LDA)**
   - Determine optimal number of topics using `ldatuning` metrics
   - Extract top terms per topic and interpret themes

## Key Findings (Summary)
- **Positive themes**: staff/service quality,location, comfort, breakfast/food, smooth check-in/out
- **Negative themes**: cleanliness/room condition, bathroom/shower issues, room size, waiting/service delays, price/value concerns and unexpected charges
- Emotion analysis indicates a predominance of positive emotions (e.g., trust/joy) alongside a non-trivial presence of negative emotions, supporting the need for targeted improvements.

