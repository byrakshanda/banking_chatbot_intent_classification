# Banking Chatbot Query-Intent Classification (Naive Bayes)

## What this project does
Classifies banking customer queries (like "I still haven't received my new card") 
into the correct intent category, so a chatbot can understand what the customer needs.

## Dataset
3,080 customer queries across 77 banking intents (e.g., card arrival, PIN blocked, 
refund not showing up). The intents are fairly balanced, at around 40 queries each.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook

## What I did
- Checked for missing values, duplicates, and empty queries (dataset was clean)
- Explored the top banking intents, most frequent words, and query length distribution
- Converted text into numbers using CountVectorizer (Bag-of-Words: counts how many 
  times each word appears)
- Split data 80/20 into training and test sets
- Trained a Multinomial Naive Bayes model
- Evaluated using Accuracy, weighted Precision, Recall, and F1 Score, plus a 
  confusion matrix across all 77 intents
- Tested the model on a new query ("I forgot my PIN") to see a real prediction

## Results
| Metric               | Score |
|----------------------|-------|
| Accuracy             | 0.722 |
| Precision (weighted) | 0.782 |
| Recall (weighted)    | 0.722 |
| F1 Score (weighted)  | 0.720 |

## Sample Prediction
Input: "I forgot my PIN" → Predicted intent: `pin_blocked`

## Key Insight
With 77 different intents to choose from, a simple Naive Bayes model reaches about 
72% accuracy using only word counts. Most queries are short (under 15 words), and 
common words like "card", "my", and "account" appear frequently across intents.

## How to run it
Open `banking_chatbot_intent_classification.ipynb` in Jupyter Notebook or Google 
Colab and run all cells. Requires `dev.csv` in the same folder.
