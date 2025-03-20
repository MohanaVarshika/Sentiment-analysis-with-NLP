# Sentiment-analysis-with-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: BEESETTY MOHANA VARSHIKA

INTERN ID: CT12WKFH

DOMAIN: MACHINE LEARNING

BATCH DURATION: JANUARY 10TH,2025 TO APRIL 10TH,2025

MENTOR NAME: NEELA SANTHOSH

DESCRIPTION:

Sentiment Analysis, also known as opinion mining, is a Natural Language Processing (NLP) technique used to determine the emotional tone of a piece of text. It classifies text into sentiments such as positive, negative, or neutral, helping businesses and researchers analyze customer opinions, social media posts, product reviews, and more.
=>Use Cases of Sentiment Analysis
Social Media Monitoring: Understanding public opinions on platforms like Twitter, Facebook, and Instagram.
Customer Feedback Analysis: Evaluating product and service reviews on e-commerce platforms.
Brand Reputation Management: Tracking company mentions and identifying negative sentiments.
Market Research: Analyzing trends in consumer preferences.
Financial Analysis: Assessing public sentiment toward stocks or companies.
Sentiment analysis typically involves three main steps:

A. Data Collection
Text data is collected from sources like:

Social Media (Twitter, Reddit, Facebook, etc.)
Product Reviews (Amazon, Yelp, etc.)
Customer Support Chats
News Articles & Blogs
B. Text Preprocessing
Since raw text contains noise, it must be cleaned using NLP techniques like:

Tokenization: Splitting text into individual words or phrases.
Removing Stopwords: Eliminating common words like "is", "the", "and", etc.
Stemming/Lemmatization: Reducing words to their root forms (e.g., "running" → "run").
Removing Special Characters & Punctuation.
Lowercasing: Converting all text to lowercase for consistency.
C. Sentiment Classification
The processed text is analyzed using one of the following approaches:

Rule-Based Approach:

Uses predefined sentiment dictionaries (e.g., SentiWordNet, VADER for Twitter).
Assigns a sentiment score based on positive/negative words in the text.
Machine Learning-Based Approach:

Uses supervised learning models like Naïve Bayes, Logistic Regression, Random Forest, or SVM.
Requires labeled datasets where text is already categorized as positive, negative, or neutral.
Features are extracted using techniques like TF-IDF or word embeddings (Word2Vec, GloVe).
Deep Learning-Based Approach:

Uses advanced neural networks like LSTMs, CNNs, or Transformers (BERT, GPT).
Learns from massive amounts of data and detects complex patterns in sentiment.

CODE:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re

EXPLAINATION:

numpy: A library used for numerical operations, especially working with arrays and matrices.
pandas: A powerful library used for data manipulation and analysis, particularly for handling data in tabular form (DataFrames).
matplotlib.pyplot: A plotting library for creating static, animated, and interactive visualizations, commonly used for creating plots like histograms, line graphs, and scatter plots.
seaborn: Built on top of matplotlib, Seaborn is used for making attractive and informative statistical graphics, like heatmaps, boxplots, etc.
nltk: The Natural Language Toolkit is a comprehensive library for working with human language data (text).
PorterStemmer: A tool to perform stemming, which reduces words to their base form (e.g., "running" → "run").
stopwords: A module in NLTK that provides a list of common words (e.g., "the", "and") that are often removed in NLP tasks as they do not contribute significant meaning.
STOPWORDS: A set containing stopwords from the English language, useful for cleaning text data.
train_test_split: Splits the dataset into training and testing sets.
MinMaxScaler: A feature scaling technique that transforms features into a range between 0 and 1.
CountVectorizer: Converts a collection of text documents into a matrix of token counts (bag-of-words model).
cross_val_score: Performs cross-validation to evaluate the performance of a model using different data subsets.
RandomForestClassifier: A machine learning model based on ensemble learning, combining several decision trees for classification tasks.
confusion_matrix & ConfusionMatrixDisplay: Evaluate model performance by comparing predicted vs. actual labels.
GridSearchCV: Used for hyperparameter tuning by searching through a grid of possible hyperparameters to find the best combination for the model.
StratifiedKFold: Used for cross-validation, ensuring each fold maintains the same percentage of samples for each class.
accuracy_score: Measures the accuracy of a model's predictions.
DecisionTreeClassifier: A machine learning algorithm that splits data into decision nodes to classify based on features.
XGBClassifier: A gradient boosting algorithm that combines the predictions of multiple weak models to produce a stronger model.

OUTPUT:
![Image](https://github.com/user-attachments/assets/ddb989c4-4884-4eb9-a0e3-5902d941d06d)
![Image](https://github.com/user-attachments/assets/c7c019b0-59ed-4afc-b17e-97cd809ea211)
![Image](https://github.com/user-attachments/assets/559e1c1e-aaa6-4b99-bed3-8030859c7032)

