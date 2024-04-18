from pymongo import MongoClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from bson import ObjectId
from datetime import datetime
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re

# List of programming languages
programming_languages = ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'swift', 'go', 'rust', 'bash']

nltk.download('stopwords')

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["analyDb"]  # Replace 'your_database' with your actual database name
collection = db["queries"]  # Replace 'your_collection' with your actual collection name

# Fetch data from MongoDB
data = list(collection.find())

# print("data:: ", data)

# Extracting data for analysis
queries = [entry["queryText"] for entry in data]
feedbacks = [entry["feedback"] for entry in data if "feedback" in entry]
ratings = [entry["rating"] for entry in data if "rating" in entry]

# Query Analysis
user_queries = Counter([entry["user"]["name"] for entry in data])
topic_queries = Counter([entry["topic"] for entry in data])
query_lengths = [len(query.split()) for query in queries]

# Response Analysis
response_lengths = [len(entry["response"].split()) for entry in data]

# Rating Analysis
average_rating = sum(ratings) / len(ratings)

# Feedback Analysis
feedback_sentiments = [TextBlob(feedback).sentiment.polarity for feedback in feedbacks]

# Word Cloud Generation
all_words = ' '.join(queries)
stop_words = set(stopwords.words('english'))
filtered_words = ' '.join([word for word in all_words.split() if word.lower() not in stop_words])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_words)

# Visualizing Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Query Text')
plt.show()

# Sentiment Analysis Visualization
plt.hist(feedback_sentiments, bins=3, color='skyblue', edgecolor='black', linewidth=1.2)
plt.title('Sentiment Analysis of Feedbacks')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.xticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
plt.show()

# User Queries vs Counter
plt.figure(figsize=(10, 5))
plt.bar(user_queries.keys(), user_queries.values(), color='skyblue')
plt.title('User Queries Distribution')
plt.xlabel('User')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Topic Queries vs Counter
plt.figure(figsize=(10, 5))
plt.bar(topic_queries.keys(), topic_queries.values(), color='skyblue')
plt.title('Topic Queries Distribution')
plt.xlabel('Topic')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Print Analytics Results
print("Query Analysis:")
print("User Queries:", user_queries)
print("Topic Queries:", topic_queries)
print("Average Query Length:", sum(query_lengths) / len(query_lengths))
print("\nResponse Analysis:")
print("Average Response Length:", sum(response_lengths) / len(response_lengths))
print("\nRating Analysis:")
print("Average Rating:", average_rating)

# Descriptive statistics of feedback sentiments
mean_sentiment = sum(feedback_sentiments) / len(feedback_sentiments)
median_sentiment = sorted(feedback_sentiments)[len(feedback_sentiments) // 2]
mode_sentiment = max(set(feedback_sentiments), key=feedback_sentiments.count)

# Cognitive impact analysis
# Assuming negative sentiments indicate negative cognitive impact
negative_feedbacks = [sentiment for sentiment in feedback_sentiments if sentiment < 0]
percentage_negative_feedbacks = (len(negative_feedbacks) / len(feedback_sentiments)) * 100

# Usefulness analysis
# Assuming positive sentiments indicate usefulness
positive_feedbacks = [sentiment for sentiment in feedback_sentiments if sentiment > 0]
percentage_positive_feedbacks = (len(positive_feedbacks) / len(feedback_sentiments)) * 100

# Print analysis results
print("\nFeedback Analysis:")
print("Average Sentiment Polarity of Feedbacks:", mean_sentiment)
print("Median Sentiment Polarity of Feedbacks:", median_sentiment)
print("Mode Sentiment Polarity of Feedbacks:", mode_sentiment)
print("Percentage of Negative Feedbacks:", percentage_negative_feedbacks)
print("Percentage of Positive Feedbacks:", percentage_positive_feedbacks)

# Create scatter plot for sentiment analysis
plt.figure(figsize=(8, 6))
plt.scatter(feedback_sentiments, feedback_sentiments, color='skyblue', label='Actual vs Predicted')
plt.plot([-1, 1], [-1, 1], color='red', linestyle='--', label='Ideal Line')
plt.title('Sentiment Analysis')
plt.xlabel('Actual Sentiment Polarity')
plt.ylabel('Predicted Sentiment Polarity')
plt.legend()
plt.show()

# Function to predict programming language
def predict_programming_language(response):
    response_lower = response.lower()
    for lang in programming_languages:
        if re.search(r'\b' + lang + r'\b', response_lower):
            return lang
    return 'unknown'

# Predict programming languages for each response
response_languages = [predict_programming_language(entry["response"]) for entry in data if entry["topic"] == "programming"]

# Count occurrences of each programming language
language_counts = Counter(response_languages)

# Visualize programming language distribution
plt.bar(language_counts.keys(), language_counts.values(), color='skyblue')
plt.title('Programming Language Distribution in Responses')
plt.xlabel('Programming Language')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()