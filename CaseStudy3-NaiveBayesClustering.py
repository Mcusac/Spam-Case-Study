# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
# The dataset should have 'text' column for email content and 'label' column for spam/ham labels
dataset = pd.read_csv('processed_emails.csv')

# Handle NaN values in the 'Content' column by replacing them with an empty string
dataset['Content'].fillna('', inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['Content'], dataset['Label'], test_size=0.2, random_state=42)

# Naive Bayes Classifier

# Convert text data into numerical vectors using CountVectorizer
cv = CountVectorizer()
X_train_vectorized = cv.fit_transform(X_train)
X_test_vectorized = cv.transform(X_test)

# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# Predict on the test set
nb_predictions = nb_classifier.predict(X_test_vectorized)

# Evaluate the Naive Bayes classifier
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_confusion_matrix = confusion_matrix(y_test, nb_predictions)

print("Naive Bayes Classifier:")
print(f"Accuracy: {nb_accuracy}")
print("Confusion Matrix:")
print(nb_confusion_matrix)

# Calculate additional metrics
precision = precision_score(y_test, nb_predictions, pos_label='ham')
recall = recall_score(y_test, nb_predictions, pos_label='ham')
f1 = f1_score(y_test, nb_predictions, pos_label='ham')

# Print additional metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Clustering (K-Means)
# Convert text data into numerical vectors using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(dataset['Content'])

# Train a K-Means clustering model
num_clusters = 2  # Assuming 2 clusters for spam/ham
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_tfidf)

# Assign cluster labels to the original dataset

dataset['cluster_label'] = kmeans.labels_

# Explore the clusters and check if one cluster corresponds to spam
print("Cluster Distribution:")
print(dataset['cluster_label'].value_counts())

# more metrics
def calculate_metrics(true_labels, predicted_labels, pos_label=None):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label=pos_label)
    recall = recall_score(true_labels, predicted_labels, pos_label=pos_label)
    f1 = f1_score(true_labels, predicted_labels, pos_label=pos_label)
    return accuracy, precision, recall, f1


true_labels = dataset['Label']

cluster_to_label_map = {0: 'ham', 1: 'spam'}  # Adjust mapping if needed

predicted_labels = dataset['cluster_label'].map(cluster_to_label_map)

metrics = calculate_metrics(true_labels, predicted_labels, pos_label='ham')  # Adjust based on your spam/ham definition

print("Accuracy:", metrics[0])
print("Precision:", metrics[1])
print("Recall:", metrics[2])
print("F1-score:", metrics[3])

# Note: Further analysis and tuning may be needed based on your dataset and specific requirements.
