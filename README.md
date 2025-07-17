# Spam-Email-Detection-Using-NLP-and-Naive-Bayes

Introduction
Email spam detection is a critical Natural Language Processing (NLP) application that helps filter unwanted emails automatically. This project aims to classify emails as Spam or Not Spam using text preprocessing, TF-IDF feature extraction, and a Naive Bayes classifier.

Objective
To preprocess and clean email text data systematically.
To convert text into numerical features using TF-IDF Vectorization.
To train a Naive Bayes model to predict whether an email is spam.
To evaluate model performance and build a reusable prediction function.

Tools Used
Python for implementation.
Pandas, NumPy for data manipulation.
Matplotlib, Seaborn for optional visualization.
NLTK for stemming (Porter Stemmer).
Scikit-learn (sklearn) for:
TF-IDF Vectorization (TfidfVectorizer)
Model training (MultinomialNB)

Accuracy
Using the Multinomial Naive Bayes classifier with TF-IDF features, the model achieved:
✅ Accuracy: ~97% on the test set
✅ High precision and recall on spam class, making it effective in detecting spam while minimizing false positives.

Conclusion
This project successfully demonstrates spam email classification using NLP and machine learning. By systematically cleaning text, extracting relevant features, and using a probabilistic model, the system effectively predicts whether an email is spam.
