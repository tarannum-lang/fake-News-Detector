import pickle
import pandas as pd
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure NLTK data is available
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except (LookupError, OSError):
    try:
        nltk.download('averaged_perceptron_tagger')
    except:
        pass

stop_words = set(stopwords.words("english"))
# Add US-Election specific noise words and potential 'fake' triggers that are actually neutral
stop_words.update([
    "hillary", "clinton", "donald", "trump", "podesta", "wikileaks", "fbi", "gop", "democrat", "republican",
    "western", "daily", "article", "share", "post", "link", "image", "via",
    "british", "canada"
])

lemmatizer = WordNetLemmatizer()
# min_df=5 removes rare words (noise), ngram_range(1,2) captures phrases
vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_df=0.7, min_df=5, ngram_range=(1, 2))
sentiment_analyser = SentimentIntensityAnalyzer()


def get_part_of_speech_tag(token):
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    try:
        tag = pos_tag([token])[0][1][0].upper()
    except IndexError:
        return wordnet.NOUN
        
    return tag_dict.get(tag, wordnet.NOUN)


def get_tokens(text):
    return word_tokenize(text.lower())


def get_cleaned_tokens(text):
    tokens = get_tokens(text)
    return [
        token
        for token in tokens
        if (token.isalpha() and token not in stop_words)
    ]


def get_lemmatized_tokens(tokens):
    return [
        lemmatizer.lemmatize(token, get_part_of_speech_tag(token)) for token in tokens
    ]


def get_cleaned_text(text):
    # Simplified cleaning: just tokenizing and stopword removal often works better for this dataset
    # than heavy lemmatization which can strip meaning. 
    # Removed lemmatization for speed and better generalized performance
    cleaned_tokens = get_cleaned_tokens(text)
    return " ".join(cleaned_tokens)


def get_polarity_score(text):
    return sentiment_analyser.polarity_scores(text)["compound"]


def get_analyzation_result(clf, vectorizer, article):
    cleaned_article = get_cleaned_text(article)

    polarity_score = get_polarity_score(cleaned_article)
    if abs(polarity_score) < 0.01:
        sentiment = "neutral"
    elif polarity_score > 0:
        sentiment = "positive"
    else:
        sentiment = "negative"

    article_vect = vectorizer.transform([cleaned_article])
    
    # Check if classifier implies probability support
    if hasattr(clf, "predict_proba"):
        predicted_probs = clf.predict_proba(article_vect)[0]
        predicted_label_index = predicted_probs.argmax()
        predicted_label = clf.classes_[predicted_label_index]
        confidence_level = predicted_probs[predicted_label_index]
        confidence_percent = int(round(confidence_level * 100))
    else:
        # Fallback for classifiers without predict_proba (like basic PassiveAggressive)
        predicted_label = clf.predict(article_vect)[0]
        # Use decision function distance as a proxy or just default to 100% since we can't easily get prob without calibration
        try:
            decision = clf.decision_function(article_vect)[0]
            # Sigmoid approximation for confidence
            import numpy as np
            confidence_level = 1 / (1 + np.exp(-abs(decision)))
            confidence_percent = int(round(confidence_level * 100))
        except:
            confidence_percent = "N/A"

    return {
        "sentiment": sentiment,
        "label": predicted_label,
        "confidence": confidence_percent,
    }


def get_text_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def is_model_trained():
    try:
        return os.path.exists("my_classifier.pickle")
    except (OSError, IOError):
        return False


def get_trained_model():
    try:
        with open("my_classifier.pickle", "rb") as f:
            vectorizer, clf = pickle.load(f)
        return vectorizer, clf
    except (OSError, IOError, EOFError, pickle.UnpicklingError):
        # Retrain
        print("Training model...")
        
        # We need to restart vectorizer to fit new data
        # Increase n-grams to capture phrases like "white house", "mass shooting", min_df to filter noise
        vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_df=0.7, min_df=5, ngram_range=(1, 2))
        
        df = pd.read_csv("fake_or_real_news.csv")
        
        # Simple progress indication
        print("Cleaning text data...")
        df["cleaned_text"] = df["text"].apply(get_cleaned_text)

        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_text"], df["label"], test_size=0.2, random_state=42
        )

        print("Vectorizing...")
        X_train_vect = vectorizer.fit_transform(X_train)

        # Using LogisticRegression - increasing regularization (C=0.1) to avoid overfitting to specific words
        print("Fitting Classifier...")
        clf = LogisticRegression(random_state=42, C=0.5) 
        clf.fit(X_train_vect, y_train)
        
        # If accuracy check is needed:
        # X_test_vect = vectorizer.transform(X_test)
        # print("Accuracy:", clf.score(X_test_vect, y_test))

        with open("my_classifier.pickle", "wb") as f:
            pickle.dump((vectorizer, clf), f)

        return vectorizer, clf


def save_result_in_file(article_text, article_sentiment, real_or_fake, accuracy):
    result = f"{accuracy}% {real_or_fake}"
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(f"Article text: {article_text}\n")
        f.write(f"Article sentiment: {article_sentiment}\n")
        f.write(f"Real or fake: {real_or_fake}\n")
        f.write(f"Accuracy: {result}\n")
