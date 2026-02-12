import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from news_analysis import get_cleaned_text

def analyze_model():
    print("Loading model...")
    try:
        with open("my_classifier.pickle", "rb") as f:
            vectorizer, clf = pickle.load(f)
    except FileNotFoundError:
        print("Model not found. Please run the app to train it first.")
        return

    # Check top features
    print("\n--- Identifying Top Features ---")
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(clf, 'coef_'):
        # For Linear models like Logistic Regression / PassiveAggressive
        coefs = clf.coef_[0]
        top_positive_coeffs = np.argsort(coefs)[-20:]
        top_negative_coeffs = np.argsort(coefs)[:20]
        
        print("\nTop 20 words for 'FAKE' (Negative Coeffs assuming fake code is 0/neg or real is 1/pos):")
        # Need to verify class mapping. Usually classes_ tells us.
        print(f"Classes: {clf.classes_}")
        # If 'FAKE' is index 0 and 'REAL' is index 1, then negative coefs -> FAKE, positive -> REAL.
        # Let's verify mapping by looking at classes_
        
        target_class_0 = clf.classes_[0] # Likely FAKE?
        target_class_1 = clf.classes_[1] # Likely REAL?
        
        print(f"Class 0 (Negative coefs tend towards): {target_class_0}")
        print(f"Class 1 (Positive coefs tend towards): {target_class_1}")

        top_features_0 = [feature_names[i] for i in top_negative_coeffs]
        top_features_1 = [feature_names[i] for i in top_positive_coeffs]
        
        print(f"\nStrongest features for {target_class_0}:")
        print(", ".join(top_features_0))
        
        print(f"\nStrongest features for {target_class_1}:")
        print(", ".join(top_features_1))
    else:
        print("Model does not expose coefficients easily (probably Naive Bayes).")
        # For NB, we can look at feature_log_prob_

    # Analyze specific files
    files_to_test = [
        "Local Cat Elected Mayor After Promi.txt",
        "🌍 Global & World Events.txt"
    ]

    print("\n--- Analyzing Specific Files ---")
    for filename in files_to_test:
        safe_filename = filename.encode('ascii', 'ignore').decode('ascii')
        print(f"\nAnalyzing: {safe_filename}")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            
            cleaned_text = get_cleaned_text(text)
            print(f"Cleaned Text Snippet: {cleaned_text[:100]}...")
            
            vect = vectorizer.transform([cleaned_text])
            prediction = clf.predict(vect)[0]
            
            prob = "N/A"
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(vect)[0]
                prob = probs.max()
            
            print(f"Prediction: {prediction}")
            print(f"Confidence: {prob}")
            
            # Feature contribution for this specific text
            if hasattr(clf, 'coef_'):
                # element-wise multiply vector by coefficients
                # This is a sparse matrix operation
                pass 
                # Doing a manual check of which words in the text have high coeffs
                words = cleaned_text.split()
                word_scores = []
                for word in words:
                    if word in vectorizer.vocabulary_:
                        idx = vectorizer.vocabulary_[word]
                        score = clf.coef_[0][idx]
                        word_scores.append((word, score))
                
                word_scores.sort(key=lambda x: x[1])
                print("Top words pushing towards Class 0 (Fake?):")
                print(word_scores[:5])
                print("Top words pushing towards Class 1 (Real?):")
                print(word_scores[-5:])

        except FileNotFoundError:
            print(f"File {filename} not found.")

if __name__ == "__main__":
    analyze_model()
