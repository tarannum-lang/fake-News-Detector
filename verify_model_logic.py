import news_analysis
import os

print("Starting verification...")
# Force training
if os.path.exists("my_classifier.pickle"):
    os.remove("my_classifier.pickle")

vectorizer, clf = news_analysis.get_trained_model()
print("Model trained successfully.")

# Test with a known string
text = "The world is flat and the moon is made of cheese."
result = news_analysis.get_analyzation_result(clf, vectorizer, text)
print(f"Test Result: {result}")
