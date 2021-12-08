import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import nltk
# Uncomment to download "stopwords"
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

df = pd.read_csv('../PSIMILAN/suicide_datasetv2.csv')

encoding = {
    'suicidal': 0,
    'normal': 1
}

y_encoded = [encoding[cls] for cls in df['class'].values]

X = df['text'].values
y = np.array(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove [name], [religion] (only for GoEmotion Dataset)
    s = re.sub(r'(\[name\])[\s]', '', s)
    s = re.sub(r'(\[religion\])[\s]', '', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

X_train_preprocessed = np.array([text_preprocessing(text) for text in X_train])
X_test_preprocessed = np.array([text_preprocessing(text) for text in X_test])

tf_idf = TfidfVectorizer(ngram_range=(1, 3), binary=True, smooth_idf=True)
X_train_tfidf = tf_idf.fit_transform(X_train_preprocessed)
X_test_tfidf = tf_idf.transform(X_test_preprocessed)

# Run NB classifier
print("NB")
nb_classifier = MultinomialNB(alpha=1.2, fit_prior=False)
t0 = time.time()
nb_classifier.fit(X_train_tfidf, y_train)
print("Training time:", round(time.time()-t0, 3), "s")

t1 = time.time()
y_pred = nb_classifier.predict(X_test_tfidf)
print("Predict time:", round(time.time()-t1, 3), "s")

parameters = {'alpha': np.arange(1, 10, 0.1), 'fit_prior': [True, False]}
grid_nb_clf = GridSearchCV(nb_classifier, parameters)
grid_nb_clf.fit(X_train_tfidf, y_train)
y_pred = grid_nb_clf.predict(X_test_tfidf)

df_label_columns = df.columns[9:]
label_names = list(df_label_columns)
report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('../PSIMILAN/nb_suicide_report.csv')

# Run SVM classifier
print("SVM")
svm_classifier = svm.LinearSVC(C=0.2, class_weight='balanced', random_state=42)
t0 = time.time()
svm_classifier.fit(X_train_tfidf, y_train)
print("Training time:", round(time.time()-t0, 3), "s")

t1 = time.time()
y_score = svm_classifier.decision_function(X_test_tfidf)
y_pred = svm_classifier.predict(X_test_tfidf)
print("Predict time:", round(time.time()-t1, 3), "s")

parameters = {'C': np.arange(0.1, 1.1, 0.1)}
grid_svm_clf = GridSearchCV(svm_classifier, parameters)
grid_svm_clf.fit(X_train_tfidf, y_train)
y_pred = grid_svm_clf.predict(X_test_tfidf)

report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('../PSIMILAN/svm_suicide_report.csv')



