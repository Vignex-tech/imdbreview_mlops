import json
import pathlib
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import numpy as np
import mlflow


MIN_F1 = 0.85

MODEL_DIR = pathlib.Path('model')
MODEL_DIR.mkdir(exist_ok=True)

ds =  load_dataset('stanfordnlp/imdb')
df = pd.DataFrame(ds['train']).sample(500,random_state=100)

X_train,X_test,y_train,y_test = train_test_split(df['text'],df['label'],test_size = 0.2,random_state = 42, stratify = df['label'])

model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2),min_df=2 , sublinear_tf=True)), ('clf', LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)

pred = model.predict(X_test)

f1 = f1_score(y_test, pred, average='macro')

if f1 < MIN_F1:
    raise SystemExit(f"F1 score {f1:.3f} is below the minimum threshold of {MIN_F1}")

joblib.dump(model, MODEL_DIR / 'model.joblib')    
json.dump({"macro_f1": f1}, open(MODEL_DIR / "metrics.json", "w"))

print(f"macro F1: {f1:.3f}")
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

cv = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
print(f"Cross-validation F1 scores: {cv}, score mean: {cv.mean()}, score std: {cv.std()}")

mlflow.set_experiment("imdb-review")
tfidf = model.named_steps['tfidf']
lr = model.named_steps['clf']
words = tfidf.get_feature_names_out()
weights = lr.coef_[0]
order = np.argsort(weights)


with mlflow.start_run():
    mlflow.log_param("ngram_range", (1, 2))
    mlflow.log_param("min_df", 2)
    mlflow.log_metric("macro_f1", f1)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("top_positive_words", [words[i] for i in order[-10:]])
    mlflow.log_param("top_negative_words", [words[i] for i in order[:10]])