import joblib
import pathlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, confusion_matrix


MIN_F1 = 0.85


MODEL_DIR = pathlib.Path("model")
MODEL_DIR.mkdir(exist_ok=True)

ds = load_dataset("stanfordnlp/imdb')
df = pd.DataFrame(ds['train']).sample(500,random_state = 42)

X_train, X_test , y_train, y_test = train_test_split(df['text'],df['label'],test_size = 0.2, random_state = 50, stratify = df['label'])

model = Pipeline([('tfid',TfidfVectorizer(ngram_range=(1,2),min_df=2, sublinear_tf = True),('lr',LogisticRegression(max_iteration=1500))])

model.fit(X_train, y_train)

pred = model.predict(X_test)

f1 = f1_score(y_test,pred, average = 'macro')

if f1 < MIN_F1:
    raise SystemExit(f"F1 score {f1:.3f} is below the minimum threshold of {MIN_F1}")
joblib.dump(model, MODEL_DIR / 'model.joblib')
json.dump({"macro_f1": f1}, open(MODEL_DIR / "metrics.json", "w"))

cm = confusion_matrix(y_test,pred)
cr = classification_report(y_test,pred)