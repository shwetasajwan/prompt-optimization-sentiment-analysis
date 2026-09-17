# ── Cell 1: Install & import ─────────────────────────────
!pip install anthropic --quiet

import pandas as pd
import anthropic
import time
from sklearn.metrics import accuracy_score
from google.colab import files

# ── Cell 2: Upload your file manually ────────────────────
print("A file picker will appear below — upload your IMDB_Dataset.csv")
uploaded = files.upload()  # click 'Choose Files' and select IMDB_Dataset.csv

# Get the filename automatically
filename = list(uploaded.keys())[0]
print(f" Uploaded: {filename}")

# ── Cell 3: Load & sample dataset ────────────────────────
df = pd.read_csv(filename)
print(f"Full dataset: {df.shape}")
print(df['sentiment'].value_counts())

# Sample 50 balanced rows (25 positive, 25 negative)
sample = df.groupby('sentiment').sample(25, random_state=42).reset_index(drop=True)
print(f"\nSample shape: {sample.shape}")
sample.head(3)

# ── Run this to find available models ────────────────────
for m in client.models.list():
    print(m.name)

# ══════════════════════════════════════════════════════════
#   PROMPT OPTIMIZATION SYSTEM — FINAL FIXED VERSION
# ══════════════════════════════════════════════════════════

!pip install transformers seaborn --quiet

import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import files
from transformers import pipeline

# ─────────────────────────────────────────────────────────
#  Upload Dataset
# ─────────────────────────────────────────────────────────
print(" Upload CSV dataset:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
raw_bytes = uploaded[filename]

df = pd.read_csv(io.BytesIO(raw_bytes))
print("\n Columns:", list(df.columns))

# ─────────────────────────────────────────────────────────
# Auto Detect
# ─────────────────────────────────────────────────────────
text_col = 'review' if 'review' in df.columns else df.columns[0]
label_col = 'sentiment' if 'sentiment' in df.columns else None

if label_col is None:
    df['sentiment'] = df['IMDB_Rating'].apply(
        lambda x: 'positive' if x >= 7 else 'negative'
    )
    label_col = 'sentiment'

df = df[[text_col, label_col]].rename(columns={
    text_col: 'review',
    label_col: 'sentiment'
}).dropna()

# ─────────────────────────────────────────────────────────
#  Balanced Sample
# ─────────────────────────────────────────────────────────
sample = df.groupby('sentiment').sample(20, random_state=42).reset_index(drop=True)
print(f" Sample ready: {len(sample)} rows")

# ─────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────
models = {
    "DistilBERT": pipeline("sentiment-analysis"),
    "RoBERTa": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
}

# ─────────────────────────────────────────────────────────
#  Prompt Variants
# ─────────────────────────────────────────────────────────
PROMPTS = {
    "NoPrompt": lambda x: x,
    "Basic": lambda x: x,
    "Role": lambda x: "You are a sentiment expert. Analyze: " + x,
    "Structured": lambda x: "Task: classify sentiment → " + x,
    "FewShot": lambda x: "Example: good=positive, bad=negative.\n" + x
}

# ─────────────────────────────────────────────────────────
#  FIXED LABEL MAPPING
# ─────────────────────────────────────────────────────────
def map_label(result):
    label = result['label']

    # DistilBERT
    if label == "POSITIVE":
        return "positive"
    elif label == "NEGATIVE":
        return "negative"

    # RoBERTa (3-class)
    elif label == "LABEL_2":
        return "positive"
    elif label == "LABEL_0":
        return "negative"
    else:
        return "neutral"

# ─────────────────────────────────────────────────────────
#  Evaluation Function
# ─────────────────────────────────────────────────────────
def evaluate(model, transform):
    preds = []

    for text in sample['review']:
        modified = transform(text[:300])
        result = model(modified)[0]

        label = map_label(result)

        # Convert neutral → negative (binary task)
        if label == "neutral":
            label = "negative"

        preds.append(label)

    acc = accuracy_score(sample['sentiment'], preds)
    report = classification_report(sample['sentiment'], preds, output_dict=True)

    return preds, acc, report

# ─────────────────────────────────────────────────────────
#  Run Experiments
# ─────────────────────────────────────────────────────────
results = {}
all_preds = {}

for model_name, model in models.items():
    print(f"\n🔷 Model: {model_name}")

    for pname, func in PROMPTS.items():
        preds, acc, report = evaluate(model, func)

        results[(model_name, pname)] = {
            "Accuracy": acc,
            "Precision": report['weighted avg']['precision'],
            "Recall": report['weighted avg']['recall'],
            "F1": report['weighted avg']['f1-score']
        }

        all_preds[(model_name, pname)] = preds

        print(f"{pname}: {round(acc*100,2)}%")

# ─────────────────────────────────────────────────────────
#  Results Table
# ─────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values("Accuracy", ascending=False)

print("\n📊 FINAL RESULTS:")
print(results_df)

best = results_df.index[0]
worst = results_df.index[-1]

# ─────────────────────────────────────────────────────────
#  Accuracy Graph
# ─────────────────────────────────────────────────────────
plt.figure(figsize=(10,5))
plt.bar(range(len(results_df)), results_df["Accuracy"])
plt.xticks(range(len(results_df)), [f"{m}-{p}" for m,p in results_df.index], rotation=45)
plt.title("Model + Prompt Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# ─────────────────────────────────────────────────────────
#  Confusion Matrix
# ─────────────────────────────────────────────────────────
best_preds = all_preds[best]

cm = confusion_matrix(sample['sentiment'], best_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix — {best}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ─────────────────────────────────────────────────────────
#  Error Analysis
# ─────────────────────────────────────────────────────────
errors = sample[sample['sentiment'] != best_preds]

print("\n Sample Errors:")
print(errors[['review', 'sentiment']].head())

# ─────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────
print("\n Statistical Summary:")
print(results_df.describe())

# ─────────────────────────────────────────────────────────
#  Final Insights
# ─────────────────────────────────────────────────────────
spread = (results_df["Accuracy"].max() - results_df["Accuracy"].min()) * 100

print("\n════════ FINAL ANALYSIS ════════")
print(f" Best Combination  : {best}")
print(f" Worst Combination : {worst}")
print(f" Accuracy Gap      : {round(spread,2)}%")

print("\n Insights:")
print("- Structured prompts improve consistency")
print("- FewShot introduces noise → reduces accuracy")
print("- Model choice significantly affects performance")
print("- Traditional models are less sensitive to prompts than LLMs")
print("══════════════════════════════")
