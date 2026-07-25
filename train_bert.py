import pandas as pd

from datasets import Dataset
from sklearn.metrics import accuracy_score
import numpy as np

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split

print("Loading datasets...")

# LOAD CSV
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

# MERGE
data = pd.concat([fake, real])

# COMBINE TEXT
data["content"] = (
    data["title"].fillna("") +
    " " +
    data["text"].fillna("")
)

data = data[["content", "label"]]

# SMALL SAMPLE
data = data.sample(
    44000,
    random_state=42
)

print("Dataset ready!")

# SPLIT
train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# DATASET
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# TOKENIZER
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):

    return tokenizer(
        batch["content"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

print("Tokenizing...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

# MODEL
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# SETTINGS
training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=5,

    per_device_train_batch_size=8,

    logging_steps=1
)
def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions)
    }
# TRAINER
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)

print("Training started...")

trainer.train()
results = trainer.evaluate()
print(results)
# SAVE
model.save_pretrained(
    "bert_fake_news_model"
)

tokenizer.save_pretrained(
    "bert_fake_news_model"
)

print("✅ DistilBERT model trained successfully!")