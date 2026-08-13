# Ex.No 10 - Fine-Tuning DistilBERT for Domain-Specific Text Classification
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer)
import numpy as np
from sklearn.metrics import accuracy_score

# 1. Load a domain-specific dataset (example: IMDB movie reviews)
dataset = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

# 2. Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = small_train.map(tokenize, batched=True)
test_ds = small_test.map(tokenize, batched=True)

# 3. Load pre-trained model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# 4. Training arguments
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_steps=50
)

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# 5. Train
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                   eval_dataset=test_ds, compute_metrics=compute_metrics)
trainer.train()

# 6. Evaluate and save
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)
model.save_pretrained("./fine_tuned_distilbert_imdb")
