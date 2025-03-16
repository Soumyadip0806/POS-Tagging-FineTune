import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, AdamW
from transformers import DataCollatorForTokenClassification, TrainerCallback, TrainerControl, TrainerState
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score
from conllu import parse_incr

# -------------- Argument Parser -------------- #
parser = argparse.ArgumentParser(description="Train a POS tagging model.")
parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset (CoNLL-U format).")
parser.add_argument("--valid_file", type=str, required=True, help="Path to the validation dataset (CoNLL-U format).")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset (CoNLL-U format).")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training.")
args = parser.parse_args()

# -------------- Load Dataset -------------- #
def read_conllu(file_path):
    """Reads a CoNLL-U formatted file and extracts tokens and POS tags."""
    dataset = {"tokens": [], "pos_tags": []}
    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            tokens = [token["form"] for token in tokenlist]
            pos = [token["upos"] for token in tokenlist]
            dataset["tokens"].append(tokens)
            dataset["pos_tags"].append(pos)
    return dataset

train_data = read_conllu(args.train_file)
valid_data = read_conllu(args.valid_file)
test_data = read_conllu(args.test_file)

# -------------- Label Encoding -------------- #
unique_labels = list(set(sum(train_data["pos_tags"], [])))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

def convert_labels(dataset):
    """Converts POS tags to their corresponding integer labels."""
    dataset["pos_tags"] = [[label2id[tag] for tag in sentence] for sentence in dataset["pos_tags"]]
    return dataset

train_data = convert_labels(train_data)
valid_data = convert_labels(valid_data)
test_data = convert_labels(test_data)

# -------------- Tokenization & Label Alignment -------------- #
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

def tokenize_and_align_labels(examples):
    """Tokenizes input and aligns labels to match subword tokens."""
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["pos_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(new_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = Dataset.from_dict(train_data).map(tokenize_and_align_labels, batched=True)
valid_dataset = Dataset.from_dict(valid_data).map(tokenize_and_align_labels, batched=True)
test_dataset = Dataset.from_dict(test_data).map(tokenize_and_align_labels, batched=True)

# -------------- Load Model & Gradual Training Setup -------------- #
num_labels = len(label2id)
model = AutoModelForTokenClassification.from_pretrained(args.model_path, num_labels=num_labels)

# Freeze all layers initially except classifier
for param in model.base_model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

class GradualUnfreezingCallback(TrainerCallback):
    """Gradually unfreezes model layers during training."""
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = int(state.epoch)
        if epoch == 2:
            for param in model.base_model.encoder.layer[-3:].parameters():
                param.requires_grad = True
        elif epoch == 8:
            for param in model.base_model.encoder.layer[-6:].parameters():
                param.requires_grad = True
        elif epoch == 16:
            for param in model.base_model.parameters():
                param.requires_grad = True
        print(f"Epoch {epoch}: Unfreezing model layers")

# -------------- Optimizer Setup -------------- #
optimizer = AdamW([
    {"params": model.base_model.embeddings.parameters(), "lr": 1e-5},
    {"params": model.base_model.encoder.layer[:6].parameters(), "lr": 2e-5},
    {"params": model.base_model.encoder.layer[6:].parameters(), "lr": 3e-5},
    {"params": model.classifier.parameters(), "lr": 5e-5},
], weight_decay=0.01)

# -------------- Training Arguments -------------- #
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    warmup_steps=500,
    learning_rate=args.learning_rate
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    optimizers=(optimizer, None),
    callbacks=[GradualUnfreezingCallback()]
)

# -------------- Train & Save Model -------------- #
trainer.train()
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# -------------- Evaluation -------------- #
def compute_metrics(p):
    """Computes evaluation metrics (accuracy & classification report)."""
    predictions, labels = p.predictions, p.label_ids
    predictions = np.argmax(predictions, axis=2)
    
    true_labels = [id2label[l] for label in labels for l in label if l != -100]
    true_predictions = [id2label[p] for prediction, label in zip(predictions, labels)
                        for p, l in zip(prediction, label) if l != -100]
    
    accuracy = accuracy_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    
    print("Classification Report:\n", classification_report(true_labels, true_predictions, zero_division=0))
    
    return {
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

predictions = trainer.predict(test_dataset)
compute_metrics(predictions)

