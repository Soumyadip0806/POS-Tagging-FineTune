import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score
from conllu import parse_incr

# -------------- Argument Parsing -------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="POS Tagging Fine-tuning Script")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training CONLLU file")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to the validation CONLLU file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test CONLLU file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    return parser.parse_args()

# -------------- Load Dataset -------------- #
def read_conllu(file_path):
    dataset = {"tokens": [], "pos_tags": []}
    with open(file_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            tokens = [token["form"] for token in tokenlist]
            pos = [token["upos"] for token in tokenlist]
            dataset["tokens"].append(tokens)
            dataset["pos_tags"].append(pos)
    return dataset

# -------------- Tokenization & Alignment -------------- #
def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["pos_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(new_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# -------------- Evaluation Function -------------- #
def compute_metrics(p, id2label):
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

# -------------- Main Execution -------------- #
def main():
    args = parse_args()
    
    # Load datasets
    train_data = read_conllu(args.train_file)
    valid_data = read_conllu(args.valid_file)
    test_data = read_conllu(args.test_file)
    
    # Create label mappings
    unique_labels = list(set(sum(train_data["pos_tags"], [])))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    def convert_labels(dataset):
        dataset["pos_tags"] = [[label2id[tag] for tag in sentence] for sentence in dataset["pos_tags"]]
        return dataset
    
    train_data = convert_labels(train_data)
    valid_data = convert_labels(valid_data)
    test_data = convert_labels(test_data)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    train_dataset = Dataset.from_dict(train_data).map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=True)
    valid_dataset = Dataset.from_dict(valid_data).map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=True)
    test_dataset = Dataset.from_dict(test_data).map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=True)
    
    model = AutoModelForTokenClassification.from_pretrained(args.model_path, num_labels=len(label2id))
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    predictions = trainer.predict(test_dataset)
    compute_metrics(predictions, id2label)

if __name__ == "__main__":
    main()

