# 🏷️ POS Tagging Fine-Tuning with Transformers

Welcome to the **POS Tagging Fine-tuning Repository**! This project fine-tunes transformer-based models for **Part-of-Speech (POS) tagging** using **CoNLL-U formatted datasets**. We provide two training approaches:  

✅ **Standard Fine-tuning** – Trains the model normally.  
🔥 **Gradual Unfreezing Fine-tuning** – Gradually unfreezes layers for better adaptation. 


## 📌 Features  
- 🏗 Supports **any transformer-based model** from Hugging Face 🤗.  
- 📖 Reads data in **CoNLL-U format**.  
- 🚀 **Gradual unfreezing** for stable training.  
- 📊 Computes **accuracy & classification reports**.  

---


## 📂 Repository Structure
```
├── 📝 gradual_finetune.py   → Fine-tuning with gradual unfreezing  
├── ⚡ normal_finetune.py    → Standard fine-tuning method  
├── 📦 requirements.txt      → All dependencies listed here  
├── 📁 dataset/              → Store your dataset files here  
└── 📜 README.md             → All the documentation you need!  
```

---

## 📊 Dataset Information
The dataset must be in **CoNLL-U format** and should contain sentences with their corresponding POS tags.
- Each token in the dataset should have its **form (word)** and **UPOS (Universal POS Tag)**.
- Example format:

```
# sent_id = 1
# text = This is an example sentence.
1	This	_	DET	_	_	_	_	_	_
2	is	_	VERB	_	_	_	_	_	_
3	an	_	DET	_	_	_	_	_	_
4	example	_	NOUN	_	_	_	_	_	_
5	sentence	_	NOUN	_	_	_	_	_	_
6	.	_	PUNCT	_	_	_	_	_	_
```


## 📄 Download the Dataset
### 📥 You can find the actual dataset here:
### 🔗 Dataset Link

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
  git clone https://github.com/your-username/pos-tagging-finetune.git
  cd pos-tagging-finetune
```

### 2️⃣ Install Dependencies
```bash
  pip install -r requirements.txt
```

### 3️⃣ Run Fine-Tuning Scripts
#### 🏃‍♂️ **Standard Fine-Tuning**
```bash
python normal_finetune.py \
    --train_file ./dataset/train.conllu \
    --valid_file ./dataset/valid.conllu \
    --test_file ./dataset/test.conllu \
    --model_path ./path/to/pretrained/model \
    --output_dir ./path/to/output/directory \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5
```

#### 🔥 **Gradual Unfreezing Fine-Tuning**
```bash
python gradual_finetune.py \
    --train_file ./dataset/train.conllu \
    --valid_file ./dataset/valid.conllu \
    --test_file ./dataset/test.conllu \
    --model_path ./path/to/pretrained/model \
    --output_dir ./path/to/output/directory \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5
```

---

## 📌 Key Features
✅ Supports **POS tagging** using transformer-based models (e.g., BERT, XLM-R, MURIL)  
✅ Uses **CoNLL-U format datasets**  
✅ Implements **label alignment** for subword tokenization  
✅ **Gradual unfreezing** method to fine-tune large models efficiently  
✅ **Custom learning rates** for different model components  
✅ **Evaluation metrics** including accuracy & classification report  

---

## 📜 Requirements
The following packages are required:
```bash
transformers
numpy
scikit-learn
datasets
conllu
torch
argparse
```
Alternatively, install them all using:
```bash
pip install -r requirements.txt
```

Additionally, ensure you have:
- **Python 3.6+** 🐍
- **CUDA 12.1** (for GPU acceleration) ⚡

---

## 📊 Evaluation & Metrics
After training, the script will evaluate on the test set and display:
- **Accuracy** 📊
- **Macro & Weighted F1 Scores** 📈
- **Detailed Classification Report** 📄

---



🌟 *If you find this useful, don't forget to star ⭐ the repository!*

