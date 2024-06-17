import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from utils import compute_metrics

# kcBERT 토크나이저 및 모델 초기화
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
model = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', num_labels=5).to('cuda')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def train_model(train_texts, train_labels, eval_texts, eval_labels):
    train_dataset = TextDataset(train_texts, train_labels)
    eval_dataset = TextDataset(eval_texts, eval_labels)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer

def evaluate_model(trainer, test_df):
    test_texts = test_df['content'].tolist()
    test_labels = test_df['score'].tolist()
    test_dataset = TextDataset(test_texts, test_labels)
    return trainer.evaluate(test_dataset)
