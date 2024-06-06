import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import faiss  # Core-Set 샘플링을 위해 사용
import matplotlib.pyplot as plt

# 데이터셋 로드
link = '/home1/mose1103/koactive/JM-BERT/'
data_pool = pd.read_csv(link + 'data/data_pool.csv')
start_data = pd.read_csv(link + 'data/start_data.csv')
test_df = pd.read_csv(link + 'data/test_df.csv')

# kcBERT 토크나이저 및 모델 초기화
num_labels = 5  # 실제 클래스 수로 설정
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
model = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', num_labels=num_labels)

# PyTorch 데이터셋 정의
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = [label - 1 for label in labels]  # 라벨 값 0부터 시작하도록 수정

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# 평가 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 학습 함수 정의
def train_model(train_texts, train_labels):
    train_dataset = TextDataset(train_texts, train_labels)
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
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer

# 초기 데이터로 모델 학습
initial_texts = start_data['content'].tolist()
initial_labels = start_data['score'].tolist()
trainer = train_model(initial_texts, initial_labels)

# 테스트 데이터셋 생성
test_texts = test_df['content'].tolist()
test_labels = test_df['score'].tolist()
test_dataset = TextDataset(test_texts, test_labels)

# 초기 평가
initial_results = trainer.evaluate(test_dataset)

# 결과 저장
results = {'random': [], 'entropy': [], 'core-set': [], 'hybrid': []}
results['random'].append(initial_results['f1'])
results['entropy'].append(initial_results['f1'])
results['core-set'].append(initial_results['f1'])
results['hybrid'].append(initial_results['f1'])

# 액티브 러닝 샘플링 함수 정의 (엔트로피, Core-Set, 하이브리드 모델, 랜덤)
def active_learning_sample(data_pool, strategy, sample_size, trainer):
    if strategy == 'random':
        indices = random.sample(range(len(data_pool)), sample_size)
    elif strategy == 'entropy':
        texts = data_pool['content'].tolist()
        dataset = TextDataset(texts, [0]*len(texts))  # 임시 라벨
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        entropies = []

        for batch in dataloader:
            with torch.no_grad():
                outputs = trainer.model(**batch)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                entropies.extend(entropy.cpu().numpy())

        indices = np.argsort(entropies)[-sample_size:].tolist()
    elif strategy == 'core-set':
        texts = data_pool['content'].tolist()
        dataset = TextDataset(texts, [0]*len(texts))  # 임시 라벨
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        embeddings = []

        for batch in dataloader:
            with torch.no_grad():
                outputs = trainer.model.bert(**{k: v for k, v in batch.items() if k != 'labels'})
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        kmeans = faiss.Kmeans(d=embeddings.shape[1], k=sample_size, gpu=True)
        kmeans.train(embeddings)
        _, indices = kmeans.index.search(embeddings, 1)
        indices = indices.squeeze().tolist()
    elif strategy == 'hybrid':
        texts = data_pool['content'].tolist()
        dataset = TextDataset(texts, [0]*len(texts))  # 임시 라벨
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        entropies = []
        embeddings = []

        for batch in dataloader:
            with torch.no_grad():
                outputs = trainer.model(**{k: v for k, v in batch.items() if k != 'labels'})
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                entropies.extend(entropy.cpu().numpy())

                outputs = trainer.model.bert(**{k: v for k, v in batch.items() if k != 'labels'})
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

        entropies = np.array(entropies)
        entropy_indices = np.argsort(entropies)[-sample_size//2:].tolist()

        embeddings = np.concatenate(embeddings, axis=0)
        kmeans = faiss.Kmeans(d=embeddings.shape[1], k=sample_size//2, gpu=True)
        kmeans.train(embeddings)
        _, core_set_indices = kmeans.index.search(embeddings, 1)
        core_set_indices = core_set_indices.squeeze().tolist()

        indices = list
