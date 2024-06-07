import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AdamW


from torch.utils.data import DataLoader, Dataset
import torch

import random
import faiss  # Core-Set 샘플링을 위해 사용

# 데이터셋 로드
link = '/home1/mose1103/koactive/JM-BERT/'
data_pool = pd.read_csv(link + 'data/data_pool.csv')
start_data = pd.read_csv(link + 'data/start_data.csv')
test_df = pd.read_csv(link + 'data/test_df.csv')

# 레이블이 1에서 5사이로 되어 있을 때, 이를 0에서 4사이로 변환
data_pool['score'] -= 1
start_data['score'] -= 1
test_df['score'] -= 1

# kcBERT 토크나이저 및 모델 초기화
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
model = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', num_labels=5)

# PyTorch 데이터셋 정의
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

# 초기 데이터로 모델 학습
initial_texts = start_data['content'].tolist()
initial_labels = start_data['score'].tolist()
eval_texts = test_df['content'].tolist()
eval_labels = test_df['score'].tolist()

trainer = train_model(initial_texts, initial_labels, eval_texts, eval_labels)

# 테스트 데이터셋 생성
test_texts = test_df['content'].tolist()
test_labels = test_df['score'].tolist()
test_dataset = TextDataset(test_texts, test_labels)

# 초기 평가
initial_results = trainer.evaluate(test_dataset)

# 결과 저장
results = {'hybrid': []}
results['hybrid'].append(initial_results['eval_f1'])

# 평가 결과를 CSV 파일로 저장하는 함수
def save_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

# 액티브 러닝 샘플링 함수 정의 (Hybrid)
def active_learning_sample(data_pool, strategy, sample_size, trainer):
    if strategy == 'hybrid':
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

        indices = list(set(entropy_indices + core_set_indices))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return indices

# 액티브 러닝을 통한 반복 학습
initial_texts = start_data['content'].tolist()
initial_labels = start_data['score'].tolist()
print('hybrid')
for i in range(5):
    print(i)
    # 샘플링
    pool_texts = data_pool['content'].tolist()
    pool_labels = data_pool['score'].tolist()
    
    indices = active_learning_sample(data_pool, 'hybrid', 25, trainer)
    sampled_texts = [pool_texts[i] for i in indices]
    sampled_labels = [pool_labels[i] for i in indices]
    
    # 샘플 추가 및 모델 재학습
    initial_texts.extend(sampled_texts)
    initial_labels.extend(sampled_labels)
    trainer = train_model(initial_texts, initial_labels, eval_texts, eval_labels)
    
    # 평가 및 결과 저장
    result = trainer.evaluate(test_dataset)
    results['hybrid'].append(result['eval_f1'])
    print(results)

    # 평가 결과를 CSV 파일로 저장
    save_results_to_csv(results, link + '/hybrid_results.csv')
