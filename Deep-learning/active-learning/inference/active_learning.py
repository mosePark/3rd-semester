import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import cvxpy as cp
import pandas as pd
import os
from model_training import train_model, evaluate_model, TextDataset
from utils import compute_metrics

def active_learning_sample(data_pool, strategy, sample_size, trainer, lambda_reg=1.0):
    if strategy == 'random':
        indices = random.sample(range(len(data_pool)), sample_size)
    elif strategy == 'entropy':
        texts = data_pool['content'].tolist()
        dataset = TextDataset(texts, [0]*len(texts))  # 임시 라벨
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        entropies = []

        for batch in dataloader:
            with torch.no_grad():
                with autocast():
                    outputs = trainer.model(**batch)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                    entropies.extend(entropy.cpu().numpy())

        indices = np.argsort(entropies)[-sample_size:].tolist()
        
    elif strategy == 'imbalance_entropy':
        texts = data_pool['content'].tolist()
        dataset = TextDataset(texts, [0]*len(texts))  # 임시 라벨
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        entropies = []
        probs_list = []

        for batch in dataloader:
            with torch.no_grad():
                with autocast():
                    outputs = trainer.model(**batch)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                    entropies.extend(entropy.cpu().numpy())
                    probs_list.extend(probs.cpu().numpy())

        probs_array = np.array(probs_list)
        P = torch.tensor(probs_array, dtype=torch.float32)
        N, C = P.shape
        z = cp.Variable(N, boolean=True)
        omega = np.ones(C) * sample_size / C
        entropy_term = -torch.sum(P * torch.log(P + 1e-12), axis=1).numpy()
        objective = cp.Minimize(z.T @ entropy_term + lambda_reg * cp.norm(P.T @ z - omega, 1))
        constraints = [cp.sum(z) == sample_size]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        indices = np.where(z.value > 0.5)[0]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return indices

def active_learning(data_pool, start_data, test_df):
    results = {'random': [], 'entropy': [], 'imbalance_entropy': []}
    results_details = {'random': [], 'entropy': [], 'imbalance_entropy': []}
    class_distribution = {'random': [], 'entropy': [], 'imbalance_entropy': []}

    initial_texts = start_data['content'].tolist()
    initial_labels = start_data['score'].tolist()
    eval_texts = test_df['content'].tolist()
    eval_labels = test_df['score'].tolist()

    trainer = train_model(initial_texts, initial_labels, eval_texts, eval_labels)
    initial_results = evaluate_model(trainer, test_df)

    results['random'].append(initial_results['eval_f1'])
    results['entropy'].append(initial_results['eval_f1'])
    results['imbalance_entropy'].append(initial_results['eval_f1'])

    results_details['random'].append(initial_results)
    results_details['entropy'].append(initial_results)
    results_details['imbalance_entropy'].append(initial_results)

    for strategy in ['random', 'entropy', 'imbalance_entropy']:
        print(strategy)
        for i in range(5):
            print(i)
            indices = active_learning_sample(data_pool, strategy, 250, trainer, lambda_reg=0.5)
            sampled_texts = [data_pool['content'].iloc[i] for i in indices]
            sampled_labels = [data_pool['score'].iloc[i] for i in indices]

            initial_texts.extend(sampled_texts)
            initial_labels.extend(sampled_labels)
            trainer = train_model(initial_texts, initial_labels, eval_texts, eval_labels)

            sampled_class_distribution = pd.Series(sampled_labels).value_counts().to_dict()
            class_distribution[strategy].append(sampled_class_distribution)

            result = evaluate_model(trainer, test_df)
            results[strategy].append(result['eval_f1'])
            results_details[strategy].append(result)
            print(results)
            print(f"Cycle {i} sampled class distribution for {strategy}: {sampled_class_distribution}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join('/home1/mose1103/koactive/JM-BERT/inference/results', 'active_learning_results.csv'), index=False)

    results_details_df = pd.DataFrame({(k1, k2): v2 for k1, v1 in results_details.items() for k2, v2 in enumerate(v1)})
    results_details_df.to_csv(os.path.join('/home1/mose1103/koactive/JM-BERT/inference/results', 'active_learning_results_details.csv'), index=False)

    class_distribution_df = pd.DataFrame({(k1, k2): v2 for k1, v1 in class_distribution.items() for k2, v2 in enumerate(v1)})
    class_distribution_df.to_csv(os.path.join('/home1/mose1103/koactive/JM-BERT/inference/results', 'class_distribution.csv'), index=False)
