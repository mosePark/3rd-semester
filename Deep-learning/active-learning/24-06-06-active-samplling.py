'''
불확실성 샘플링
'''

import numpy as np
import torch
import torch.nn.functional as F

def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)

def uncertainty_sampling(model, unlabeled_data):
    model.eval()
    with torch.no_grad():
        outputs = model(unlabeled_data)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        uncertainties = entropy(probs)
    return uncertainties.argsort()[-10:]  # 상위 10개 샘플 선택

'''
다양성 샘플링
'''

from sklearn.metrics.pairwise import euclidean_distances

def core_set_sampling(embedding_model, labeled_data, unlabeled_data):
    labeled_embeddings = embedding_model(labeled_data).cpu().numpy()
    unlabeled_embeddings = embedding_model(unlabeled_data).cpu().numpy()
    distances = euclidean_distances(unlabeled_embeddings, labeled_embeddings)
    min_distances = distances.min(axis=1)
    return min_distances.argsort()[-10:]  # 상위 10개 샘플 선택

'''
하이브리드 샘플링
'''

def hybrid_sampling(model, embedding_model, labeled_data, unlabeled_data):
    uncertainties = uncertainty_sampling(model, unlabeled_data)
    diversity = core_set_sampling(embedding_model, labeled_data, unlabeled_data)
    combined_scores = uncertainties + diversity
    return combined_scores.argsort()[-10:]  # 상위 10개 샘플 선택


'''
랜덤 샘플링
'''

def random_sampling(unlabeled_data):
    indices = np.random.choice(len(unlabeled_data), 10, replace=False)
    return indices
