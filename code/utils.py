
from sklearn import metrics

def get_metrics(trues, preds):
    return {'accuracy': metrics.accuracy_score(trues, preds),
            'f1-micro': metrics.f1_score(trues, preds, average='micro'),
            'f1-macro': metrics.f1_score(trues, preds, average='macro')}