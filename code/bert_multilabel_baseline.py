
import os
import pickle as pkl
import tqdm
import logging
import json

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from bert_multilabel_baseline_templates import templates
from hierarchy import map_to_max_level
from utils import get_metrics


def get_template_string(row, tokenizer, language):
    first = True

    row = {key: None if pd.isna(val) else str(val) for key, val in dict(row).items()}

    book, written = templates[language]['the book'], templates[language]['written']
    title, title_supplement = row['title'], row['title_supplement']
    start = f'{book} "{title}{": " + title_supplement if title_supplement else ""}" {written}'

    for key, data in templates[language].items():

        if key == 'the book' or key == 'written':
            continue

        temp, feats = data
        if not list(filter(None, (row[feat] for feat in feats))):
            continue

        if first:
            sep, first = ' ', False
        else:
            sep = ', '

        if isinstance(temp, str):
            start += sep + temp + ' ' + ' '.join(row[feat] for feat in feats)
        else: # it's fn
            start += sep + temp(*[row[feat] for feat in feats])

    return start + "."


def get_string(row, tokenizer, language):
    target = [
        'title', 'title_supplement', 'title_continuing_resource',
        # 'author_first_name', 'author_last_name', 
        # 'year',
        # 'place_publication', 
        # 'pages', 'editor_first_name', 'editor_last_name',
        'summary', 'work_info', 'work_title','expression_info',
        'expression_title', 'RVK_j', 'keyword', 'keyword_loc'
        ]

    return (" " + tokenizer.sep_token +  " ").join(str(row[f]) if not pd.isna(row[f]) else 'NA' for f in target)


class CustomDataset(Dataset):
    def __init__(self, df,  tokenizer, mlb, stringify_type='template', device='cpu'):
        self.df = df
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.stringify_type = stringify_type
        if stringify_type == 'template':
            self.stringify_fn = get_template_string
        else:
            self.stringify_fn = get_string
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = dict(self.df.iloc[idx])
        targets = torch.tensor(self.mlb.transform([row['BK_split']])[0], dtype=torch.float)
        language = row['language_text']
        text = self.stringify_fn(row, self.tokenizer, language)
        inputs = tokenizer.encode_plus(
            text, 
            None, 
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True)

        batch = {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'targets': targets}

        if self.device != 'cpu':
            batch = {key: val.to(self.device) for key, val in batch.items()}

        return batch


class Model(torch.nn.Module):
    def __init__(self, bert, n_labels, dropout=0.3):
        super(Model, self).__init__()

        self.bert = bert
        self.dropout = dropout
        self.output = torch.nn.Linear(bert.config.hidden_size, n_labels)

    def forward(self, batch):
        feats = self.bert(batch['ids'], batch['mask'], batch['token_type_ids'])
        feats = feats[1]
        feats = torch.nn.functional.dropout(feats, p=self.dropout, training=self.training)
        return self.output(feats)


def focal_loss(probs, targets, alpha=0.25, gamma=2.0):
    t1 = torch.pow(1 - probs, gamma) * torch.log(probs)
    t2 = torch.pow(probs, gamma) * torch.log(1 - probs)
    loss = -targets * t1 * alpha - (1 - targets) * t2 * (1 - alpha)
    return loss.mean()


def train_epochs(
        model, training_loader, dev_loader, epochs=3, lr=1e-5, 
        use_focal_loss=False, alpha=0.25, gamma=2.0,
        lr_gamma=1, lr_steps=(10, 15, 20),
        report_every=10, eval_every=5000, **meta):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_gamma != 1:
        scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

    run_eval = eval_every
    instances = total_loss = report_loss = 0
    
    for epoch in range(epochs):
        pbar = tqdm.tqdm(training_loader, total=len(training_loader))
        for b_id, batch in enumerate(pbar):
            probs = torch.sigmoid(model(batch))

            optimizer.zero_grad()
            targets = batch['targets']
            if use_focal_loss:
                loss = focal_loss(probs, targets, alpha=alpha, gamma=gamma)
            else:
                loss = torch.nn.functional.binary_cross_entropy(probs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            report_loss += loss.item()
            if b_id % report_every == 0:
                pbar.set_description(
                    "Epoch {}; batch {}; training loss {:g}".format(
                        epoch, b_id, report_loss / report_every))
                report_loss = 0

            instances += len(targets)
            if instances > run_eval:
                logging.log(logging.INFO, "Evaluation. Epoch: {}, Batch: {}".format(
                    epoch, 100 * (b_id / len(training_loader))))
                result = evaluate(model, dev_loader, alpha=alpha, gamma=gamma)
                yield dict(result, epoch=epoch, batch=b_id, **meta)
                model.train()
                run_eval += eval_every

        if eval_every > len(training_loader):
            logging.log(logging.INFO, "Evaluation. Epoch: {}".format(epoch))
            result = evaluate(model, dev_loader, alpha=alpha, gamma=gamma)
            yield dict(result, epoch=epoch, batch=b_id, **meta)
            model.train()

        if scheduler is not None:
            scheduler.step()


def evaluate(model, dev_loader, use_focal_loss=False, **kwargs):
    model.eval()

    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        pbar = tqdm.tqdm(dev_loader, total=len(dev_loader))
        for b_id, batch in enumerate(pbar):
            targets = batch['targets']
            probs = torch.sigmoid(model(batch))
            if use_focal_loss:
                total_loss += focal_loss(probs, targets, **kwargs).item()
            else:
                total_loss += torch.nn.functional.binary_cross_entropy(probs, targets).item()

            pbar.set_description(
                    "Evaluation. Batch {}; dev loss {:g}".format(
                        b_id, total_loss / (b_id + 1)))
            preds.extend(probs.cpu().detach().numpy().tolist())
            trues.extend(targets.cpu().detach().numpy().tolist())

    trues = np.array(trues)
    accs, ths = [], []
    for th in np.arange(0.05, 1, 0.05):
        accs.append(accuracy_score(trues, np.array(preds) >= th))
        ths.append(th)
    best_th = ths[np.argmax(accs)]
    preds = np.array(preds) >= best_th

    metrics = get_metrics(trues, preds)
    metrics['loss'] = total_loss / (b_id + 1)
    return metrics


def predict(model, loader):
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        pbar = tqdm.tqdm(loader, total=len(loader))
        for _, batch in enumerate(pbar):
            targets = batch['targets']
            probs = torch.sigmoid(model(batch))
            preds.extend(probs.cpu().detach().numpy().tolist())
            trues.extend(targets.cpu().detach().numpy().tolist())

    trues = np.array(trues)
    accs, ths = [], []
    for th in np.arange(0.05, 1, 0.05):
        accs.append(accuracy_score(trues, np.array(preds) >= th))
        ths.append(th)
    best_th = ths[np.argmax(accs)]
    preds = np.array(preds) >= best_th

    return preds


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data')
    parser.add_argument('--dev-data')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--stringify-type', default='template')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--max-level', type=int, default=0)
    parser.add_argument('--use-focal-loss', action='store_true')
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--lr-steps', nargs="+", default=[10, 15, 20], type=int)
    parser.add_argument('--lr-gamma', default=1, type=float)
    parser.add_argument('--label-hierarchy', default='./data/label-hierarchy.tsv')
    parser.add_argument('--output-path', default='./bert_multilabel.preds')
    args = parser.parse_args()

    # class Object():
    #     pass
    # args = Object()
    # args.label_hierarchy = '../data/label-hierarchy.tsv'
    # args.train_data = '../data/1980_2019_data.train.parquet'
    # args.dev_data = '../data/1980_2019_data.dev.parquet'
    # args.max_level = 2
    # args.max_samples = 20
    # args.batch_size = 10
    # args.device = "cpu"
    # args.stringify_type = "other"

    hierarchy = pd.read_csv(args.label_hierarchy, sep='\t')
    if args.train_data.endswith('.parquet'):
        train = pd.read_parquet(args.train_data)
    else:
        train = pd.read_csv(args.train_data)
    if args.dev_data.endswith('.parquet'):
        dev = pd.read_parquet(args.dev_data)
    else:
        dev = pd.read_csv(args.dev_data)

    # map labels
    if args.max_level > 0:
        mapping = map_to_max_level(hierarchy, args.max_level)
        train['BK_split'] = train['BK_split'].apply(lambda row: tuple(set(map(mapping.get, row))))
        dev['BK_split'] = dev['BK_split'].apply(lambda row: tuple(set(map(mapping.get, row))))

    # filter languages
    target_languages = train['language_text'].value_counts()[
        list(train['language_text'].value_counts() > 1000)
    ].keys()
    target_languages = [lang for lang in target_languages if '|' not in lang]
    train = train[train['language_text'].isin(target_languages)]
    dev = dev[dev['language_text'].isin(target_languages)]

    # subsample data
    if args.max_samples > 0:
        dev = dev.sample(n=args.max_samples, random_state=1001)
        train = train.sample(n=args.max_samples, random_state=1001)

    mlb = MultiLabelBinarizer()
    mlb.fit(train['BK_split'].tolist() + dev['BK_split'].tolist())
    logging.log(logging.INFO, "Training with {} labels".format(len(mlb.classes_)))

    # load model
    bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    training_data = CustomDataset(
        train, tokenizer, mlb, device=args.device, stringify_type=args.stringify_type)
    training_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

    dev_data = CustomDataset(
        dev, tokenizer, mlb, device=args.device, stringify_type=args.stringify_type)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size)

    model = Model(bert, len(mlb.classes_))
    model = torch.nn.DataParallel(model)
    model.to(args.device)

    for result in train_epochs(model, training_loader, dev_loader, **args.__dict__):
        print(json.dumps(result))

    preds = predict(model, dev_loader)
    with open(args.output_path, 'wb') as f:
        np.savez(f, preds=preds, index=dev.index)
    with open(args.output_path + '.label_mapping', 'wb') as f:
        pkl.dump(mlb, f)
