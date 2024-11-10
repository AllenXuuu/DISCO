import random
import time
import torch
from torch import nn
import os
import argparse
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('-lr','--learning_rate', type=float, help="learning rate")
parser.add_argument('-model_type','--model_type', type=str, default='bert-base', help="one of roberta-large, roberta-base, bert-base, bert-large")
parser.add_argument('-task_desc_only','--task_desc_only', action='store_true', help="one of roberta-large, roberta-base, bert-base, bert-large")
parser.add_argument('--append', action='store_true')
parser.add_argument('--name', type=str)
parser.add_argument('--bz', default=100, type=int)
parser.add_argument('--epoch', default=200, type=int)
args = parser.parse_args()

device = torch.device('cuda')
root = f'logs/{args.name}'
os.makedirs(root, exist_ok=True)

post = '_append' if args.append else ''

train_set       = pkl.load(open(f'data/instruction/train_instruction{post}.pkl', 'rb'))
val_set_seen    = pkl.load(open(f'data/instruction/valid_seen_instruction{post}.pkl', 'rb'))
val_set_unseen  = pkl.load(open(f'data/instruction/valid_unseen_instruction{post}.pkl', 'rb'))


train_x, train_y = train_set['x'], train_set['y'] 
vs_x, vs_y       = val_set_seen['x'], val_set_seen['y']
vu_x, vu_y       = val_set_unseen['x'], val_set_unseen['y']

if args.model_type in ['bert-base', 'bert-large'] : 
    from transformers import BertTokenizer as Tokenizer
    tok_type = args.model_type + '-uncased'
    from transformers import BertForSequenceClassification as BertModel
elif args.model_type in ['roberta-base', 'roberta-large']:
    from transformers import RobertaTokenizer as Tokenizer
    tok_type = args.model_type
    from transformers import RobertaForSequenceClassification as BertModel
tokenizer = Tokenizer.from_pretrained(tok_type)
model = BertModel.from_pretrained(tok_type, num_labels=7).to(device)
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

train_x = tokenizer(train_x, return_tensors='pt', padding=True, truncation=True)
train_y = torch.tensor(train_y)

vs_x = tokenizer(vs_x, return_tensors='pt', padding=True, truncation=True)
vs_y = torch.tensor(vs_y)

vu_x = tokenizer(vu_x, return_tensors='pt', padding=True, truncation=True)
vu_y = torch.tensor(vu_y)



model.train()
from transformers import AdamW


def accuracy(y_pred, y_batch):
    #y_pred has shape [batch, no_classes]
    maxed = torch.max(y_pred, 1)
    y_hat = maxed.indices
    num_accurate = torch.sum((y_hat == y_batch).float())
    train_accuracy = num_accurate/ y_hat.shape[0]
    return train_accuracy.item()

def accurate_total(y_pred, y_batch):
    #y_pred has shape [batch, no_classes]
    maxed = torch.max(y_pred, 1)
    y_hat = maxed.indices
    num_accurate = torch.sum((y_hat == y_batch).float())
    return num_accurate

start_train_time = time.time()
for t in range(args.epoch):
    model.train()

    for i in range(0, len(train_x['input_ids']), args.bz):
        outputs = model(
            train_x['input_ids'].to(device),
            attention_mask=train_x['attention_mask'].to(device), 
            labels=train_y.to(device))
        optimizer.zero_grad()
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    if t % 1 ==0:
        with torch.no_grad():
            model.eval()

            for x,y in [(vs_x, vs_y), (vu_x, vu_y)]:
                logits = []
                for i in range(0, len(x['input_ids']), args.bz):
                    outputs = model(
                        x['input_ids'].to(device),
                        attention_mask=x['attention_mask'].to(device))
                    logits.append(outputs.logits.cpu())
                logits = torch.cat(logits)
                print("acc: ", accuracy(y_pred=logits, y_batch=y))

        path = os.path.join(root, f'epoch_{t:05d}.pth')
        model_name = 'epoch_' + str(t) + '.pt'
        torch.save({'model' : model.state_dict(), 'optim' : optimizer.state_dict()}, path)