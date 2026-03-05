from transformers import LiltModel
import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from transformers.utils import ModelOutput
import requests
import json
from collections import OrderedDict
import flwr as fl
from flwr.server.strategy import FedAvg
from evaluation import re_score
from transformers.trainer_utils import EvalPrediction


class BiaffineAttention(torch.nn.Module):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


class REDecoder(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.entity_emb = nn.Embedding(3, input_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(input_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()

    def build_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if entities[b]["label"][i] == 1 and entities[b]["label"][j] == 2
                ]
            )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations)
            )
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        all_logits = []
        all_labels = []

        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_labels = torch.tensor(entities[b]["label"], device=device)
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            head_repr = torch.cat(
                (hidden_states[b][head_index], head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (hidden_states[b][tail_index], tail_label_repr),
                dim=-1,
            )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
            all_logits.append(logits)
            all_labels.append(relation_labels)
        all_logits = torch.cat(all_logits, 0)
        all_labels = torch.cat(all_labels, 0)
        loss = self.loss_fct(all_logits, all_labels)
        return loss, all_pred_relations


@dataclass
class ReOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    entities: Optional[Dict] = None
    relations: Optional[Dict] = None
    pred_relations: Optional[Dict] = None

class REHead(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.extractor = REDecoder(config, config.hidden_size)

  def forward(self,sequence_output, entities, relations):
    sequence_output = self.dropout(sequence_output)
    loss, pred_relations = self.extractor(sequence_output, entities, relations)
    return ReOutput(
            loss=loss,
            entities=entities,
            relations=relations,
            pred_relations=pred_relations,
        )
  
model_name = 'D:\FYP\lilt-app\models\lilt-base' #'kavg/layoutxlm-finetuned-xfund-fr-re'
model = LiltModel.from_pretrained(model_name) #nielsr/lilt-xlm-roberta-base
rehead = REHead(model.config)
del(model)
model = rehead

with open('D:\FYP\FL\client\download.png', 'rb') as f:
    file = f.read()
base_url = 'http://127.0.0.1:8000/'
ser_response = requests.post(base_url+'label-tokens', files=dict(file=file))

re_response = requests.post(base_url+'extract-relations', json=ser_response.json())
re_response = re_response.json()
sequence_output = json.loads(re_response['sequence_output'])
pred_relations = json.loads(re_response['pred_relations'])
input_ids = json.loads(re_response['input_ids'])
entities = json.loads(re_response['entities'])

input_ids = torch.tensor(input_ids)
sequence_output =  torch.tensor(sequence_output)

actual_relations = {'head': [0, 0, 0, 0, 5, 6, 7, 9, 11, 13, 14],
 'tail': [4, 1, 2, 3, 16, 17, 8, 10, 12, 18, 15],
 'start_index': [33, 33, 33, 33, 441, 443, 445, 450, 453, 485, 489],
 'end_index': [99, 63, 78, 95, 499, 500, 450, 453, 485, 505, 498]}

entity_dict = {'start': [entity[0] for entity in entities], 'end': [entity[1] for entity in entities], 'label': [entity[3] for entity in entities]}
inputs = dict(sequence_output=sequence_output, entities=[entity_dict], relations=[actual_relations])

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics(p):
    pred_relations, gt_relations = p
    score = re_score(pred_relations, gt_relations, mode="boundaries")
    return score

def train(model, inputs, epochs=2):
  model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=5-5)
  model.train()
  for epoch in range(epochs):
    # zero the parameter gradients
    optimizer.zero_grad()

    inputs['sequence_output'] =  inputs['sequence_output'].to(device)

    outputs = model(**inputs)

    loss = outputs.loss
    loss.backward()

    optimizer.step()

def test(model, input):
    metric_key_prefix = 'eval'
    loss = 0
    model.eval()
    pred_relations = None
    entities = None
    re_labels = None
    label_names = ['labels', 'relations']

    input['sequence_output'] =  input['sequence_output'].to(device)
    with torch.no_grad():
        outputs = model(**input)
    labels = tuple(input.get(name) for name in label_names)
    re_labels = labels[1]
    pred_relations = outputs.pred_relations
    entities = outputs.entities

    gt_relations = []
    for b in range(len(re_labels)):
        rel_sent = []
        for head, tail in zip(re_labels[b]["head"], re_labels[b]["tail"]):
            try:
                rel = {}
                rel["head_id"] = head
                rel["head"] = (entities[b]["start"][rel["head_id"]], entities[b]["end"][rel["head_id"]])
                rel["head_type"] = entities[b]["label"][rel["head_id"]]

                rel["tail_id"] = tail
                rel["tail"] = (entities[b]["start"][rel["tail_id"]], entities[b]["end"][rel["tail_id"]])
                rel["tail_type"] = entities[b]["label"][rel["tail_id"]]

                rel["type"] = 1

                rel_sent.append(rel)
            except:
                pass

        gt_relations.append(rel_sent)

    re_metrics = compute_metrics(EvalPrediction(predictions=pred_relations, label_ids=gt_relations))

    re_metrics = {
        "precision": re_metrics["ALL"]["p"],
        "recall": re_metrics["ALL"]["r"],
        "f1": re_metrics["ALL"]["f1"],
    }
    re_metrics[f"{metric_key_prefix}_loss"] = outputs.loss.mean().item()

    metrics = {}

    # # Prefix all keys with metric_key_prefix + '_'
    for key in list(re_metrics.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            metrics[f"{metric_key_prefix}_{key}"] = re_metrics.pop(key)
        else:
            metrics[f"{key}"] = re_metrics.pop(key)

    return metrics[metric_key_prefix+'_loss'], metrics[metric_key_prefix+'_f1']


class LiltClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]
        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(model, inputs, epochs=100)
            print("Training Finished.")
            return self.get_parameters(config={}), len(inputs), {}
        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, f1 = test(model, inputs)
            return float(loss), 1, {"f1": float(f1)}
        
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=LiltClient())

# def client_fn(cid: str):
#     # Return a standard Flower client
#     return LiltClient()

# # Launch the simulation
# hist = fl.simulation.start_simulation(
#     client_fn=client_fn, # A function to run a _virtual_ client when required
#     num_clients=2, # Total number of clients available
#     config=fl.server.ServerConfig(num_rounds=3), # Specify number of FL rounds
#     strategy=FedAvg(), # A Flower strategy
# )