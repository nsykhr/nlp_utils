from overrides import overrides
from typing import Dict, Optional, List, Any

import torch
from torch.nn import Linear, CrossEntropyLoss, Dropout

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed


@Model.register('basic_tagger')
class BasicTagger(Model):
    """
    This is a sequence labeling model for single entity tags
    (it has no CRF, therefore will not work very well with IOB/BIOUL etc.).
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = 'labels',
                 feedforward: Optional[FeedForward] = None,
                 dropout: Optional[float] = None,
                 weight: Optional[Dict[str, float]] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self.dropout = Dropout(dropout) if dropout else None
        self.feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_tags))

        self.loss = CrossEntropyLoss(weight=self.__get_weight_tensor(weight), ignore_index=-1) \
            if weight is not None else CrossEntropyLoss(ignore_index=-1)

        self.metrics = {
            f'f1_score_{name}': F1Measure(self.vocab.get_token_index(name, namespace=label_namespace))
            for name in self.vocab.get_token_to_index_vocabulary(namespace=label_namespace)
        }
        self.metrics['accuracy'] = CategoricalAccuracy()

        initializer(self)

    def __get_weight_tensor(self, weight: Dict[str, float]) -> torch.Tensor:
        return torch.tensor([weight[label] for label in self.vocab.get_token_to_index_vocabulary(self.label_namespace)])

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self.dropout is not None:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout is not None:
            encoded_text = self.dropout(encoded_text)

        if self.feedforward is not None:
            encoded_text = self.feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)

        output = {'logits': logits, 'mask': mask}

        if labels is not None:
            flipped_mask = (mask == 0)
            masked_labels = labels.masked_fill(flipped_mask, -1)
            output['loss'] = self.loss(logits.transpose(1, 2), masked_labels)
            for name, metric in self.metrics.items():
                metric(logits, labels, mask.float())

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_values = {}
        for name, metric in self.metrics.items():
            if 'f1_score' in name:
                metric_values[name] = metric.get_metric(reset)[2]
                continue
            metric_values[name] = metric.get_metric(reset)

        return metric_values
