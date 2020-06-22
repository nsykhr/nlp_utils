from overrides import overrides
from typing import Dict, Optional, List, Any

import torch
import torch.nn.functional
from torch.nn import Linear, CrossEntropyLoss, Dropout, Sequential

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder

from nlp_utils.modules import SoftmaxLoss, F1Measure


@Model.register('spellchecker')
class SpellChecker(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 spellchecker_namespace: str = 'target_tokens',
                 punct_namespace: str = 'punct_labels',
                 feedforward: Optional[FeedForward] = None,
                 punct_hidden: int = 256,
                 embedding_dropout: Optional[float] = None,
                 encoded_dropout: Optional[float] = None,
                 punct_dropout: Optional[float] = None,
                 punct_weight: Optional[Dict[str, float]] = None,
                 num_samples: Optional[int] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = punct_namespace
        self.text_field_embedder = text_field_embedder
        self.token_vocab_size = self.vocab.get_vocab_size(spellchecker_namespace)
        self.punct_vocab_size = self.vocab.get_vocab_size(punct_namespace)
        self.encoder = encoder
        self.embedding_dropout = Dropout(embedding_dropout) if embedding_dropout is not None else None
        self.encoded_dropout = Dropout(encoded_dropout) if encoded_dropout is not None else None
        self.feedforward = feedforward

        if feedforward is not None:
            self.output_dim = feedforward.get_output_dim()
        else:
            self.output_dim = self.encoder.get_output_dim()

        if punct_dropout is not None:
            self.punct_projection = Sequential(
                Linear(self.output_dim, punct_hidden),
                Dropout(punct_dropout),
                Linear(punct_hidden, self.punct_vocab_size)
            )
        else:
            self.punct_projection = Sequential(
                Linear(self.output_dim, punct_hidden),
                Linear(punct_hidden, self.punct_vocab_size)
            )

        self.losses = {
            'spellchecker': SoftmaxLoss(
                num_words=self.token_vocab_size,
                embedding_dim=self.output_dim + 1
            )
            if num_samples is None else SampledSoftmaxLoss(
                num_words=self.token_vocab_size,
                embedding_dim= self.output_dim + 1,
                num_samples=num_samples
            ),
            'punct': CrossEntropyLoss(weight=self.__get_weight_tensor(punct_weight), reduction='sum', ignore_index=-1)
        }
        self.add_module('spellchecker_loss', self.losses['spellchecker'])
        self.add_module('punct_loss', self.losses['punct'])

        self.metrics = {
            'punct_accuracy': CategoricalAccuracy()
        }
        self.metrics.update({
            f'f1_score_{name}': F1Measure(self.vocab.get_token_index(name, namespace=punct_namespace))
            for name in self.vocab.get_token_to_index_vocabulary(namespace=punct_namespace)
        })

        initializer(self)

    def __get_weight_tensor(self, weight: Dict[str, float]) -> torch.Tensor:
        vocab_size = self.vocab.get_vocab_size(self.label_namespace)

        if weight is None:
            return torch.ones(vocab_size)

        return torch.tensor([weight[label] for label in self.vocab.get_token_to_index_vocabulary(self.label_namespace)])

    def __compute_spellchecker_loss(self, embeddings: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        mask = targets > 0
        non_masked_targets = targets.masked_select(mask) - 1
        non_masked_embeddings = embeddings.masked_select(
            mask.unsqueeze(-1)
        ).view(-1, self.output_dim + 1)

        return self.losses['spellchecker'](non_masked_embeddings, non_masked_targets)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                token_lengths: torch.Tensor,
                target_tokens: torch.LongTensor = None,
                punct_labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedded_text_input = self.text_field_embedder(tokens)
        if self.embedding_dropout is not None:
            embedded_text_input = self.embedding_dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask.bool())
        if self.encoded_dropout is not None:
            encoded_text = self.encoded_dropout(encoded_text)
        if self.feedforward is not None:
            encoded_text = self.feedforward(encoded_text)

        punct_logits = self.punct_projection(encoded_text)
        token_lengths = token_lengths.unsqueeze(-1)
        encoded_text = torch.cat((encoded_text, token_lengths), dim=-1)

        output = {
            'mask': mask,
            'punct_logits': punct_logits,
            'embeddings': encoded_text
        }

        if target_tokens is not None:
            output['loss'] = self.__compute_spellchecker_loss(encoded_text, target_tokens)

        if punct_labels is not None:
            flipped_mask = (mask == 0)
            masked_punct_labels = punct_labels.masked_fill(flipped_mask, -1)
            punct_loss = self.losses['punct'](punct_logits.transpose(1, 2), masked_punct_labels)

            if 'loss' in output:
                output['loss'] += punct_loss
            else:
                output['loss'] = punct_loss

            for name, metric in self.metrics.items():
                metric(punct_logits, punct_labels, mask.float())

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

    @overrides
    def decode(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        W, b = self.losses['spellchecker'].softmax_w, self.losses['spellchecker'].softmax_b
        logits = output['embeddings'].matmul(W.transpose(0, 1)) + b
        output['spellchecker_logits'] = logits

        return output
