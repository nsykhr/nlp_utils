from overrides import overrides
from typing import List, Tuple, Iterator, Dict, Optional

from allennlp.data import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


@DatasetReader.register('punctuation_conll')
class PunctuationCONLLDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 lazy: bool = True):
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, tokens: List[Token], labels: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self._token_indexers)
        fields = {'tokens': sentence_field}

        if labels is not None:
            labels_field = SequenceLabelField(labels=labels, sequence_field=sentence_field)
            fields['labels'] = labels_field

        return Instance(fields)

    @staticmethod
    def read_file(file_path: str) -> Iterator[Tuple[List[Token], List[str]]]:
        with open(file_path, 'r', encoding='cp1251') as f:
            tokens, labels = [], []
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    # We are filtering out sentences that are too short or
                    # end with punctuation marks that actual sentences cannot end with.
                    if len(tokens) >= 3 and labels[-1] in ('_', 'FULL_STOP'):
                        yield list(map(Token, tokens)), labels

                    tokens, labels = [], []
                    continue

                token, label = line_stripped.split('\t')
                tokens.append(token)
                labels.append(label)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        for tokens, labels in self.read_file(file_path):
            yield self.text_to_instance(tokens, labels)
