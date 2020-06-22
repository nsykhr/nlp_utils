{
  "dataset_reader": {
    "type": "spellchecker_conll",
    "max_sequence_length": 50,
    "lazy": true
  },
  "train_data_path": "Punct/data/full_dataset_punct.txt",
  "vocabulary": {
    "tokens_to_add": {
      "token_characters": ["<SOT>", "<EOT>"]
    },
    "min_count": {"target_tokens": 100},
    "max_vocab_size": {"target_tokens": 100000},
    "non_padded_namespaces": ["*labels", "*tags"]
  },
  "model": {
    "type": "spellchecker",
    "embedding_dropout": 0.125,
    "encoded_dropout": 0.25,
    "punct_dropout": 0.25,
    "num_samples": 32768,
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "token_characters": ["token_characters"]
      },
      "token_embedders": {
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 32
          },
          "encoder": {
            "type": "stacked_bidirectional_lstm",
            "input_size": 32,
            "hidden_size": 256,
            "num_layers": 1,
            "recurrent_dropout_probability": 0.125,
            "layer_dropout_probability": 0.125,
            "use_highway": false
          }
        }
      }
    },
    "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": 512,
        "hidden_size": 512,
        "num_layers": 2,
        "recurrent_dropout_probability": 0.25,
        "layer_dropout_probability": 0.25,
        "use_highway": true
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 96,
    "biggest_batch_first": true,
    "sorting_keys": [["target_tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 8e-4,
      "weight_decay": 1e-4
    },
    "num_epochs": 5,
    "num_serialized_models_to_keep": 5,
    "grad_norm": 5.0,
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "step",
      "step_size": 1,
      "gamma": 0.5
    },
    "should_log_learning_rate": true
  }
}
