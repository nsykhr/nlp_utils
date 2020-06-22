{
  "dataset_reader": {
    "type": "punctuation_conll",
    "token_indexers": {
      "tokens": {
        "type": "rubert-pretrained",
        "pretrained_model": std.extVar('RUBERT_DIR') + "/rubert_tokenizer.pickle",
        "use_starting_offsets": true,
        "truncate_long_sequences": false,
        "do_lowercase": false
      }
    }
  },
  "train_data_path": "Punct/data/full_train_punct.txt",
  "validation_data_path": "Punct/data/full_valid_punct.txt",
  "model": {
    "type": "basic_tagger",
    "dropout": 0.5,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "rubert-pretrained",
          "pretrained_model": std.extVar('RUBERT_DIR') + "/rubert_model.pickle",
          "top_layer_only": true
        },
      },
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"]
      },
      "allow_unmatched_keys": true
    },
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 768,
      "hidden_size": 512,
      "num_layers": 2,
      "recurrent_dropout_probability": 0.3,
      "layer_dropout_probability": 0.5,
      "use_highway": true
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 16,
    "sorting_keys": [["tokens", "num_tokens"]],
    "biggest_batch_first": true
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
      "weight_decay": 1e-4
    },
    "num_epochs": 10,
    "patience": 2,
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
