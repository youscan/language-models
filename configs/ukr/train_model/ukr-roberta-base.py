from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer

from language_model.modelling.trainer import RobertaForMaskedLMTrainTask

_model_config = RobertaConfig(
    vocab_size=52000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
    intermediate_size=3072,
)

_model = RobertaForMaskedLM(_model_config)

_tokenizer = RobertaTokenizer.from_pretrained("outputs/ukr/train_tokenizer/ukr-roberta-base/tokenizer", max_len=512)

task = RobertaForMaskedLMTrainTask(
    file_path="data/ukr/aggregated_data/ukr-roberta-base/data.txt",
    model=_model,
    tokenizer=_tokenizer,
    batch_size_per_gpu=40,
)
