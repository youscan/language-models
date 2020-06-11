from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast

from src.configs import TrainModelConfig

_model_config = RobertaConfig(
    vocab_size=32000, max_position_embeddings=514, num_attention_heads=12, num_hidden_layers=6, type_vocab_size=1
)

_model = RobertaForMaskedLM(_model_config)

_tokenizer = RobertaTokenizerFast.from_pretrained("results/version_1/ukr/train_tokenizer/baseline", max_len=512)

config = TrainModelConfig(
    file_path="data/version_1/ukr/prepare_one_line_text_format/sampled_mentions/data.txt",
    model=_model,
    tokenizer=_tokenizer,
)
