import os

from transformers import PreTrainedTokenizerFast

from language_model.data.extract import ExtractVectorsFromTexts, LineByLineSource, ShuffledSources

TOKENIZER_PATH = "outputs/cyr/gpt/train_tokenizer/convert-to-transformers/tokenizer/"

IN_HOUSE_TRAIN_DATA_PATH = "outputs/cyr/gpt/extract_texts/in-house-data/texts.txt"
OPEN_TRAIN_DATA_PATH = "outputs/cyr/gpt/extract_texts/train-validation-open-data/train_shuffled.txt"
MODEL_MAX_LENGTH = 1024

# data
train_data_source = ShuffledSources(
    (text for text in LineByLineSource(IN_HOUSE_TRAIN_DATA_PATH)),
    (text for text in LineByLineSource(OPEN_TRAIN_DATA_PATH)),
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

task = ExtractVectorsFromTexts(
    data_source=train_data_source,
    tokenizer=PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH),
    block_size=MODEL_MAX_LENGTH,
    process_batch_size=100_000,
    workers=18,
)
