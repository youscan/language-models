import os

from transformers import PreTrainedTokenizerFast

from language_model.data.extract import LineByLineSource, ExtractVectorsFromTexts

TOKENIZER_PATH = "outputs/cyr/gpt/train_tokenizer/convert-to-transformers/tokenizer/"

OPEN_VALIDATION_DATA_PATH = "outputs/cyr/gpt/extract_texts/train-validation-open-data/validation.txt"
MODEL_MAX_LENGTH = 1024

# data
validation_data_source = LineByLineSource(OPEN_VALIDATION_DATA_PATH)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

task = ExtractVectorsFromTexts(
    data_source=validation_data_source,
    tokenizer=PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH),
    block_size=MODEL_MAX_LENGTH,
    process_batch_size=100_000,
    workers=18
)
