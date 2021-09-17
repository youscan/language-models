from transformers import PreTrainedTokenizerFast

from language_model.tokenization.factory import FAST_TOKENIZER_DEFAULT_FILE_NAME
from language_model.tokenization.tasks import PreTrainedTokenizerFastSavingTask

TOKENIZER_PATH = f"outputs/cyr/gpt/train_tokenizer/ukr-gpt/{FAST_TOKENIZER_DEFAULT_FILE_NAME}"

IN_HOUSE_TRAIN_DATA_PATH = "outputs/cyr/gpt/extract_texts/in-house-data/texts.txt"
OPEN_TRAIN_DATA_PATH = "outputs/cyr/gpt/extract_texts/train-validation-open-data/train_shuffled.txt"
MODEL_MAX_LENGTH = 1024


# tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH, model_max_length=MODEL_MAX_LENGTH, padding_side="right"
)
tokenizer.add_special_tokens({"bos_token": "<|endoftext|>"})
# basically `pad_token` wont be used for training, as DataCollatorForGroupTextForCasualLMDataset pack sequences up to
# max_length but to avoid an error within DataCollatorForGroupTextForCasualLMDataset
tokenizer.pad_token = tokenizer.bos_token

task = PreTrainedTokenizerFastSavingTask(pretrained_fast_tokenizer=tokenizer)
