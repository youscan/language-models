from transformers import (
    AutoModelForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    IntervalStrategy,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from language_model.data.dataset import LineByLineTextDataset, split_lazy_dataset
from language_model.modelling.trainer import TransformersTrainTask
from language_model.tokenization.factory import FAST_TOKENIZER_DEFAULT_FILE_NAME

LOGGING_FORMAT: str = "%(asctime)s : %(levelname)s : %(module)s : %(message)s"

TRANSFORMER_MODEL_NAME = "DeepPavlov/rubert-base-cased-conversational"
TOKENIZER_PATH = (
    f"outputs/cyr/adapt_rubert-base-cased-conversational/tokenizer/extend_base_vocab=96k"
    f"/{FAST_TOKENIZER_DEFAULT_FILE_NAME}"
)
TEXT_FILE_PATHS = [
    "data/cyr/adapt_rubert-base-cased-conversational/extract_texts/test.txt",
    "data/cyr/adapt_rubert-base-cased-conversational/extract_texts/test2.txt",
    "data/cyr/adapt_rubert-base-cased-conversational/extract_texts/test3.txt",
]
TRAIN_TEST_PORTIONS = [0.95, 0.05]

tokenizer = BertTokenizerFast(
    vocab_file=None,
    tokenizer_file=TOKENIZER_PATH,
    from_slow=False,
    do_lower_case=False,
    strip_accents=False,
    tokenize_chinese_chars=False,
)
model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=TRANSFORMER_MODEL_NAME)
model.resize_token_embeddings(len(tokenizer.get_vocab()))

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_paths=TEXT_FILE_PATHS, block_size=128)
train_set, eval_set = split_lazy_dataset(dataset, portions=TRAIN_TEST_PORTIONS)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="temp",
    do_train=True,
    do_eval=True,
    evaluation_strategy=IntervalStrategy.EPOCH,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=False,
    learning_rate=5e-5,
    logging_dir="logs",
    seed=42,
)

trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=train_set, eval_dataset=eval_set
)

task = TransformersTrainTask(trainer=trainer)
