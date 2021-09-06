from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    IntervalStrategy,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from language_model.data.dataset import DataCollatorForGroupTextForCasualLMDataset, GroupTextForCasualLMDataset
from language_model.data.extract import LineByLineSource, ShuffledSources
from language_model.modelling.trainer import TransformersTrainTaskWithTokenizerSaving
from language_model.tokenization.factory import FAST_TOKENIZER_DEFAULT_FILE_NAME

TOKENIZER_PATH = f"outputs/cyr/gpt/train_tokenizer/ukr-gpt/{FAST_TOKENIZER_DEFAULT_FILE_NAME}"

IN_HOUSE_TRAIN_DATA_PATH = "outputs/cyr/gpt/extract_texts/in-house-data/texts.txt"
OPEN_TRAIN_DATA_PATH = "outputs/cyr/gpt/extract_texts/train-validation-open-data/train_shuffled.txt"
VALIDATION_DATA_PATH = "outputs/cyr/gpt/extract_texts/train-validation-open-data/validation.txt"
MODEL_MAX_LENGTH = 1024


# tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH, model_max_length=MODEL_MAX_LENGTH, padding_side="right"
)
tokenizer.add_special_tokens({"bos_token": "<|endoftext|>"})
# basically `pad_token` wont be used, as DataCollatorForGroupTextForCasualLMDataset pack sequences up to max_length
# but to avoid an error within DataCollatorForGroupTextForCasualLMDataset
tokenizer.pad_token = tokenizer.bos_token

# model
model_config = GPT2Config(vocab_size=len(tokenizer), bos_token_id=tokenizer.bos_token_id)
model = GPT2LMHeadModel(model_config)


# data
train_data_source = ShuffledSources(
    (text for text in LineByLineSource(IN_HOUSE_TRAIN_DATA_PATH)),
    (text for text in LineByLineSource(OPEN_TRAIN_DATA_PATH)),
)
validation_data_path = LineByLineSource(VALIDATION_DATA_PATH)

train_dataset = GroupTextForCasualLMDataset(
    tokenizer=tokenizer, data_source=train_data_source, block_size=MODEL_MAX_LENGTH
)
valid_dataset = GroupTextForCasualLMDataset(
    tokenizer=tokenizer, data_source=validation_data_path, block_size=MODEL_MAX_LENGTH
)
data_collator = DataCollatorForGroupTextForCasualLMDataset()


training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=250000,
    num_train_epochs=5,
    per_device_train_batch_size=8,  # overall bs = 8 * 8 * num_gpus (GPT2 used 512)
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    output_dir="temp",
    overwrite_output_dir=True,
    save_steps=250000,
    save_total_limit=2,
    prediction_loss_only=False,
    learning_rate=0.0002,  # (was manually tuned in GPT2 on held-out validation)
    warmup_ratio=0.004,
    fp16=True,
    logging_dir="logs",
    seed=42,
    lr_scheduler_type="cosine",  # type: ignore
    logging_first_step=True,
    logging_steps=500,
    label_names=["labels"],
    load_best_model_at_end=True,
    group_by_length=False,
    report_to=["mlflow"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

task = TransformersTrainTaskWithTokenizerSaving(trainer=trainer)
