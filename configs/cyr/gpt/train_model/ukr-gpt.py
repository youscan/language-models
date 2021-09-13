from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    IntervalStrategy,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from language_model.data.dataset import DataCollatorForGroupTextForCasualLMDataset, FromInputIdsDataset
from language_model.modelling.trainer import TransformersTrainTask

TOKENIZER_PATH = "outputs/cyr/gpt/train_tokenizer/convert-to-transformers/tokenizer/"

TRAIN_IDS_PATH = "outputs/cyr/gpt/extract_vectors/vectorize-train/processed_batch.jsonl"
VALIDATION_IDS_PATH = "outputs/cyr/gpt/extract_vectors/vectorize-validation/processed_batch.jsonl"
MODEL_MAX_LENGTH = 1024


# tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
# model
model_config = GPT2Config(vocab_size=len(tokenizer), bos_token_id=tokenizer.bos_token_id)
model = GPT2LMHeadModel(model_config)


# data
train_dataset = FromInputIdsDataset(TRAIN_IDS_PATH)
valid_dataset = FromInputIdsDataset(VALIDATION_IDS_PATH)
data_collator = DataCollatorForGroupTextForCasualLMDataset(MODEL_MAX_LENGTH)


training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=250000,
    num_train_epochs=5,
    per_device_train_batch_size=4,  # overall bs = 4 * 16 * num_gpus (GPT2 used 512)
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=4,
    output_dir="checkpoints",
    overwrite_output_dir=False,
    save_steps=250000,
    save_total_limit=10,
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
    dataloader_num_workers=1,  # because of IterableDataset that reads from one opened file
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

task = TransformersTrainTask(trainer=trainer)
