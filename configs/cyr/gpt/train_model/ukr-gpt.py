from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    IntervalStrategy,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from language_model.data.dataset import DataCollatorForGroupTextForCasualLMDataset, GroupTextForCasualLMDataset
from language_model.modelling.trainer import TransformersTrainTask
from language_model.tokenization.factory import FAST_TOKENIZER_DEFAULT_FILE_NAME

TOKENIZER_PATH = f"outputs/cyr/gpt/train_tokenizer/ukr-gpt/{FAST_TOKENIZER_DEFAULT_FILE_NAME}"

MODEL_MAX_LENGTH = 1024


# tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH, model_max_length=MODEL_MAX_LENGTH, padding_side="right"
)
tokenizer.add_special_tokens({"bos_token": "<|endoftext|>"})  # TODO: tokenizer saving
# basically `pad_token` wont be used, as DataCollatorForGroupTextForCasualLMDataset pack sequences up to max_length
# but to avoid an error within DataCollatorForGroupTextForCasualLMDataset
tokenizer.pad_token = tokenizer.bos_token

# model
model_config = GPT2Config(vocab_size=len(tokenizer), bos_token_id=tokenizer.bos_token_id)
model = GPT2LMHeadModel(model_config)


# data  # TODO
# oscar_train = (item["text"] for item in load_dataset("oscar", "unshuffled_deduplicated_uk", split="train"))
# mc4_train = (item["text"] for item in load_dataset("mc4", "uk", split="train"))
# cc100_train = (item["text"] for item in load_dataset("cc100", "uk", split="train"))
# mc4_valid = (item["text"] for item in load_dataset("mc4", "uk", split="validation"))
# wiki_train = (item["text"] for item in PostWikiExtractorDataSource(WIKI_EXTRACTED_PATH))

oscar_train = (item["text"] for item in load_dataset("oscar", "unshuffled_deduplicated_uk", split="train[:5000]"))
oscar_valid = (item["text"] for item in load_dataset("oscar", "unshuffled_deduplicated_uk", split="train[5000:10000]"))

train_dataset = GroupTextForCasualLMDataset(
    tokenizer=tokenizer, data_sources=[oscar_train], block_size=MODEL_MAX_LENGTH
)
valid_dataset = GroupTextForCasualLMDataset(
    tokenizer=tokenizer, data_sources=[oscar_valid], block_size=MODEL_MAX_LENGTH
)
data_collator = DataCollatorForGroupTextForCasualLMDataset()

# train-iters 500000
# batch per gpu 4, grad acc 4, whole batch 256 samples == 512k tokens

training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=250000,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    output_dir="temp",
    overwrite_output_dir=True,
    save_steps=250000,
    save_total_limit=2,
    prediction_loss_only=False,
    learning_rate=5e-5,
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
    report_to=["mlflow"],  # ???
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

print("task")
task = TransformersTrainTask(trainer=trainer)
