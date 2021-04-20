from torch import cuda, device
from transformers import (
    AutoModelForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from language_model.data.dataset import LineByLineTextDataset
from language_model.modelling.trainer import TransformersTrainTask
from language_model.tokenization.factory import FAST_TOKENIZER_DEFAULT_FILE_NAME

DEVICE = device(f"cuda:{0}")
cuda.set_device(DEVICE)
TRANSFORMER_MODEL_NAME = "DeepPavlov/rubert-base-cased-conversational"
TOKENIZER_PATH = (
    f"outputs/cyr/adapt_rubert-base-cased-conversational/tokenizer/extend_base_vocab=96k"
    f"/{FAST_TOKENIZER_DEFAULT_FILE_NAME}"
)
TEXT_FILE_PATHS = [
    "/home/pk/language-models/data/cyr/adapt_rubert-base-cased-conversational/extract_texts/rus/texts.txt",
    "/home/pk/language-models/data/cyr/adapt_rubert-base-cased-conversational/extract_texts/unc/texts.txt",
    "/home/pk/language-models/data/cyr/adapt_rubert-base-cased-conversational/extract_texts/ukr/texts.txt",
]

tokenizer = BertTokenizerFast(
    vocab_file="",
    tokenizer_file=TOKENIZER_PATH,
    from_slow=False,
    do_lower_case=False,
    strip_accents=False,
    tokenize_chinese_chars=False,
)
model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=TRANSFORMER_MODEL_NAME)
model.to(DEVICE)
model.resize_token_embeddings(tokenizer.get_vocab_size(with_added_tokens=True))

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=TEXT_FILE_PATHS, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="temp",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=5000,
    save_total_limit=2,
    prediction_loss_only=True,
    seed=42,
)


trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)


task = TransformersTrainTask(trainer=trainer)
