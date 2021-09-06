from itertools import islice

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from language_model.data.extract import LineByLineSource
from language_model.tokenization.trainer import TrainTokenizerTask

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)


TRAIN_DATA_PATH = "outputs/cyr/gpt/extract_texts/train-validation-open-data/train.txt"
NUM_TRAIN_LINES = 1_000_000
TRAIN_SAMPLING_STEP = 200
train_data_source = islice(
    (line for i, line in enumerate(LineByLineSource(TRAIN_DATA_PATH)) if i % TRAIN_SAMPLING_STEP == 0), NUM_TRAIN_LINES
)
trainer = trainers.BpeTrainer(vocab_size=50264, special_tokens=["<|endoftext|>"])

task = TrainTokenizerTask(
    tokenizer=tokenizer,
    iterator=train_data_source,
    trainer=trainer,
)
