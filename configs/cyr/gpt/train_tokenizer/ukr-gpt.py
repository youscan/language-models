from itertools import chain

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from language_model.tokenization.trainer import TrainTokenizerTask

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)


oscar_uk = (item["text"] for item in load_dataset("oscar", "unshuffled_deduplicated_uk", split="train"))
# mc4_uk = (item["text"] for item in load_dataset("mc4", "uk", split="train"))
# cc100_uk = (item["text"] for item in load_dataset("cc100", "uk", split="train"))

trainer = trainers.BpeTrainer(vocab_size=50264, special_tokens=["<|endoftext|>"])

task = TrainTokenizerTask(
    tokenizer=tokenizer,
    # iterator=chain(oscar_uk, mc4_uk, cc100_uk),
    iterator=chain(oscar_uk),
    trainer=trainer,
)
