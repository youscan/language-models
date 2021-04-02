import os

from transformers import (
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

from ..pipeline import SandboxTask


class Test(object):
    def __init__(self, s: str, i: int):
        self.i = i
        self.s = s


class RobertaForMaskedLMTrainTask(SandboxTask):
    def __init__(
        self,
        file_path: str,
        model: RobertaForMaskedLM,
        tokenizer: RobertaTokenizerFast,
        block_size: int = 128,
        mlm_probability: float = 0.15,
        epochs: int = 1,
        batch_size_per_gpu: int = 64,
        save_steps: int = 10000,
        save_total_limit: int = 2,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mlm_probability = mlm_probability
        self.epochs = epochs
        self.batch_size_per_gpu = batch_size_per_gpu
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit

    def execute(self, environment_path: str) -> None:
        dataset = LineByLineTextDataset(tokenizer=self.tokenizer, file_path=self.file_path, block_size=self.block_size)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_probability
        )

        training_args = TrainingArguments(
            output_dir=self.sandbox_folder_path,
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_gpu_train_batch_size=self.batch_size_per_gpu,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            prediction_loss_only=True,
        )

        trainer.train()

        trainer.save_model(os.path.join(self.sandbox_folder_path, "model"))
        self.tokenizer.save_pretrained(os.path.join(self.sandbox_folder_path, "tokenizer"))
