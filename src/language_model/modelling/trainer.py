import logging
import os
from typing import Optional

from transformers import (
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

from ..pipeline import ITask


class TransformersTrainTask(ITask):
    def __init__(self, trainer: Trainer, checkpoint_folder: Optional[str] = None, model_folder_name: str = "model"):
        super().__init__()
        self.trainer = trainer
        self.checkpoint_folder = checkpoint_folder
        self.model_folder_name = model_folder_name

    def execute(self, environment_path: str) -> None:
        result = self.trainer.train(resume_from_checkpoint=self.checkpoint_folder)
        logging.info(f"Training result: {result}")
        # self.trainer.model.save_pretrained(os.path.join(environment_path, self.model_folder_name))
        self.trainer.save_model(os.path.join(environment_path, self.model_folder_name))


class TransformersTrainTaskWithTokenizerSaving(TransformersTrainTask):
    def __init__(
        self,
        trainer: Trainer,
        checkpoint_folder: Optional[str] = None,
        model_folder_name: str = "model",
        tokenizer_folder_name: str = "tokenizer",
    ):
        super().__init__(trainer, checkpoint_folder, model_folder_name)
        self.tokenizer_folder_name = tokenizer_folder_name

    def execute(self, environment_path: str) -> None:
        super().execute(environment_path)
        self.trainer.tokenizer.save_pretrained(os.path.join(environment_path, self.tokenizer_folder_name))


class RobertaForMaskedLMTrainTask(ITask):
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
            output_dir=os.path.join(environment_path, "temp"),
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

        trainer.save_model(os.path.join(environment_path, "model"))
        self.tokenizer.save_pretrained(os.path.join(environment_path, "tokenizer"))
