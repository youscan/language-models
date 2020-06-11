import argparse

from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.configs import TrainModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None, type=str, required=True, help="Configuration file")
    args = parser.parse_args()

    config = TrainModelConfig.load(args.config_file)
    logger = config_file.logger()  # noqa: F841

    dataset = LineByLineTextDataset(
        tokenizer=config.tokenizer,
        file_path=config.file_path,
        block_size=config.block_size,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=config.tokenizer, mlm=True, mlm_probability=config.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=config.saving_folder,
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        per_gpu_train_batch_size=config.batch_size_per_gpu,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
    )

    trainer = Trainer(
        model=config.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train()

    trainer.save_model(config.saving_folder)
    config.tokenizer.save_pretrained(config.saving_folder)


if __name__ == '__main__':
    main()
