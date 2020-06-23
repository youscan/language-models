from transformers import RobertaForMaskedLM, RobertaTokenizerFast

from .default_with_saving import ConfigWithSaving


class TrainModelConfig(ConfigWithSaving):
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
        saving_folder_prefix: str = "results",
        logs: str = "logs",
    ) -> None:
        super().__init__(saving_folder_prefix=saving_folder_prefix, logs_dir=logs)
        self.file_path = file_path
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mlm_probability = mlm_probability
        self.epochs = epochs
        self.batch_size_per_gpu = batch_size_per_gpu
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
