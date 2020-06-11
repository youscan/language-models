from typing import List, Optional

from .default_with_saving import ConfigWithSaving


class PrepareDataFromEsConfig(ConfigWithSaving):
    def __init__(
        self,
        source_folder_path: str,
        cache_size: int = 50000,
        logging_step: int = 100000,
        checkpoint_step: int = 1000000,
        proper_languages: Optional[List[str]] = None,
        saving_folder_prefix: str = "data",
        logs: str = "logs",
    ) -> None:
        super().__init__(saving_folder_prefix=saving_folder_prefix, logs_dir=logs)
        self.source_folder_path = source_folder_path
        self.cache_size = cache_size
        self.proper_languages = proper_languages
        self.checkpoint_step = checkpoint_step
        self.logging_step = logging_step
