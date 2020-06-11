from .default_with_saving import ConfigWithSaving


class PrepareOneLineTextFormatConfig(ConfigWithSaving):
    def __init__(
        self, source_folder_path: str, keep_title: bool = True, saving_folder_prefix: str = "data", logs: str = "logs"
    ) -> None:
        super().__init__(saving_folder_prefix=saving_folder_prefix, logs_dir=logs)
        self.source_folder_path = source_folder_path
        self.keep_title = keep_title
