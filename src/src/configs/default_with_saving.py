import os
from typing import Any, Optional

from overrides import overrides

from .default import Config


class ConfigWithSaving(Config):
    def __init__(
        self, saving_folder_prefix: str, logs_dir: str, statistics_folder_name: str = "statistics", **kwargs: Any
    ) -> None:
        super().__init__(logs_dir=logs_dir)
        self._saving_folder_prefix = saving_folder_prefix
        self._saving_folder: Optional[str] = None
        self._statistics_folder_name = statistics_folder_name

    @property
    def saving_folder(self) -> str:
        if self._saving_folder is not None:
            return self._saving_folder
        raise ValueError("Saving folder was not set!")

    @overrides  # type: ignore
    def set_attributes_from_path(self, config_path: str) -> None:
        super().set_attributes_from_path(config_path)
        self._saving_folder = os.path.join(self._saving_folder_prefix, self.path_suffix)
        if not os.path.exists(self.saving_folder):
            os.makedirs(self.saving_folder)

    @property
    def statistics_folder_path(self) -> str:
        folder_path = os.path.join(self.logging_path(), self._statistics_folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path
