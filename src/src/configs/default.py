import logging
import os
import pathlib
import sys
from importlib import import_module
from typing import Any, Optional, Type, TypeVar

T = TypeVar("T", bound="Config")
# we append a root project path to be able to import `configs`,
# because now we have src/custom_vtorch and we don't see `configs` without changes
sys.path.append(str(pathlib.Path().absolute()))


class Config:
    def __init__(self, logs_dir: str = "logs", **kwargs: Any) -> None:
        super(Config, self).__init__()
        self._logs_dir = logs_dir
        self._experiment_name: Optional[str] = None
        self._language: Optional[str] = None
        self._version: Optional[int] = None
        self._task_name: Optional[str] = None
        self._path_suffix: Optional[str] = None

    @property
    def experiment_name(self) -> str:
        if self._experiment_name is not None:
            return self._experiment_name
        raise ValueError("The experiment name has not been set yet")

    @property
    def version(self) -> int:
        if self._version is not None:
            return self._version
        raise ValueError("The version has not been set")

    @property
    def path_suffix(self) -> str:
        if self._path_suffix is None:
            self._path_suffix = ""
        return self._path_suffix

    def logging_path(self) -> str:
        return os.path.join(self._logs_dir, self.path_suffix)

    def logger(self, mode: str = "w") -> logging.Logger:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logging_path = self.logging_path()
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)

        fh = logging.FileHandler(os.path.join(logging_path, "log.log"), mode=mode, encoding="utf-8")

        formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(module)s : %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @classmethod
    def load(cls: Type[T], module_path: str) -> T:
        module_name = os.path.splitext(module_path)[0].replace("/", ".")
        module = import_module(module_name)
        config: T = module.config  # type: ignore
        config.set_attributes_from_path(module_path)
        return config

    def set_attributes_from_path(self, config_path: str) -> None:
        path, file_name = os.path.split(config_path)
        self._experiment_name = os.path.splitext(file_name)[0]

        _, task_name, version, language = path.rsplit(sep="/", maxsplit=3)
        self._task_name = task_name

        if len(language) != 3:
            raise ValueError(
                f"Language name should consist of three chars, check path {config_path}. "
                "Proper path is configs/task_name/version_'version_number'/language/experiment_name"
            )
        self._language = language

        if "version" not in version or not version.split("_")[-1].isdigit():
            raise ValueError(
                f"Your folder should have 'version_version_number' format, check path {config_path}. "
                "Proper path is configs/task_name/version_'version_number'/language/experiment_name"
            )
        self._version = int(version.split("_")[-1])

        self._path_suffix = os.path.join(
            f"version_{self.version}", self._language, self._task_name, self._experiment_name
        )
