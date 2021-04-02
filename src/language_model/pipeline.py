import os
import pathlib
import sys
from abc import ABC
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Sequence

sys.path.append(str(pathlib.Path().absolute()))

LOGGING_FORMAT: str = "%(asctime)s : %(levelname)s : %(module)s : %(message)s"
DEFAULT_LOG_DIR: str = "logs"
DEFAULT_LOG_FILE: str = "log.txt"
DEFAULT_CONFIGURATION_DIR: str = "configs"


class ITask(object):
    def execute(self, environment_path: str) -> None:
        raise NotImplementedError()


class Sandbox(object):
    def get_sandbox_folder_path(self) -> str:
        raise NotImplementedError()


class AbstractSandbox(Sandbox):
    def __init__(self, sandbox_folder_path: str) -> None:
        self.sandbox_folder_path = sandbox_folder_path

    def get_sandbox_folder_path(self) -> str:
        return self.sandbox_folder_path


class SandboxTask(ITask, AbstractSandbox, ABC):
    def __init__(self, sandbox_folder_path: str = "outputs") -> None:
        super().__init__(sandbox_folder_path)


def identifiers_from_config_file(filepath: str) -> Sequence[str]:
    path = os.path.normpath(filepath)
    path_components = path.split(os.sep)
    path_components[-1] = os.path.splitext(path_components[-1])[0]
    config_dir_index = path_components[:-1].index(DEFAULT_CONFIGURATION_DIR)
    if config_dir_index != -1:
        path_components = path_components[config_dir_index + 1 :]
    return path_components


def init_logger(experiment_identifiers: Sequence[str], overwrite: bool = True, log_to_stderr: bool = False) -> Logger:
    path_components = [DEFAULT_LOG_DIR] + list(experiment_identifiers)
    log_path = os.path.join(*path_components)
    return configure_logger(log_path, file=DEFAULT_LOG_FILE, overwrite=overwrite, log_to_stderr=log_to_stderr)


def configure_logger(
    path: str = DEFAULT_LOG_DIR, file: str = DEFAULT_LOG_FILE, overwrite: bool = True, log_to_stderr: bool = False
) -> Logger:
    logger = getLogger()
    logger.setLevel(INFO)

    logging_path = os.path.join(path)
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    formatter = Formatter(LOGGING_FORMAT)

    fh = FileHandler(os.path.join(logging_path, file), mode="w" if overwrite else "a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if log_to_stderr:
        sh = StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
