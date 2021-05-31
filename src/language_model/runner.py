import os
from importlib import import_module
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Sequence

from .pipeline import ITask, TaskRunner

LOGGING_FORMAT: str = "%(asctime)s : %(levelname)s : %(module)s : %(message)s"
DEFAULT_LOG_DIR: str = "logs"
DEFAULT_LOG_FILE: str = "log.txt"
DEFAULT_CONFIGURATION_DIR: str = "configs"
TASK_FIELD_NAME: str = "task"


class SandboxRunner(TaskRunner):
    def __init__(self, config_path: str, sandbox_root_path: str = "outputs") -> None:
        self.config_path = config_path
        self.sandbox_root_path = sandbox_root_path

    def get_root_folder_path(self) -> str:
        return self.sandbox_root_path

    def run(self) -> None:
        experiment_ids = identifiers_from_config_file(self.config_path)
        module_name = ".".join(experiment_ids)
        module = import_module(module_name)
        task: ITask = getattr(module, TASK_FIELD_NAME)

        pure_experiment_ids = drop_configuration_dir(experiment_ids=experiment_ids)
        experiment_sandbox_path = os.path.join(*pure_experiment_ids)
        sandbox_folder_path = os.path.join(self.get_root_folder_path(), experiment_sandbox_path)

        logger = init_logger(experiment_ids, overwrite=True)
        logger.info(f"Running task from {self.config_path}")
        if not os.path.exists(sandbox_folder_path) or not os.path.isdir(sandbox_folder_path):
            os.makedirs(sandbox_folder_path)
        task.execute(sandbox_folder_path)


def identifiers_from_config_file(filepath: str) -> Sequence[str]:
    path = os.path.normpath(filepath)
    path_components = path.split(os.sep)
    path_components[-1] = os.path.splitext(path_components[-1])[0]
    return path_components


def drop_configuration_dir(experiment_ids: Sequence[str]) -> Sequence[str]:
    config_dir_index = experiment_ids[:-1].index(DEFAULT_CONFIGURATION_DIR)
    if config_dir_index != -1:
        experiment_ids = experiment_ids[config_dir_index + 1 :]
    return experiment_ids


def init_logger(experiment_identifiers: Sequence[str], overwrite: bool = True, log_to_stderr: bool = False) -> Logger:
    path_components = [DEFAULT_LOG_DIR] + list(experiment_identifiers)
    log_path = os.path.join(*path_components)

    logger = getLogger()
    logger.setLevel(INFO)

    logging_path = os.path.join(log_path)
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    formatter = Formatter(LOGGING_FORMAT)

    fh = FileHandler(os.path.join(logging_path, DEFAULT_LOG_FILE), mode="w" if overwrite else "a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if log_to_stderr:
        sh = StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
