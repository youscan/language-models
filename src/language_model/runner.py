import os
from importlib import import_module

from .pipeline import SandboxTask, identifiers_from_config_file, init_logger

TASK_FIELD_NAME: str = "task"


class TaskRunner(object):
    def run(self) -> None:
        raise NotImplementedError()


class ConfigurationFileRunner(TaskRunner):
    def __init__(self, config_path: str):
        self.config_path = config_path
        module_name = os.path.splitext(config_path)[0].replace("/", ".")
        module = import_module(module_name)
        self.task: SandboxTask = getattr(module, TASK_FIELD_NAME)

    def run(self) -> None:
        experiment_ids = identifiers_from_config_file(self.config_path)
        experiment_sandbox_path = os.path.join(*experiment_ids)
        logger = init_logger(experiment_ids, overwrite=True)
        logger.info(f"Running task from {self.config_path}")
        self.task.execute(os.path.join(self.task.get_sandbox_folder_path(), experiment_sandbox_path))
