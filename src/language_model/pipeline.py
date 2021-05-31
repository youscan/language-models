class ITask(object):
    def execute(self, environment_path: str) -> None:
        raise NotImplementedError()


class TaskRunner(object):
    def run(self) -> None:
        raise NotImplementedError()
