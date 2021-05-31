import argparse

from language_model.runner import SandboxRunner

DATA_FOLDER_PATH = "data"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True, help="Configuration file")
    args = parser.parse_args()
    runner = SandboxRunner(config_path=args.task, sandbox_root_path=DATA_FOLDER_PATH)
    runner.run()
