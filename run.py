import argparse

from language_model.runner import SandboxRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True, help="Configuration file")
    args = parser.parse_args()
    runner = SandboxRunner(config_path=args.task)
    runner.run()
