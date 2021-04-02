import argparse

from language_model.runner import ConfigurationFileRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True, help="Configuration file")
    args = parser.parse_args()
    runner = ConfigurationFileRunner(args.task)
    runner.run()
