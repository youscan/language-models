import argparse
import logging
import os
import sys
from typing import Any, Dict, List

import requests
from ds_shared.download import ESDownloader

from src.configs import DownloadDataFromESConfig

logging.getLogger("elasticsearch").setLevel(logging.ERROR)

_TOPIC_LIST_URL = "http://ys-mentionstreamapi-production.yscan.zone51/topics"


def block_print() -> None:
    sys.stdout = open(os.devnull, "w")


def get_available_topics() -> List[int]:
    topics: List[Dict[str, Any]] = requests.get(_TOPIC_LIST_URL).json()["topics"]
    return [topic_info["id"] for topic_info in topics]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=None, type=str, required=True, help="Configuration file")
    args = parser.parse_args()

    config = DownloadDataFromESConfig.load(args.config_file)
    logger = config.logger()

    logger.info("Get available topics ids")
    topics_ids = get_available_topics()

    logger.info("Downloading topics data ...")

    downloader = ESDownloader(
        saving_folder=config.saving_folder,
        date_from=config.date_from,
        date_to=config.date_to,
        fields=config.fields,
        task=config.task,
        correction_types=config.correction_types,
        correction_filter=config.correction_filter,
        filters=config.filters,
        max_mentions_from_topic=config.max_mentions_from_topic,
        languages=config.languages_to_download,
        num_workers=config.num_workers,
    )
    downloader.download(topics_ids)


if __name__ == "__main__":
    block_print()
    main()
