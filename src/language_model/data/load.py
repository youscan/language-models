import json
import logging
import os
from typing import Any, Callable, Dict, Optional

from ds_shared.download import YsDownloader
from ds_shared.saving import save_pickle

from ..pipeline import SandboxTask


class YSDataDownloadTask(SandboxTask):
    def __init__(
        self,
        credentials_path: str,
        topic_id: int,
        query: Dict[str, str],
        from_mention_id: Optional[int] = None,
        batch_size: int = 50000,
        mention_processor: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda x: x,
        sandbox_folder_path: str = "data",
    ):
        super().__init__(sandbox_folder_path=sandbox_folder_path)
        self.credentials_path = credentials_path
        with open(self.credentials_path, "r", encoding="utf-8") as f:
            self.credentials = json.load(f)
        self.topic_id = topic_id
        self.query = query
        self.from_mention_id = from_mention_id
        self.batch_size = batch_size
        self.mention_processor = mention_processor

    def execute(self, environment_path: str) -> None:
        downloader = YsDownloader(self.credentials, self.query, max_mentions=self.batch_size)

        logging.info(f"Loading {self.topic_id}. First batch")
        mentions_chunk = downloader.download(self.topic_id, last_mention_id=self.from_mention_id)
        while mentions_chunk:
            first_mention_id = min(mention["seq"] for mention in mentions_chunk)
            last_mention_id = max(mention["seq"] for mention in mentions_chunk)
            chunk_name = f"data,topic_id={self.topic_id},seq={first_mention_id}-{last_mention_id}.p"
            mentions_chunk = list(map(self.mention_processor, mentions_chunk))
            save_pickle(mentions_chunk, os.path.join(environment_path, chunk_name))
            logging.info(f"Loading {self.topic_id}. Next batch")
            mentions_chunk = downloader.download(self.topic_id, last_mention_id=last_mention_id)

        logging.info("Download completed.")
