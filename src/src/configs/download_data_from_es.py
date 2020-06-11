from typing import Any, Dict, List, Optional

from ds_shared.download import EsCorrectionFilter, EsCorrectionType, EsDownloadOption
from ds_shared.errors import ConfigurationError

from .default_with_saving import ConfigWithSaving


class DownloadDataFromESConfig(ConfigWithSaving):
    def __init__(
        self,
        date_from: str,
        date_to: str,
        fields: List[str],
        languages_to_download: List[str],
        task: EsDownloadOption,
        correction_types: Optional[List[EsCorrectionType]],
        correction_filter: Optional[EsCorrectionFilter],
        num_workers: int = 10,
        max_mentions_from_topic: int = 100000,
        filters: Optional[List[Dict[str, Any]]] = None,
        saving_folder_prefix: str = "data",
        logs_dir: str = "logs",
    ):
        super().__init__(saving_folder_prefix=saving_folder_prefix, logs_dir=logs_dir)
        self.date_from = date_from
        self.date_to = date_to
        self.fields = fields
        self.task = task

        if self.task is EsDownloadOption.DOWNLOAD_CORRECTIONS and (
            correction_types is None or correction_filter is None
        ):
            raise ConfigurationError("For DOWNLOAD_CORRECTIONS you should set correction_types and correction_filter")

        self.correction_types = correction_types
        self.correction_filter = correction_filter

        self.filters = filters
        self.languages_to_download = languages_to_download
        self.max_mentions_from_topic = max_mentions_from_topic
        self.num_workers = num_workers
