from ds_shared.download import EsDownloadOption

from src.configs import DownloadDataFromESConfig

config = DownloadDataFromESConfig(
    date_from="2016-01-01",
    date_to="2020-06-01",
    fields=[],
    max_mentions_from_topic=10000,
    task=EsDownloadOption.DOWNLOAD_SAMPLED_TOPICS,
    correction_types=None,
    correction_filter=None,
    languages_to_download=["ukr"],
    num_workers=20,
)
