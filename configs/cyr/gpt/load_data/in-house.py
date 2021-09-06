from language_model.data.load import YSDataDownloadTask
from language_model.data.processing import LightweightMention

task = YSDataDownloadTask(
    credentials_path="credentials",
    topic_id=275648,
    query={"from": "2019-01-01", "to": "2021-09-01", "sanitize": False, "dedup": False},
    mention_processor=LightweightMention(),
)
