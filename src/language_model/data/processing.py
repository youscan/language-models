import logging
from typing import Any, Dict

ID_KEY = "id"
AUTHOR_KEY = "author"
AUTHOR_ID_KEY = "author_id"
CHANNEL_KEY = "channel"
CHANNEL_ID_KEY = "channel_id"
SOURCE_KEY = "source"
SOURCE_ID_KEY = "source_id"

SYSTEM_TAGS_KEY = "systemTags"
AUTOCATEGORIES_KEY = "autoCategories"

DROP_KEYS = [
    "publishedLocal",
    "publishedUtc",
    "region",
    "city",
    "tags",
    "trends",
    "similarCount",
    "duplicatesCount",
    "processed",
    "deleted",
    "starred",
    "engagement",
    "sentiment",
    "url",
    "appliedRuleIds",
    "integrations",
    "origin",
    "topic_id",
    "apiName",
    "postId",
    "parentPostId",
    "discussionId",
]


class LightweightMention(object):
    def __call__(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        mention = dict(mention)
        try:
            if AUTHOR_KEY in mention:
                mention[AUTHOR_ID_KEY] = mention[AUTHOR_KEY].get(ID_KEY, None)
                mention.pop(AUTHOR_KEY)
        except Exception as e:
            logging.error(e)
        try:
            if CHANNEL_KEY in mention:
                mention[CHANNEL_ID_KEY] = mention[CHANNEL_KEY].get(ID_KEY, None)
                mention.pop(CHANNEL_KEY)
        except Exception as e:
            logging.error(e)
        try:
            if SOURCE_KEY in mention:
                mention[SOURCE_ID_KEY] = mention[SOURCE_KEY].get(ID_KEY, None)
                mention.pop(SOURCE_KEY)
        except Exception as e:
            logging.error(e)
        try:
            if tuple(sorted(mention.get(SYSTEM_TAGS_KEY, []))) == tuple(sorted(mention.get(AUTOCATEGORIES_KEY, []))):
                mention.pop(SYSTEM_TAGS_KEY, None)
        except Exception as e:
            logging.error(e)
        for key in DROP_KEYS:
            try:
                mention.pop(key, None)
            except Exception as e:
                logging.error(e)
        return mention
