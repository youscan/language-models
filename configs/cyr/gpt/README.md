# Steps:

1) `python run.py --task configs/cyr/gpt/load_data/wiki.py`
2) `python -m wikiextractor.WikiExtractor outputs/cyr/gpt/load_data/wiki/ukwiki-latest-pages-articles.xml.bz2 -o outputs/cyr/gpt/load_data/wiki/ukwiki-latest-pages-articles -b 1M --no-templates`
3) `python run.py --task configs/cyr/gpt/load_data/in-house.py`
4) `python run.py --task configs/cyr/gpt/extract_texts/train-validation-open-data.py`
5) `python run.py --task configs/cyr/gpt/extract_texts/in-house-data.py`
6) `python run.py --task configs/cyr/gpt/train_tokenizer/ukr-gpt.py`
7) `shuf outputs/cyr/gpt/extract_texts/train-validation-open-data/train.txt -o outputs/cyr/gpt/extract_texts/train-validation-open-data/train_shuffled.txt`
