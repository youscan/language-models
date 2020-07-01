# Ukrainian Roberta

Ukrainian Roberta was trained by YouScan data science team.

## Pre-training corpora
Below is the list of corpora used along with the output of wc command (counting lines, words and characters). These corpora were concatenated and tokenized with HuggingFace Roberta Tokenizer.

| Tables        | Lines           | Words  | Characters  |
| ------------- |--------------:| -----:| -----:|
| [Ukrainian Wikipedia - May 2020](https://dumps.wikimedia.org/ukwiki/latest/ukwiki-latest-pages-articles.xml.bz2)      | 18 001 466| 201 207 739 | 2 647 891 947 |
| [Ukrainian OSCAR deduplicated dataset](https://oscar-public.huma-num.fr/shuffled/uk_dedup.txt.gz) | 56 560 011      |    2 250 210 650 | 29 705 050 592 |
| Sampled mentions from social networks | 11 245 710      |    128 461 796 | 1 632 567 763 |
| Total | 85 807 187      |    2 579 880 185 | 33 985 510 302 |

## Pre-training details

* Ukrainian Roberta was trained with code provided in [HuggingFace tutorial](https://huggingface.co/blog/how-to-train)
* Currently released model follows roberta-base-cased model architecture (12-layer, 768-hidden, 12-heads, 125M parameters)
* The model was trained on 4xV100 (85 hours)
* Training configuration you can find in the config file (`configs/train_model/version_1/ukr/ukr-roberta-base.py`)

## Evaluation
ukr-roberta-base was tested on internal YouScan tasks, we got improvements by 2 percent (fscore) comparing to [mBERT](https://huggingface.co/bert-base-multilingual-cased).

## Usage
Ukrainian Roberta is released via [HuggingFace Transformers library](https://huggingface.co/transformers/).

```python
from transformers import pipeline, RobertaForMaskedLM, RobertaTokenizer

model = RobertaForMaskedLM.from_pretrained("ukr-roberta-base")
tokenizer = RobertaTokenizer.from_pretrained("ukr-roberta-base")

fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
fill_mask("Тарас Шевченко – великий українсьский <mask>.")
# [{'sequence': '<s> Тарас Шевченко – великий українсьский поет.</s>',
#   'score': 0.48607954382896423,
#   'token': 11426},
#  {'sequence': '<s> Тарас Шевченко – великий українсьский письменник.</s>',
#   'score': 0.23330871760845184,
#   'token': 10121},
#  {'sequence': '<s> Тарас Шевченко – великий українсьский художник.</s>',
#   'score': 0.06583040952682495,
#   'token': 12836},
#  {'sequence': '<s> Тарас Шевченко – великий українсьский князь.</s>',
#   'score': 0.021497823297977448,
#   'token': 17247},
#  {'sequence': '<s> Тарас Шевченко – великий українсьский народ.</s>',
#   'score': 0.020411811769008636,
#   'token': 1021}]
```

## Author
Vitalii Radchenko - contact me on Twitter [@vitaliradchenko](https://twitter.com/vitaliradchenko)
