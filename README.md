<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy + Trankit

This package wraps the [Trankit](https://github.com/nlp-uoregon/trankit) library, so you can use trankit models in a
[spaCy](https://spacy.io) pipeline. 


[//]: # ([![tests]&#40;https://github.com/imvladikon/spacy-trankit/actions/workflows/tests.yml/badge.svg&#41;]&#40;https://github.com/imvladikon/spacy-trankit/actions/workflows/tests.yml&#41;)
[//]: # ([![PyPi]&#40;https://img.shields.io/pypi/v/spacy-trankit.svg?style=flat-square&#41;]&#40;https://pypi.python.org/pypi/spacy-trankit&#41;)
[//]: # ([![GitHub]&#40;https://img.shields.io/github/release/imvladikon/spacy-trankit/all.svg?style=flat-square&#41;]&#40;https://github.com/imvladikon/spacy-trankit&#41;)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

Using this wrapper, you'll be able to use the following annotations, computed by
your pretrained `trankit` pipeline/model:

- Statistical tokenization (reflected in the `Doc` and its tokens)
- Lemmatization (`token.lemma` and `token.lemma_`)
- Part-of-speech tagging (`token.tag`, `token.tag_`, `token.pos`, `token.pos_`)
- Morphological analysis (`token.morph`)
- Dependency parsing (`token.dep`, `token.dep_`, `token.head`)
- Named entity recognition (`doc.ents`, `token.ent_type`, `token.ent_type_`,
  `token.ent_iob`, `token.ent_iob_`)
- Sentence segmentation (`doc.sents`)

## ️️️⌛️ Installation

As of v0.1.0 `spacy-trankit` is only compatible with **spaCy v3.x**. To install
the most recent version:

```bash
pip install git+https://github.com/imvladikon/spacy-trankit
```



## 📖 Usage & Examples

Load pre-trained `trankit` model into a spaCy pipeline:

```python
import spacy_trankit

# Initialize the pipeline
nlp = spacy_trankit.load("english")

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc.ents)
```

Load it from the path:

```python
import spacy_trankit

# Initialize the pipeline
nlp = spacy_trankit.load_from_path(name="en", path="./cache") 

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc.ents)
```