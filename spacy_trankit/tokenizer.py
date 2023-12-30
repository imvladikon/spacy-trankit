#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os.path
import re
import warnings
from typing import List, Tuple, Dict, Optional

from spacy.tokens import Doc
from spacy.util import registry
from spacy.vocab import Vocab
from trankit import Pipeline
from trankit.utils import code2lang, lang2treebank

logger = logging.getLogger(__name__)
logging.getLogger("trankit").setLevel(logging.CRITICAL)


@registry.tokenizers("spacy_trankit.PipelineAsTokenizer.v1")
def create_tokenizer(lang: str, cache_dir: Optional[str] = None):
    def tokenizer_factory(
        nlp, lang=lang, cache_dir=cache_dir, **kwargs
    ) -> TrankitTokenizer:
        load_from_path = cache_dir is not None
        if lang not in lang2treebank and lang in code2lang:
            lang = code2lang[lang]

        if load_from_path:
            if not os.path.exists(cache_dir):
                raise ValueError(
                    f"Path {cache_dir} does not exist. "
                    f"Please download the model and save it to this path."
                )
            model = Pipeline(lang=lang, cache_dir=cache_dir, **kwargs)
        else:
            model = Pipeline(lang=lang, **kwargs)
        return TrankitTokenizer(model=model, vocab=nlp.vocab)

    return tokenizer_factory


class TrankitTokenizer:
    def __init__(self, model: Pipeline, vocab: Vocab):
        self.pipeline = model
        self.vocab = vocab
        self._ws_pattern = re.compile(r"\s+")

    def __call__(self, text):
        """Convert a Stanza Doc to a spaCy Doc.

        text (unicode): The text to process.
        RETURNS (spacy.tokens.Doc): The spaCy Doc object.
        """
        if not text:
            return Doc(self.vocab)
        elif text.isspace():
            return Doc(self.vocab, words=[text], spaces=[False])

        doc = self.pipeline(text)
        text = doc["text"]
        snlp_tokens, snlp_heads, entities = self.get_tokens_with_heads(doc)

        pos = []
        tags = []
        morphs = []
        deps = []
        heads = []
        lemmas = []
        token_texts = [t["text"] for t in snlp_tokens]
        is_aligned = True
        try:
            words, spaces = self.get_words_and_spaces(token_texts, text)
        except ValueError:
            words = token_texts
            spaces = [True] * len(words)
            is_aligned = False
            warnings.warn(
                "Due to multiword token expansion or an alignment "
                "issue, the original text has been replaced by space-separated "
                "expanded tokens.",
                stacklevel=4,
            )
        offset = 0
        for i, word in enumerate(words):
            if word.isspace() and (
                i + offset >= len(snlp_tokens) or word != snlp_tokens[i + offset]["text"]
            ):
                # insert a space token
                pos.append("SPACE")
                tags.append("_SP")
                morphs.append("")
                deps.append("")
                lemmas.append(word)

                # increment any heads left of this position that point beyond
                # this position to the right (already present in heads)
                for j in range(0, len(heads)):
                    if j + heads[j] >= i:
                        heads[j] += 1

                # decrement any heads right of this position that point beyond
                # this position to the left (yet to be added from snlp_heads)
                for j in range(i + offset, len(snlp_heads)):
                    if j + snlp_heads[j] < i + offset:
                        snlp_heads[j] -= 1

                # initial space tokens are attached to the following token,
                # otherwise attach to the preceding token
                if i == 0:
                    heads.append(1)
                else:
                    heads.append(-1)

                offset -= 1
            else:
                token = snlp_tokens[i + offset]
                assert word == token["text"]

                pos.append(token.get("upos", ""))
                tags.append(token.get("xpos") or token.get("upos") or "")
                morphs.append(token.get("feats", ""))
                deps.append(token.get("deprel", ""))
                heads.append(snlp_heads[i + offset])
                lemmas.append(token.get("lemma", ""))

        spacy_doc = Doc(
            self.vocab,
            words=words,
            spaces=spaces,
            pos=pos,
            tags=tags,
            morphs=morphs,
            lemmas=lemmas,
            deps=deps,
            heads=[head + i for i, head in enumerate(heads)],
        )

        if entities is not None:
            ents = [
                spacy_doc.char_span(start, end, self.normalize_entity_tag(label))
                for label, start, end in entities
            ]
            if not is_aligned or not all(ents):
                warnings.warn(
                    f"Can't set named entities because of multi-word token "
                    f"expansion or because the character offsets don't map to "
                    f"valid tokens produced by the Trankit tokenizer:\n"
                    f"Words: {words}\n"
                    f"Entities: {[(label, start, end) for label, start, end in entities]}",
                    stacklevel=4,
                )
            else:
                spacy_doc.ents = ents

        return spacy_doc

    def normalize_entity_tag(self, tag):
        if "-" in tag:
            return tag.split("-")[-1]
        else:
            return tag

    def pipe(self, texts):
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts.
        YIELDS (Doc): A sequence of Doc objects, in order.
        """
        for text in texts:
            yield self(text)

    def get_tokens_with_heads(
        self, doc: Dict, exlude_tag="O"
    ) -> Tuple[List, List, List]:
        """Flatten the tokens in the Trankit Doc and extract the token indices
        of the sentence start tokens to set is_sent_start.

        doc (Dict): The processed Stanza doc in the Dict format
        RETURNS (list): The tokens (words).
        """
        tokens = []
        heads = []
        entities = None
        offset = 0
        for sentence in doc["sentences"]:
            s_offset = 0
            for token in sentence["tokens"]:
                words = token.get("expanded", [token])
                for word in words:
                    # Here, we're calculating the absolute token index in the doc,
                    # then the *relative* index of the head, -1 for zero-indexed
                    # and if the governor is 0 (root), we leave it at 0
                    if word["head"]:
                        head = word["head"] + offset - len(tokens) - 1
                    else:
                        head = 0
                    heads.append(head)
                    tokens.append(word)
                    if "ner" in word:
                        if entities is None:
                            entities = []
                        if word["ner"] != exlude_tag:
                            entities.append(
                                (
                                    word["ner"],
                                    word["dspan"][0],
                                    word["dspan"][1],
                                )
                            )
                s_offset += len(words)
                char_offset = token["span"][1]
            offset += s_offset
        return tokens, heads, entities

    def get_words_and_spaces(self, words, text):
        if "".join("".join(words).split()) != "".join(text.split()):
            raise ValueError("Unable to align mismatched text and words.")
        text_words = []
        text_spaces = []
        text_pos = 0
        # normalize words to remove all whitespace tokens
        norm_words = [word for word in words if not word.isspace()]
        # align words with text
        for word in norm_words:
            try:
                word_start = text[text_pos:].index(word)
            except ValueError:
                raise ValueError("Unable to align mismatched text and words.")
            if word_start > 0:
                text_words.append(text[text_pos : text_pos + word_start])
                text_spaces.append(False)
                text_pos += word_start
            text_words.append(word)
            text_spaces.append(False)
            text_pos += len(word)
            if text_pos < len(text) and text[text_pos] == " ":
                text_spaces[-1] = True
                text_pos += 1
        if text_pos < len(text):
            text_words.append(text[text_pos:])
            text_spaces.append(False)
        return (text_words, text_spaces)

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes_data, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
