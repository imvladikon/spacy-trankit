#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=C0114, C0115, C0116, W0613
import logging
import os
import os.path
import re
import sys
import urllib.request
import warnings
import zipfile
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from spacy import Vocab
from spacy.tokens import Doc, Token
from spacy.util import registry

# HACK: alias `torch.optim.AdamW` onto `transformers` so Trankit's `from transformers import AdamW` import doesn't blow up on transformers 4.46+ where AdamW was removed.
try:  # pragma: no cover - import-time shim
    import transformers as _transformers

    if not hasattr(_transformers, "AdamW"):
        try:
            from torch.optim import AdamW as _AdamW

            _transformers.AdamW = _AdamW
        except ImportError:
            pass
except ImportError:
    pass

def _is_py312_dataclass_error(exc: Exception) -> bool:
    return (
        isinstance(exc, ValueError)
        and "mutable default" in str(exc)
        and "adapter_config" in str(exc)
    )


def _patch_trankit_adapter_config_for_py312(
    adapter_config_path: Optional[Path] = None,
) -> bool:
    if adapter_config_path is None:
        spec = find_spec("trankit")
        if (
            spec is None
            or spec.submodule_search_locations is None
            or len(spec.submodule_search_locations) == 0
        ):
            return False
        adapter_config_path = (
            Path(spec.submodule_search_locations[0])
            / "adapter_transformers"
            / "adapter_config.py"
        )
    if not adapter_config_path.exists():
        return False

    source = adapter_config_path.read_text(encoding="utf-8")
    if "default_factory=lambda: InvertibleAdapterConfig(" in source:
        return True

    target = (
        "invertible_adapter: Optional[dict] = InvertibleAdapterConfig(\n"
        '        block_type="nice", non_linearity="relu", reduction_factor=2\n'
        "    )"
    )
    replacement = (
        "invertible_adapter: Optional[dict] = field(\n"
        "        default_factory=lambda: InvertibleAdapterConfig(\n"
        '            block_type="nice", non_linearity="relu", reduction_factor=2\n'
        "        )\n"
        "    )"
    )
    if target not in source:
        return False

    adapter_config_path.write_text(
        source.replace(target, replacement), encoding="utf-8"
    )
    return True


def _import_trankit():
    try:
        from trankit import Pipeline as _Pipeline
        from trankit.utils import code2lang as _code2lang, lang2treebank as _lang2treebank

        return _Pipeline, _code2lang, _lang2treebank, None
    except (ImportError, ValueError) as exc:  # pragma: no cover - env-dependent
        if sys.version_info >= (3, 12) and _is_py312_dataclass_error(exc):
            if _patch_trankit_adapter_config_for_py312():
                for module_name in list(sys.modules.keys()):
                    if module_name == "trankit" or module_name.startswith("trankit."):
                        sys.modules.pop(module_name, None)
                try:
                    from trankit import Pipeline as _Pipeline
                    from trankit.utils import code2lang as _code2lang, lang2treebank as _lang2treebank

                    return _Pipeline, _code2lang, _lang2treebank, None
                except (ImportError, ValueError) as retry_exc:
                    return None, {}, {}, retry_exc
        return None, {}, {}, exc


Pipeline, code2lang, lang2treebank, TRANKIT_IMPORT_ERROR = _import_trankit()


logger = logging.getLogger(__name__)
logging.getLogger("trankit").setLevel(logging.CRITICAL)


# HACK: Trankit's PyPI release fetches models from a dead `nlp.uoregon.edu` host, so we pre-populate its cache from the HuggingFace mirror (overridable via `SPACY_TRANKIT_MODEL_URL`).
DEFAULT_TRANKIT_MODEL_URL = (
    "https://huggingface.co/uonlp/trankit/resolve/main"
    "/models/{version}/{embedding}/{lang}.zip"
)
DEFAULT_TRANKIT_CACHE_DIR = "cache/trankit"
DEFAULT_TRANKIT_MODEL_VERSION = "v1.0.0"
DEFAULT_TRANKIT_EMBEDDING = "xlm-roberta-base"


def ensure_trankit_model(
    cache_dir: str,
    lang: str,
    embedding: str = DEFAULT_TRANKIT_EMBEDDING,
    version: str = DEFAULT_TRANKIT_MODEL_VERSION,
    url_template: Optional[str] = None,
) -> None:
    """Pre-fetch a Trankit model into ``cache_dir`` if it isn't already there.

    Mirrors the directory layout Trankit's own ``download`` helper produces, so
    once this returns Trankit will see ``{lang}.downloaded`` and skip its own
    (currently broken) HTTP fetch.
    """
    if url_template is None:
        url_template = os.environ.get(
            "SPACY_TRANKIT_MODEL_URL", DEFAULT_TRANKIT_MODEL_URL
        )
    lang_dir = os.path.join(cache_dir, embedding, lang)
    marker = os.path.join(lang_dir, f"{lang}.downloaded")
    if os.path.exists(marker):
        return
    os.makedirs(lang_dir, exist_ok=True)
    url = url_template.format(version=version, embedding=embedding, lang=lang)
    zip_path = os.path.join(lang_dir, f"{lang}.zip")
    logger.info("Downloading Trankit model %s/%s from %s", embedding, lang, url)
    urllib.request.urlretrieve(url, zip_path)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(lang_dir)
    finally:
        try:
            os.remove(zip_path)
        except OSError:
            pass
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("")


@registry.tokenizers("spacy_trankit.PipelineAsTokenizer.v1")
def create_tokenizer(
    lang: str, cache_dir: Optional[str] = None, mwt_strategy: str = "auto"
):
    def tokenizer_factory(
        nlp, lang=lang, cache_dir=cache_dir, **kwargs
    ) -> "TrankitTokenizer":
        if Pipeline is None:
            msg = (
                "spacy-trankit requires a working trankit installation to "
                "create a tokenizer. Install it with `pip install trankit` or "
                "install the package with runtime dependencies."
            )
            if (
                isinstance(TRANKIT_IMPORT_ERROR, ValueError)
                and _is_py312_dataclass_error(TRANKIT_IMPORT_ERROR)
            ):
                msg += (
                    " Detected an incompatible trankit build on Python 3.12 "
                    "(dataclass mutable default error in "
                    "`trankit.adapter_transformers.adapter_config`). "
                    "Use Python 3.11/3.10 or a trankit release that supports "
                    "Python 3.12."
                )
            raise ImportError(msg) from TRANKIT_IMPORT_ERROR
        load_from_path = cache_dir is not None
        if lang not in lang2treebank and lang in code2lang:
            lang = code2lang[lang]

        embedding = kwargs.get("embedding", DEFAULT_TRANKIT_EMBEDDING)
        if load_from_path:
            if not os.path.exists(cache_dir):
                raise ValueError(
                    f"Path {cache_dir} does not exist. "
                    f"Please download the model and save it to this path."
                )
            ensure_trankit_model(cache_dir, lang, embedding=embedding)
            model = Pipeline(lang=lang, cache_dir=cache_dir, **kwargs)
        else:
            ensure_trankit_model(
                DEFAULT_TRANKIT_CACHE_DIR, lang, embedding=embedding
            )
            model = Pipeline(lang=lang, **kwargs)
        return TrankitTokenizer(
            model=model, vocab=nlp.vocab, mwt_strategy=mwt_strategy
        )

    return tokenizer_factory


class TrankitTokenizer:
    def __init__(
        self, model: Pipeline, vocab: Vocab, mwt_strategy: str = "auto"
    ):
        if mwt_strategy not in {"auto", "preserve", "expand"}:
            raise ValueError(
                "mwt_strategy must be 'auto', 'preserve', or 'expand'."
            )
        self.pipeline = model
        self.vocab = vocab
        self.mwt_strategy = mwt_strategy
        self._ws_pattern = re.compile(r"\s+")
        if not Token.has_extension("trankit_expanded"):
            Token.set_extension("trankit_expanded", default=None)

    def __call__(self, text):
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements, no-else-return
        if not text:
            return Doc(self.vocab)
        elif text.isspace():
            return Doc(self.vocab, words=[text], spaces=[False])

        doc = self.pipeline(text)
        text = doc["text"]
        (
            snlp_tokens,
            snlp_heads,
            entities,
            sent_starts,
            expanded,
        ) = self.get_tokens_with_heads(doc)

        pos, tags, morphs, deps, heads, lemmas, doc_sent_starts = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        doc_expanded = []
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
                i + offset >= len(snlp_tokens)
                or word != snlp_tokens[i + offset]["text"]
            ):
                # insert a space token
                pos.append("SPACE")
                tags.append("_SP")
                morphs.append("")
                deps.append("")
                lemmas.append(word)
                doc_sent_starts.append(False)
                doc_expanded.append(None)

                # increment any heads left of this position that point beyond
                # this position to the right (already present in heads)
                # pylint: disable=consider-using-enumerate
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
                doc_sent_starts.append(sent_starts[i + offset])
                doc_expanded.append(expanded[i + offset])

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
            sent_starts=doc_sent_starts,
        )
        for token, expansion in zip(spacy_doc, doc_expanded):
            if expansion is not None:
                token._.trankit_expanded = expansion

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
                    f"Entities: {entities}",
                    stacklevel=4,
                )
            else:
                spacy_doc.ents = ents

        return spacy_doc

    def normalize_entity_tag(self, tag):
        return tag.split("-")[-1] if "-" in tag else tag

    def pipe(self, texts):
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts.
        YIELDS (Doc): A sequence of Doc objects, in order.
        """
        for text in texts:
            yield self(text)

    def get_tokens_with_heads(
        self, doc: Dict, exlude_tag="O"
    ) -> Tuple[List, List, List, List, List]:
        """Flatten the tokens in the Trankit Doc and extract the token indices
        of the sentence start tokens to set is_sent_start.

        doc (Dict): The processed Stanza doc in the Dict format
        RETURNS (list): The tokens (words).
        """
        tokens = []
        heads = []
        sent_starts = []
        expanded = []
        entities = None
        offset = 0
        for sentence in doc["sentences"]:
            sentence_items = []
            expanded_to_output = {}
            trankit_word_idx = 1
            for token in sentence["tokens"]:
                words = token.get("expanded", [token])
                mwt_word_indices = list(
                    range(trankit_word_idx, trankit_word_idx + len(words))
                )
                use_expanded = self.should_expand_mwt(words, token)
                if use_expanded:
                    output_words = words
                    output_expanded = [None] * len(words)
                    representatives = words
                    output_word_indices = mwt_word_indices
                else:
                    representative = self.get_surface_representative(
                        words, mwt_word_indices
                    )
                    output_words = [
                        self.make_surface_token(token, representative, words)
                    ]
                    output_expanded = [self.serialize_expansion(words)]
                    representatives = [representative]
                    output_word_indices = [mwt_word_indices[0]]

                if use_expanded:
                    for position, word_idx in enumerate(mwt_word_indices):
                        expanded_to_output[word_idx] = (
                            offset + len(sentence_items) + position
                        )
                else:
                    for word_idx in mwt_word_indices:
                        expanded_to_output[word_idx] = offset + len(sentence_items)
                for word, representative, expansion, word_idx in zip(
                    output_words,
                    representatives,
                    output_expanded,
                    output_word_indices,
                ):
                    sentence_items.append(
                        {
                            "word": word,
                            "representative": representative,
                            "expanded": expansion,
                            "word_idx": word_idx,
                        }
                    )
                trankit_word_idx += len(words)

            for item_index, item in enumerate(sentence_items):
                word = item["word"]
                representative = item["representative"]
                head_idx = representative.get("head", 0)
                current_output_index = offset + item_index
                if head_idx:
                    head_output_index = expanded_to_output.get(
                        head_idx, current_output_index
                    )
                    head = head_output_index - current_output_index
                else:
                    head = 0
                heads.append(head)
                tokens.append(word)
                sent_starts.append(item_index == 0)
                expanded.append(item["expanded"])
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
            offset += len(sentence_items)
        return tokens, heads, entities, sent_starts, expanded

    def expansion_matches_surface(self, words, token):
        return (
            "".join(word["text"] for word in words).strip()
            == token["text"].strip()
        )

    def should_expand_mwt(self, words, token):
        if len(words) == 1:
            return True
        if self.mwt_strategy == "expand":
            return True
        if self.mwt_strategy == "preserve":
            return False
        return self.expansion_matches_surface(words, token)

    def get_surface_representative(self, words, word_indices):
        word_index_set = set(word_indices)
        for word in words:
            head = word.get("head", 0)
            if head == 0 or head not in word_index_set:
                return word
        return words[0]

    def make_surface_token(self, token, representative, words):
        surface = dict(representative)
        surface["text"] = token["text"]
        if "dspan" in token:
            surface["dspan"] = token["dspan"]
        elif words and "dspan" in words[0] and "dspan" in words[-1]:
            surface["dspan"] = [words[0]["dspan"][0], words[-1]["dspan"][1]]
        if "ner" in token:
            surface["ner"] = token["ner"]
        return surface

    def serialize_expansion(self, words):
        fields = ("text", "lemma", "upos", "xpos", "feats", "deprel", "head")
        return [
            {field: word[field] for field in fields if field in word}
            for word in words
        ]

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
            except ValueError as error:
                raise ValueError(
                    "Unable to align mismatched text and words."
                ) from error
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
