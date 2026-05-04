#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib.util
import os
import unittest

from spacy.vocab import Vocab

import spacy_trankit
from spacy_trankit.tokenizer import TrankitTokenizer


class FakePipeline:
    def __init__(self, doc):
        self.doc = doc

    def __call__(self, text):
        self.doc["text"] = text
        return self.doc


class TestTokenizer(unittest.TestCase):
    def make_tokenizer(self, doc, **kwargs):
        return TrankitTokenizer(FakePipeline(doc), Vocab(), **kwargs)

    def test_preserves_surface_form_for_unalignable_mwt(self):
        text = "ذهبت للبيت اليوم"
        tokenizer = self.make_tokenizer(
            {
                "sentences": [
                    {
                        "tokens": [
                            {
                                "text": "ذهبت",
                                "lemma": "ذهب",
                                "upos": "VERB",
                                "deprel": "root",
                                "head": 0,
                                "dspan": [0, 4],
                            },
                            {
                                "text": "للبيت",
                                "dspan": [5, 10],
                                "expanded": [
                                    {
                                        "text": "ل",
                                        "lemma": "ل",
                                        "upos": "ADP",
                                        "deprel": "case",
                                        "head": 3,
                                        "dspan": [5, 6],
                                    },
                                    {
                                        "text": "البيت",
                                        "lemma": "بيت",
                                        "upos": "NOUN",
                                        "deprel": "obl",
                                        "head": 1,
                                        "dspan": [6, 10],
                                    },
                                ],
                            },
                            {
                                "text": "اليوم",
                                "lemma": "يوم",
                                "upos": "NOUN",
                                "deprel": "obl",
                                "head": 1,
                                "dspan": [11, 16],
                            },
                        ]
                    }
                ]
            }
        )

        doc = tokenizer(text)

        self.assertEqual(doc.text, text)
        self.assertEqual([token.text for token in doc], ["ذهبت", "للبيت", "اليوم"])
        self.assertEqual(doc[1].lemma_, "بيت")
        self.assertEqual(doc[1].pos_, "NOUN")
        self.assertEqual(doc[1].dep_, "obl")
        self.assertEqual(doc[1].head, doc[0])
        self.assertEqual([sent.text for sent in doc.sents], [text])
        self.assertEqual(
            [word["text"] for word in doc[1]._.trankit_expanded],
            ["ل", "البيت"],
        )

    def test_arabic_mwt_expands_when_surface_aligned(self):
        text = "زارني بالمدرسة"
        tokenizer = self.make_tokenizer(
            {
                "sentences": [
                    {
                        "tokens": [
                            {
                                "text": "زارني",
                                "dspan": [0, 5],
                                "expanded": [
                                    {
                                        "text": "زار",
                                        "lemma": "زار",
                                        "upos": "VERB",
                                        "deprel": "root",
                                        "head": 0,
                                        "dspan": [0, 3],
                                    },
                                    {
                                        "text": "ني",
                                        "lemma": "أنا",
                                        "upos": "PRON",
                                        "deprel": "obj",
                                        "head": 1,
                                        "dspan": [3, 5],
                                    },
                                ],
                            },
                            {
                                "text": "بالمدرسة",
                                "dspan": [6, 14],
                                "expanded": [
                                    {
                                        "text": "ب",
                                        "lemma": "ب",
                                        "upos": "ADP",
                                        "deprel": "case",
                                        "head": 4,
                                        "dspan": [6, 7],
                                    },
                                    {
                                        "text": "المدرسة",
                                        "lemma": "مدرسة",
                                        "upos": "NOUN",
                                        "deprel": "obl",
                                        "head": 1,
                                        "dspan": [7, 14],
                                    },
                                ],
                            },
                        ]
                    }
                ]
            }
        )

        doc = tokenizer(text)

        self.assertEqual(doc.text, text)
        self.assertEqual([token.text for token in doc], ["زار", "ني", "ب", "المدرسة"])
        self.assertEqual(doc[0].dep_, "root")
        self.assertEqual(doc[0].head, doc[0])
        self.assertEqual(doc[1].dep_, "obj")
        self.assertEqual(doc[1].head, doc[0])
        self.assertEqual(doc[2].dep_, "case")
        self.assertEqual(doc[2].head, doc[3])
        self.assertEqual(doc[3].dep_, "obl")
        self.assertEqual(doc[3].head, doc[0])
        self.assertIsNone(doc[0]._.trankit_expanded)
        self.assertIsNone(doc[2]._.trankit_expanded)

    def test_preserve_strategy_keeps_aligned_mwt_as_surface_token(self):
        text = "زارني"
        tokenizer = self.make_tokenizer(
            {
                "sentences": [
                    {
                        "tokens": [
                            {
                                "text": "زارني",
                                "dspan": [0, 5],
                                "expanded": [
                                    {
                                        "text": "زار",
                                        "lemma": "زار",
                                        "upos": "VERB",
                                        "deprel": "root",
                                        "head": 0,
                                        "dspan": [0, 3],
                                    },
                                    {
                                        "text": "ني",
                                        "lemma": "أنا",
                                        "upos": "PRON",
                                        "deprel": "obj",
                                        "head": 1,
                                        "dspan": [3, 5],
                                    },
                                ],
                            },
                        ]
                    }
                ]
            },
            mwt_strategy="preserve",
        )

        doc = tokenizer(text)

        self.assertEqual(doc.text, text)
        self.assertEqual([token.text for token in doc], ["زارني"])
        self.assertEqual(doc[0].lemma_, "زار")
        self.assertEqual(
            [word["text"] for word in doc[0]._.trankit_expanded],
            ["زار", "ني"],
        )

    def test_hebrew_mwt_preserves_surface_and_external_head(self):
        text = "ראיתי בבית"
        tokenizer = self.make_tokenizer(
            {
                "sentences": [
                    {
                        "tokens": [
                            {
                                "text": "ראיתי",
                                "lemma": "ראה",
                                "upos": "VERB",
                                "deprel": "root",
                                "head": 0,
                                "dspan": [0, 5],
                            },
                            {
                                "text": "בבית",
                                "dspan": [6, 10],
                                "expanded": [
                                    {
                                        "text": "ב",
                                        "lemma": "ב",
                                        "upos": "ADP",
                                        "deprel": "case",
                                        "head": 3,
                                        "dspan": [6, 7],
                                    },
                                    {
                                        "text": "הבית",
                                        "lemma": "בית",
                                        "upos": "NOUN",
                                        "deprel": "obl",
                                        "head": 1,
                                        "dspan": [7, 10],
                                    },
                                ],
                            },
                        ]
                    }
                ]
            }
        )

        doc = tokenizer(text)

        self.assertEqual(doc.text, text)
        self.assertEqual([token.text for token in doc], ["ראיתי", "בבית"])
        self.assertEqual(doc[1].lemma_, "בית")
        self.assertEqual(doc[1].pos_, "NOUN")
        self.assertEqual(doc[1].head, doc[0])
        self.assertEqual(
            [word["text"] for word in doc[1]._.trankit_expanded],
            ["ב", "הבית"],
        )

    def test_expands_mwt_when_expansion_matches_surface(self):
        tokenizer = self.make_tokenizer(
            {
                "sentences": [
                    {
                        "tokens": [
                            {
                                "text": "abc",
                                "dspan": [0, 3],
                                "expanded": [
                                    {
                                        "text": "a",
                                        "lemma": "a",
                                        "upos": "X",
                                        "deprel": "dep",
                                        "head": 2,
                                        "dspan": [0, 1],
                                    },
                                    {
                                        "text": "bc",
                                        "lemma": "bc",
                                        "upos": "X",
                                        "deprel": "root",
                                        "head": 0,
                                        "dspan": [1, 3],
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        )

        doc = tokenizer("abc")

        self.assertEqual(doc.text, "abc")
        self.assertEqual([token.text for token in doc], ["a", "bc"])
        self.assertIsNone(doc[0]._.trankit_expanded)


@unittest.skipUnless(
    os.environ.get("SPACY_TRANKIT_INTEGRATION")
    and importlib.util.find_spec("trankit"),
    "set SPACY_TRANKIT_INTEGRATION=1 and install trankit to run model tests",
)
class TestPipelineIntegration(unittest.TestCase):
    def test_pipeline_1(self):
        nlp = spacy_trankit.load("en")
        text = "Barack Obama was born in Hawaii. He was elected president in 2008."
        doc = nlp(text)
        self.assertEqual(len(doc), 14)

        self.assertTrue(hasattr(doc[0], "lemma_"))
        self.assertTrue(hasattr(doc[0], "pos_"))
        self.assertTrue(hasattr(doc[0], "dep_"))
        self.assertTrue(hasattr(doc[0], "ent_type_"))


if __name__ == "__main__":
    unittest.main()
