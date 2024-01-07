#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import spacy_trankit


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy_trankit.load("en")

    def test_pipeline_1(self):
        text = "Barack Obama was born in Hawaii. He was elected president in 2008."
        doc = self.nlp(text)
        self.assertEqual(len(doc), 14)

        self.assertTrue(hasattr(doc[0], "lemma_"))
        self.assertTrue(hasattr(doc[0], "pos_"))
        self.assertTrue(hasattr(doc[0], "dep_"))
        self.assertTrue(hasattr(doc[0], "ent_type_"))


if __name__ == "__main__":
    unittest.main()
