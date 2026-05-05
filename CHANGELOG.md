# Changelog

## 0.2.1

- Publish patch release to recover from the removed 0.2.0 artifacts on PyPI.
- Use a single `ci.yml` workflow for both tests and trusted publishing to PyPI.
- Keep the Python 3.9 CI lane compatible by pinning `spacy<3.8` there.

## 0.2.0

- Add `mwt_strategy="auto"` for Trankit multiword-token expansion: expand
  aligned Arabic/Hebrew clitics and preserve unalignable surface forms.
- Add `mwt_strategy="preserve"` and `mwt_strategy="expand"` escape hatches.
- Store collapsed Trankit expansions on `token._.trankit_expanded`.
- Set sentence starts from Trankit sentence boundaries so `doc.sents` works.
- Make Trankit a lazy import so the package can be imported and unit-tested
  without downloading runtime models.
- Pre-fetch Trankit pretrained models from the HuggingFace mirror
  (`huggingface.co/uonlp/trankit`) before instantiating the Trankit pipeline,
  bypassing the broken `nlp.uoregon.edu` download path baked into the Trankit
  PyPI release. Override the source via the `SPACY_TRANKIT_MODEL_URL`
  environment variable.
- Alias `torch.optim.AdamW` onto `transformers` at import time so Trankit can
  load on `transformers>=4.46`, where `transformers.AdamW` was removed.
- Add fast unit tests, package build checks, GitHub CI, and PyPI publishing
  workflow.
