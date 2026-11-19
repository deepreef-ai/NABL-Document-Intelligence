# Vendored English OCR recogniser

**Replicated copy — do not edit. To update, re-copy both files from the source and
update the sha256 values here and in `app/ocr_gateway/english_engine.py`.**

Used by `app/ocr_gateway/english_engine.py` (gateway engine `deepreef_local`,
script `en`) so English OCR runs without an external deepreef-ocr checkout.

| File | Bytes | sha256 |
|---|---:|---|
| `devanagari_rec.onnx` | 7,940,361 | `d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e` |
| `devanagari_dict.txt` | 1,943 | `09c7440bfc5477e5c41052304b6b185aff8c4a5e8b2b4c23c1c706f6fe1ee9fc` |

Both hashes match the deepreef-ocr repo's own `models.sha256`.

## Source

- Copied from the deepreef-ocr repo (production OCR), `deepreef_ocr/models/`, at
  commit `48ee77836c07191dd1b2f98628e17d802097e9c4`. In that repo the model files are
  not committed (`.gitignore`: `deepreef_ocr/models/*.onnx`); they are baked into its
  models image and pinned by `models.sha256`.
- Upstream origin, as documented in deepreef-ocr `DEPLOY.md`
  ("The original sources, if a set has to be rebuilt from scratch"):
  - `devanagari_rec.onnx` = RapidOCR's ONNX export of PaddleOCR
    PP-OCRv5 `devanagari_PP-OCRv5_rec_mobile.onnx`, from
    `https://modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/rec/`
  - `devanagari_dict.txt` = PaddleOCR `ppocr/utils/dict/ppocrv5_devanagari_dict.txt`
    (saved under the v4-style name the engine expects).
- Why a Devanagari model reads English: its dictionary holds all 94 printable ASCII
  characters (deepreef-ocr `engine.py`, script `en`, issue #30: 94.6% words vs 34.8%
  for RapidOCR's built-in ch_PP-OCRv4 on 96 rendered English boards).

## Licence

deepreef-ocr's docs do not state a licence for these two files; they only name the
upstream projects (PaddleOCR by Baidu, which trains the models, and RapidOCR by
RapidAI, which converts them to ONNX). Both upstream projects publish under
Apache-2.0 — that is not stated in deepreef-ocr's docs and should be confirmed by
the reviewer before distribution.
