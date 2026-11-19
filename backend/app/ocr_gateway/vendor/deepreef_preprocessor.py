# ---------------------------------------------------------------------------
# REPLICATED COPY -- DO NOT EDIT. Re-copy from the source to update.
# Source: deepreef-ocr  deepreef_ocr/preprocessor.py
#         commit 48ee77836c07191dd1b2f98628e17d802097e9c4
# Everything below this header is the source file verbatim (LF line endings;
# sha256 of the body = 6e42805a3c3a46715f448e7d76df0b007c450b3665fc8d570382f7b932471a21).
# Used by app/ocr_gateway/english_engine.py (decode_and_resize) so English OCR
# does not depend on an external deepreef-ocr checkout.
# ---------------------------------------------------------------------------
"""
DeepReef-OCR Pre-processing Pipeline.

Redesigned around one goal only: make RapidOCR's text extraction on real
procurement-center audit photos as accurate, fast, simple, and clean as
possible. Every function here earns its place with evidence measured
against the real 278-image dataset (F:\\images, not committed to git) --
not because it's a "standard" image-enhancement technique.

Two functions from the previous version were REMOVED, not just disabled,
because they cost real compute on every single image and delivered zero
measured benefit on real photos (only ever worked on clean synthetic test
fixtures):
  - is_digital_screenshot_or_ui: 0% recall on 14 real screenshot photos --
    no threshold on its signal ever separated real screenshots from
    ordinary photos.
  - correct_small_skew: 0% recall on 19 real visibly-tilted photos -- real
    background clutter swamped its edge-based angle detection every time.
If a real fix for either is ever found (most likely: using an actual text
detector, not a generic edge/gradient heuristic), it belongs back here with
fresh evidence, not restored on assumption.

STEP ORDER IS DELIBERATE AND MUST NOT BE REARRANGED CASUALLY:
  1. Validate + decode                     (bytes -> pixels)
  2. Correct EXIF-embedded orientation      -- MUST run before every step below
  3. Downscale if oversized                 -- both resolution
  4. Upscale if undersized                  -- changes happen HERE, together
  5. Mask camera-app overlay band (optional)
  6. Measure raw quality signals (blur, illumination unevenness)
  7. Illumination normalization
  8. Contrast enhancement (CLAHE)
  9. Grayscale conversion (optional, experimental)
  10. Sharpening
  11. Compression re-encode (optional)      -- LAST, always

Orientation correction runs immediately after decode, before anything else,
because every later step either measures or edits a specific REGION of the
frame (the overlay band is "the bottom of the frame"; illumination is
measured on a spatial grid) -- none of that is meaningful until the photo is
actually right-side-up. It is EXIF-tag driven ONLY: it fixes a photo a phone
camera correctly tagged but did not physically re-encode upright, which is
the common real-world case. It does NOT detect or fix a photo whose raw
pixels are genuinely rotated with no EXIF tag present (e.g. re-saved by an
upload pipeline that stripped metadata) -- that is a content-based problem,
and this project already tried and rejected a classical heuristic for a
related problem (see correct_small_skew above); a fix for that class belongs
here only once validated against real evidence, not by assumption.

Steps 3-4 are grouped immediately after orientation-correction so resolution
is finalized ONCE and never touched again. Compression is last on purpose:
it is a lossy byte-fidelity change, and running any further processing on a
compressed image would process compression artifacts (e.g. sharpening
would amplify JPEG block edges). Nothing after compression may exist.
"""
from __future__ import annotations

import logging
import struct
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

log = logging.getLogger(__name__)

# --- Constants -------------------------------------------------------------
DEFAULT_MAX_DIM = 1536
# Below this, DBNet-style text detectors start losing small text to too few
# pixels per character -- a starting point, not yet tuned against real OCR
# output (that needs the actual model). Upscale instead of trying to find
# and crop to the sign: an ROI/crop detector was prototyped and rejected
# (see project history) after it wrongly cropped 100/273 real images.
DEFAULT_MIN_DIM = 900

# Was 85.0, then 150.0 -- see previous tuning history. Kept at 150.0: this
# is an imperfect, content-dependent signal (Laplacian variance), not a
# precise one. Recall/false-positive numbers must be re-validated whenever
# this pipeline's order or steps change, since blur is now measured after
# overlay-masking too.
BLUR_THRESHOLD = 150.0

# Many real photos in this dataset carry a "GPS Map Camera"-style app
# overlay burned into the bottom of the frame (timestamp/coordinates box).
# That is synthetic text a real text detector could latch onto as a false
# region. This fraction of the frame height, measured from the bottom, is
# blanked out (flat-filled, not cropped, so dimensions never change) when
# enable_overlay_mask=True. UNVALIDATED against real OCR output yet -- off
# by default until tested; a blunt fixed-band approach was chosen over
# trying to *detect* the overlay because a brightness/edge-based region
# detector was already tried and rejected for the crop feature (same
# failure mode: no reliable way to tell "watermark" from "real content"
# with classical heuristics alone).
DEFAULT_OVERLAY_BAND_FRACTION = 0.07

# Near-lossless by design. This is meant to be swept during testing (e.g.
# 95 -> 80 -> 60...) to find where OCR accuracy starts to degrade -- 95 is
# just a safe starting point, not a proven optimum.
DEFAULT_COMPRESSION_QUALITY = 95


# --- Individual functions ---------------------------------------------------

def verify_image_magic_bytes(image_bytes: bytes) -> str:
    """
    Validates true file format from binary headers (magic bytes) instead of file extensions.
    """
    if len(image_bytes) < 8:
        raise ValueError("Payload too small to constitute a valid image file.")

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"

    return "unknown"


def read_exif_orientation(image_bytes: bytes) -> int:
    """
    Reads the EXIF Orientation tag (1-8) directly out of a JPEG's APP1
    segment, using only stdlib `struct` -- no PIL/Pillow dependency added
    for this. Returns 1 ("normal", i.e. no rotation needed) for PNG/WEBP
    (no EXIF orientation tag exists in this pipeline for those), a JPEG
    with no EXIF block, or anything this parser doesn't recognize -- a
    safe default, not a detected orientation. MEASURED 2026-08-24: verified
    against PIL's own ImageOps.exif_transpose for all 8 orientation values,
    see the note on correct_orientation below.
    """
    if len(image_bytes) < 4 or image_bytes[0:2] != b"\xff\xd8":
        return 1

    pos, size = 2, len(image_bytes)
    while pos + 4 <= size:
        if image_bytes[pos] != 0xFF:
            break
        marker = image_bytes[pos + 1]
        if marker == 0x01 or (0xD0 <= marker <= 0xD8):
            pos += 2
            continue
        if marker in (0xD9, 0xDA):  # EOI / SOS -- image data starts, stop looking
            break
        seg_len = struct.unpack(">H", image_bytes[pos + 2: pos + 4])[0]
        seg_start = pos + 4
        if marker == 0xE1 and image_bytes[seg_start:seg_start + 6] == b"Exif\x00\x00":
            return _orientation_from_tiff(image_bytes[seg_start + 6: pos + 2 + seg_len])
        pos += 2 + seg_len
    return 1


def _orientation_from_tiff(tiff: bytes) -> int:
    """Reads the Orientation tag (0x0112) out of a TIFF/EXIF IFD0 block."""
    if len(tiff) < 8 or tiff[0:2] not in (b"II", b"MM"):
        return 1
    endian = "<" if tiff[0:2] == b"II" else ">"
    ifd0_offset = struct.unpack(endian + "I", tiff[4:8])[0]
    if ifd0_offset + 2 > len(tiff):
        return 1
    n_entries = struct.unpack(endian + "H", tiff[ifd0_offset:ifd0_offset + 2])[0]
    entry_start = ifd0_offset + 2
    for i in range(n_entries):
        entry = tiff[entry_start + i * 12: entry_start + i * 12 + 12]
        if len(entry) < 12:
            break
        tag = struct.unpack(endian + "H", entry[0:2])[0]
        if tag == 0x0112:  # Orientation
            value = struct.unpack(endian + "H", entry[8:10])[0]
            return value if 1 <= value <= 8 else 1
    return 1


def correct_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """
    Applies the geometric transform an EXIF Orientation value (1-8) implies,
    so every step after this one always sees the photo right-side-up.

    This function, together with decoding via cv2.IMREAD_IGNORE_ORIENTATION
    (see decode_and_resize), REPLACES relying on OpenCV's own default EXIF
    auto-rotation. MEASURED 2026-08-24: cv2.imdecode(..., IMREAD_COLOR) with
    no extra flag already silently applies the exact same correction on this
    opencv-python-headless==4.10.0.84 build -- verified byte-for-byte
    identical to PIL's ImageOps.exif_transpose for all 8 orientation values.
    That means this function changes NOTHING about what a photo with an
    EXIF tag looks like after processing; it makes an already-happening,
    version-and-build-dependent default into an explicit, tested,
    independently-toggleable step (enable_orientation_correction) whose
    result is reported in the metadata contract -- explicit beats implicit,
    same reasoning as rec_keys_path in engine.py.
    """
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.transpose(image)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(image), -1)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image  # 1, or unrecognized -- no-op


def estimate_blur_score(gray_image: np.ndarray) -> float:
    """
    Calculates sharpness metric using the variance of the 2D Laplacian operator.
    Content-dependent (a gravel/grain close-up scores "sharp" by texture alone
    regardless of focus) -- a real but imperfect signal, not a precise one.
    """
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def estimate_illumination_unevenness(gray_image: np.ndarray, grid: int = 8) -> float:
    """
    Measures how uneven the lighting is: splits the image into a grid x grid block
    layout, takes each block's mean brightness, and returns the standard deviation
    across those block means. A shadowed or glare-heavy photo has some blocks much
    brighter/darker than others -> high value. A flatly-lit photo -> low value.
    """
    h, w = gray_image.shape[:2]
    rows = np.array_split(np.arange(h), grid)
    cols = np.array_split(np.arange(w), grid)
    block_means = [
        gray_image[r[0]:r[-1] + 1, c[0]:c[-1] + 1].mean()
        for r in rows if len(r) for c in cols if len(c)
    ]
    return float(np.std(block_means))


def normalize_illumination_and_glare(image: np.ndarray) -> np.ndarray:
    """
    Morphological background estimation and flat-field division.
    Removes severe shadows, flash glares, and outdoor sun gradients without blowing highlights.
    """
    channels = cv2.split(image)
    normalized_channels = []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))

    for ch in channels:
        dilated = cv2.dilate(ch, kernel)
        bg = cv2.medianBlur(dilated, 21)

        norm = cv2.divide(ch.astype(np.float32), bg.astype(np.float32) + 1e-6) * 255.0
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        normalized_channels.append(norm)

    return cv2.merge(normalized_channels)


def enhance_contrast_cielab(image: np.ndarray, clip_limit: float = 2.2) -> np.ndarray:
    """
    Applies CLAHE strictly to the Lightness (L*) channel in CIELAB color space.
    Enhances weathered/faded Devanagari text on walls/boards without creating chromatic
    artifacts. Kept in color space deliberately: converting to grayscale first would
    throw away color contrast that some real signboards depend on (e.g. red text on a
    green background can have very similar brightness but very different hue).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    merged_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def apply_adaptive_unsharp_mask(image: np.ndarray, strength: float = 1.3) -> np.ndarray:
    """
    Applies high-frequency edge accentuation to enhance Devanagari 'shirorekha' (headline)
    and fine vowel diacritics (matras). Must run AFTER blur is measured (see
    decode_and_resize): it always inflates Laplacian variance regardless of the input's
    real sharpness, which is why blur/quality diagnostics are captured earlier in the
    pipeline, not after this step.
    """
    gaussian_blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian_blur, -strength, 0)
    return sharpened


def normalize_resolution(
    image: np.ndarray, max_dim: int = DEFAULT_MAX_DIM, min_dim: int = DEFAULT_MIN_DIM
) -> Tuple[np.ndarray, bool, bool]:
    """
    ONE coordinated resize decision, not two sequential ones. Downscaling to
    max_dim and then separately upscaling the shorter side back up to min_dim
    is a real bug, not a hypothetical: on an extreme-aspect image (e.g. a wide
    banner crop, 2000x300), downscaling to max_dim=1536 first leaves a short
    side of ~230px, which then gets upscaled past min_dim=900 -- producing a
    FINAL image (6010x900) bigger on its long axis than the 2000px original,
    exactly the "size changes again after another step" problem to avoid.

    Instead: compute what downscaling to max_dim would leave the short side
    at, and only apply the min_dim floor on top of THAT projected value, as
    one single combined scale factor. On ordinary (non-extreme-aspect) real
    photos this reduces to the same result as doing each independently; it
    only changes behavior for pathological aspect ratios, where the min_dim
    floor deliberately wins (legibility over the compute-cost cap) and the
    long axis may end up bigger than max_dim -- rare, and intentional.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image, False, False

    scale_down = max_dim / float(max(h, w)) if max(h, w) > max_dim else 1.0
    projected_short_side = min(h, w) * scale_down
    scale_up = min_dim / projected_short_side if projected_short_side < min_dim else 1.0
    final_scale = scale_down * scale_up

    if final_scale == 1.0:
        return image, False, False

    new_w, new_h = int(round(w * final_scale)), int(round(h * final_scale))
    interpolation = cv2.INTER_AREA if final_scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, final_scale < 1.0, final_scale > 1.0


def mask_camera_overlay_band(image: np.ndarray, band_fraction: float = DEFAULT_OVERLAY_BAND_FRACTION) -> np.ndarray:
    """
    Flat-fills the bottom band_fraction of the frame with the average color of the
    row just above it, to suppress a burned-in camera-app overlay (timestamp/GPS
    coordinates box) without changing image dimensions. UNVALIDATED -- see the
    DEFAULT_OVERLAY_BAND_FRACTION comment; only takes effect when explicitly enabled.
    """
    h, w = image.shape[:2]
    band_h = int(round(h * band_fraction))
    if band_h <= 0 or band_h >= h:
        return image
    fill_color = image[h - band_h - 1, :, :].mean(axis=0)
    out = image.copy()
    out[h - band_h:, :, :] = fill_color
    return out


def compress_and_reencode(image: np.ndarray, quality: int = DEFAULT_COMPRESSION_QUALITY) -> Tuple[np.ndarray, int]:
    """
    Re-encodes the final image as JPEG at the given quality and decodes it back,
    so the returned array actually reflects that compression level's real artifacts
    (not just a size number) -- for deliberately sweeping quality during testing to
    find where OCR accuracy starts to degrade. Must run LAST: see module docstring.
    """
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return image, 0
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded, len(encoded)


# --- Master pipeline ---------------------------------------------------------

def decode_and_resize(
    image_bytes: bytes,
    max_dim: int = DEFAULT_MAX_DIM,
    min_dim: int = DEFAULT_MIN_DIM,
    enable_orientation_correction: bool = True,
    enable_resize: bool = True,
    enable_upscale: bool = False,
    enable_overlay_mask: bool = False,
    overlay_band_fraction: float = DEFAULT_OVERLAY_BAND_FRACTION,
    enable_illumination: bool = True,
    enable_contrast: bool = True,
    clip_limit: float = 2.2,
    enable_grayscale: bool = False,
    enable_sharpen: bool = True,
    sharpen_strength: float = 1.0,
    enable_compression: bool = True,
    compression_quality: int = DEFAULT_COMPRESSION_QUALITY,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Master pipeline entry-point for deepreef-ocr pre-processing.
    Processes raw bytes into an optimal BGR numpy array ready for RapidOCR DBNet/SVTR.

    Every step is independently toggleable and defaults to whatever this project's real-
    data testing has shown to be safe (see each constant's comment for the ones that are
    still unvalidated and therefore default off: enable_overlay_mask, enable_grayscale).
    enable_upscale also defaults off (disabled 2026-08-24) -- pass enable_upscale=True to
    restore the min-dim-floor behavior documented on normalize_resolution.

    Step order is fixed and deliberate -- see the module docstring. In short:
    orientation is corrected immediately after decode, every resolution change
    happens right after that, nothing after that point changes pixel dimensions,
    and any lossy compression happens dead last.
    """
    # 1. Binary validation
    detected_format = verify_image_magic_bytes(image_bytes)
    exif_orientation = read_exif_orientation(image_bytes) if enable_orientation_correction else 1

    # 2. Decode. IMREAD_IGNORE_ORIENTATION forces the RAW, as-encoded pixels
    # regardless of what OpenCV's build would otherwise do by default -- see
    # correct_orientation()'s docstring for why this is deliberately explicit.
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"Unable to decode {detected_format} image payload into memory.")

    # 3. Orientation correction -- MUST run before resolution/overlay-masking;
    # see the module docstring for why. orig_dimensions is captured AFTER this,
    # since that's the true, intended (right-side-up) size of the photo.
    orientation_corrected = False
    if enable_orientation_correction and exif_orientation != 1:
        img = correct_orientation(img, exif_orientation)
        orientation_corrected = True
    orig_h, orig_w = img.shape[:2]

    # 4-5. Resolution finalization -- ONE coordinated decision, here only. See
    # normalize_resolution()'s docstring for why this must not be two separate steps.
    effective_max_dim = max_dim if enable_resize else 10**9
    effective_min_dim = min_dim if enable_upscale else 0
    img, resized_down, resized_up = normalize_resolution(img, max_dim=effective_max_dim, min_dim=effective_min_dim)

    # 6. Camera-overlay masking (optional, unvalidated -- off by default)
    overlay_masked = False
    if enable_overlay_mask:
        img = mask_camera_overlay_band(img, band_fraction=overlay_band_fraction)
        overlay_masked = True

    # 7. Quality diagnostics, measured on the resolution-final, overlay-masked image,
    # BEFORE any enhancement step. Enhancement (illumination, contrast, sharpen) all
    # inflate Laplacian variance regardless of real input sharpness -- measuring here
    # is the only point that reflects genuine input quality.
    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_val = estimate_blur_score(gray_raw)
    illumination_unevenness_before = estimate_illumination_unevenness(gray_raw)

    # 8. Illumination normalization
    if enable_illumination:
        img = normalize_illumination_and_glare(img)
    illumination_unevenness_after = (
        estimate_illumination_unevenness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if enable_illumination else None
    )

    # 9. Contrast enhancement
    if enable_contrast:
        img = enhance_contrast_cielab(img, clip_limit=clip_limit)

    # 10. Grayscale conversion (optional, experimental -- off by default; see
    # enhance_contrast_cielab's docstring for why color is kept by default)
    if enable_grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 11. Sharpening
    if enable_sharpen:
        img = apply_adaptive_unsharp_mask(img, strength=sharpen_strength)

    # 12. Compression re-encode -- LAST, always. Nothing may run after this.
    compressed_size_bytes: Optional[int] = None
    if enable_compression:
        img, compressed_size_bytes = compress_and_reencode(img, quality=compression_quality)

    # 13. Assemble metadata
    metadata: Dict[str, Any] = {
        "original_dimensions": [orig_w, orig_h],
        "processed_dimensions": [img.shape[1], img.shape[0]],
        "format_detected": detected_format,
        "exif_orientation": exif_orientation,
        "steps_applied": {
            "orientation_corrected": orientation_corrected,
            "resized_down": resized_down,
            "resized_up": resized_up,
            "overlay_masked": overlay_masked,
            "illumination": enable_illumination,
            "contrast": enable_contrast,
            "grayscale": enable_grayscale,
            "sharpen": enable_sharpen,
            "compression": enable_compression,
        },
        "blur_score": round(blur_val, 2),
        "quality_flag": "acceptable" if blur_val >= BLUR_THRESHOLD else "flagged_blurry",
        "illumination_unevenness_before": round(illumination_unevenness_before, 2),
        "illumination_unevenness_after": (
            round(illumination_unevenness_after, 2) if illumination_unevenness_after is not None else None
        ),
        "compression_quality": compression_quality if enable_compression else None,
        "compressed_size_bytes": compressed_size_bytes,
        "bytes_in": len(image_bytes),
    }

    return img, metadata
