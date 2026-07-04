"""US-PRE-E00-S02 — canonical seed derivation (FR-PRE-017)."""
from pre.core.determinism.canonical import derive_seed, resolve_seed


def test_identical_input_yields_identical_seed():
    a = {"text": "A warrior at night", "checkpoint": "sdxl-base-1.0"}
    b = {"checkpoint": "sdxl-base-1.0", "text": "A warrior at night"}  # different key order
    assert derive_seed(a) == derive_seed(b)


def test_different_input_yields_different_seed():
    a = {"text": "A warrior at night", "checkpoint": "sdxl-base-1.0"}
    b = {"text": "A mage at dawn", "checkpoint": "sdxl-base-1.0"}
    assert derive_seed(a) != derive_seed(b)


def test_seed_is_32_bit_unsigned():
    seed = derive_seed({"text": "anything", "checkpoint": "sdxl-base-1.0"})
    assert 0 <= seed < 2 ** 32


def test_explicit_seed_overrides_derivation():
    data = {"text": "A warrior at night", "checkpoint": "sdxl-base-1.0"}
    derived = derive_seed(data)
    explicit = derived + 1  # any distinct value
    assert resolve_seed(data, explicit) == explicit
    assert resolve_seed(data, explicit) != derived


def test_no_explicit_seed_falls_back_to_derivation():
    data = {"text": "A warrior at night", "checkpoint": "sdxl-base-1.0"}
    assert resolve_seed(data, None) == derive_seed(data)
