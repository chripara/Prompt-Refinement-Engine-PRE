"""US-PRE-E01-S01 — checkpoint-specific token conventions."""
from pre.core.resource_loader import get_checkpoint_conventions


def test_known_checkpoint_returns_specific_conventions():
    sdxl = get_checkpoint_conventions("sdxl-base-1.0")
    juggernaut = get_checkpoint_conventions("juggernaut-xl")

    assert "SDXL" in sdxl["style_notes"]
    assert "hotorealistic" in juggernaut["style_notes"]
    assert sdxl["style_notes"] != juggernaut["style_notes"]
    assert sdxl["negative_defaults"]
    assert juggernaut["negative_defaults"]


def test_unknown_checkpoint_falls_back_to_generic_default():
    conventions = get_checkpoint_conventions("some-checkpoint-nobody-registered")

    assert conventions["style_notes"]
    assert conventions["negative_defaults"]
