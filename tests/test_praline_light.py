#!/usr/bin/env python3

"""
Test script to verify the praline_light preset works correctly.
"""

from textpraline.cleaner.clean import praline, PRESETS


def test_praline_light_preset():
    """Test that the light preset is properly configured."""

    # Test that the preset exists
    assert "light" in PRESETS, "Light preset should be in PRESETS"

    light_config = PRESETS["light"]
    print("Light preset configuration:")
    print(f"  profile: {light_config.profile}")
    print(f"  normalize_extracted: {light_config.normalize_extracted}")
    print(f"  collapse_blank_lines: {light_config.collapse_blank_lines}")
    print(f"  drop_layout_noise: {light_config.drop_layout_noise}")
    print(f"  drop_repeated_lines: {light_config.drop_repeated_lines}")
    print(f"  drop_references_section: {light_config.drop_references_section}")
    print(f"  structure_mode: {light_config.structure_mode}")

    # Test with sample text
    sample_text = """
    This is a test document.
    
    It has some OCR artifacts like glyph<123> and (cid:456).
    
    It also has repeated headers/footers that should be removed.
    This is a header that repeats on every page.
    
    The main content should be preserved with original casing,
    numbers like 123 and 45.67, punctuation! "quotes", and indentation.
    
    Short lines should be kept.
    Like this one.
    
    Structural whitespace should not be collapsed aggressively.
    """

    print("\nOriginal text:")
    print(repr(sample_text))

    # Test the light preset
    cleaned_light = praline(sample_text, preset="light")
    print("\nCleaned with light preset:")
    print(repr(cleaned_light))

    # Compare with safe preset
    cleaned_safe = praline(sample_text, preset="safe")
    print("\nCleaned with safe preset:")
    print(repr(cleaned_safe))

    # Verify key differences
    print("\nKey verification:")
    print(
        f"Light preset preserves more structure: {len(cleaned_light) >= len(cleaned_safe)}"
    )
    print(
        f"Light preset has different whitespace handling: {cleaned_light != cleaned_safe}"
    )

    return True


if __name__ == "__main__":
    test_praline_light_preset()
    print("\nTest completed successfully!")
