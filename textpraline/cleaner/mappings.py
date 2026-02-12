# TextPraline mappings: conservative, multilingual-safe.

PUA_BULLETS = {
    "\uf0b7",
    "\uf06e",
    "\uf0a7",
    "\uf0d8",
    "\uf0fc",
}

# 1→1 replacements
TRANSLATE_MAP = {
    # bullets → canonical bullet
    "·": "•",
    "‧": "•",
    "∙": "•",
    # typographic quotes → ASCII
    "“": '"',
    "”": '"',
    "‟": '"',
    "„": '"',
    "’": "'",
    "‘": "'",
    "‚": "'",
    # dashes → hyphen
    "–": "-",
    "—": "-",
    # NBSP → space
    "\u00a0": " ",
}

# Optional: fastest mapping for PUA bullets (avoid replace loops)
PUA_TRANSLATE_MAP = {ch: "•" for ch in PUA_BULLETS}
