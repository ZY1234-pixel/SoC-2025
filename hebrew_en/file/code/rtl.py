import re


RTL_CHAR_RANGES = (
    ("\u0590", "\u05FF"),  # Hebrew
    ("\u0600", "\u08FF"),  # Arabic and related RTL blocks
)


def _is_rtl_char(char):
    return any(start <= char <= end for start, end in RTL_CHAR_RANGES)


def _classify_token(token):
    rtl_count = sum(1 for char in token if _is_rtl_char(char))
    ltr_count = sum(1 for char in token if char.isascii() and char.isalpha())
    if rtl_count and rtl_count >= ltr_count:
        return "rtl"
    if ltr_count:
        return "ltr"
    return "neutral"


def _resolve_token_types(tokens):
    token_types = [_classify_token(token) for token in tokens]
    resolved_types = [token_type if token_type != "neutral" else None for token_type in token_types]
    for index, token_type in enumerate(token_types):
        if token_type != "neutral":
            continue
        previous_type = next(
            (resolved_types[i] for i in range(index - 1, -1, -1) if resolved_types[i]),
            None,
        )
        next_type = next(
            (resolved_types[i] for i in range(index + 1, len(tokens)) if resolved_types[i]),
            None,
        )
        resolved_types[index] = previous_type or next_type or "ltr"
    return resolved_types


def _reverse_rtl_group(text):
    # Keep leading punctuation in place so strings like ",תא שגופ אוה" become ",הוא פוגש את".
    match = re.match(r"^([^\w\u0590-\u05FF\u0600-\u06FF]*)(.*)$", text)
    prefix, remainder = match.groups()
    return prefix + remainder[::-1]


def reverse_reversed_rtl_text(text):
    if not text or not any(_is_rtl_char(char) for char in text):
        return text

    tokens = text.split(" ")
    resolved_types = _resolve_token_types(tokens)

    normalized_groups = []
    group_start = 0
    while group_start < len(tokens):
        group_end = group_start + 1
        while (
            group_end < len(tokens)
            and resolved_types[group_end] == resolved_types[group_start]
        ):
            group_end += 1
        group_text = " ".join(tokens[group_start:group_end])
        if resolved_types[group_start] == "rtl":
            normalized_groups.append(_reverse_rtl_group(group_text))
        else:
            normalized_groups.append(group_text)
        group_start = group_end

    return " ".join(normalized_groups)
