import json
import re
import sys
import os
from collections import Counter
from wordfreq import zipf_frequency

# Optional threshold
PARAGRAPH_DISRUPTION_CHAR_THRESHOLD = 80


# COMMON_WORDS = {
#     "dear", "hello", "paypal", "account", "customer",
#     "please", "thanks", "confirm", "verify", "update",
#     "email", "message", "service", "regards", "unsubscribe"
# }


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_space(text):
    return re.sub(r"\s+", " ", text).strip()


def is_text_char(ch):
    return ch.isalpha()

def trailing_alpha(text):
    m = re.search(r"([A-Za-z]+)\s*$", text)
    return m.group(1).lower() if m else ""


def leading_alpha(text):
    m = re.search(r"^\s*([A-Za-z]+)", text)
    return m.group(1).lower() if m else ""

def leading_word(text):
    m = re.search(r"^\s*([A-Za-z]+(?:-[A-Za-z]+)?)", text)
    return m.group(1).lower() if m else ""

def has_word_boundary_after_leading_alpha(text):
    stripped = text.lstrip()
    m = re.match(r"[A-Za-z]+", stripped)

    if not m:
        return False

    rest = stripped[m.end():]
    return bool(re.search(r"\s", rest))

def has_word_boundary_after_leading_word(text):
    stripped = text.lstrip()
    m = re.match(r"[A-Za-z]+(?:-[A-Za-z]+)?", stripped)

    if not m:
        return False

    rest = stripped[m.end():]
    return bool(re.search(r"\s", rest))


# def is_real_word(word):
#     return word.lower() in COMMON_WORDS
LANGUAGES = [
    "en",
    "de",
    "fr",
    "es",
    "it",
    "nl",
    "hu",  # Hungarian
]


def is_real_word(word, min_zipf=3.0):
    word = word.strip().lower()

    if not word:
        return False

    # avoid obvious garbage
    if not re.fullmatch(r"[a-zA-Z]+", word):
        return False

    for lang in LANGUAGES:
        if zipf_frequency(word, lang) >= min_zipf:
            return True

    return False

def is_real_or_hyphenated_word(word, min_zipf=3.0):
    word = word.strip().lower()

    # Normal word
    if is_real_word(word, min_zipf):
        return True

    # Not hyphenated and not a real word
    if "-" not in word:
        return False

    parts = [p for p in word.split("-") if p]

    # Whole word without hyphens
    joined = "".join(parts)
    if is_real_word(joined, min_zipf):
        return True

    # Every component is a real word
    if len(parts) >= 2 and all(
        is_real_word(part, min_zipf)
        for part in parts
    ):
        return True

    return False


def has_internal_space(text):
    stripped = text.strip()
    return bool(re.search(r"\s", stripped))

def starts_with_boundary(text):
    return bool(re.match(r"^\s*[\s\W]", text))


def ends_with_boundary(text):
    return bool(re.search(r"[\s\W]\s*$", text))


def has_left_insert_boundary(prev_visible, hidden_text):
    if prev_visible is None:
        return True

    return (
        ends_with_boundary(prev_visible["text"])
        or starts_with_boundary(hidden_text)
    )


def has_right_insert_boundary(next_visible, hidden_text):
    if next_visible is None:
        return True

    return (
        ends_with_boundary(hidden_text)
        or starts_with_boundary(next_visible["text"])
    )

def classify_non_disrupt_hidden_text(
    hidden_text,
    prev_visible=None,
    next_visible=None
):
    stripped = hidden_text.strip()

    if not stripped:
        return "insert_word"

    if not has_internal_space(hidden_text):

        # Inserted hidden words must be token-separated from
        # surrounding visible text. Boundary characters may appear
        # either in the visible fragment or in the hidden fragment.
        #
        # Example insert_word:
        #   visible "PayPal " + hidden "security" + visible " account"
        #
        # Example NOT insert_word:
        #   hidden "k" + visible "PayPal"
        #   visible "Pal" + hidden "s"
        #
        # TODO:
        # Token-adjacent hidden affixes may later be handled as a
        # disrupt_word subtype, e.g. hidden_affix / brand_padding.
        if not (
            has_left_insert_boundary(prev_visible, hidden_text)
            and has_right_insert_boundary(next_visible, hidden_text)
        ):
            return "add_paragraph"

        if is_real_word(stripped):
            return "insert_word"

        return "add_paragraph"

    return "add_paragraph"


def weak_disrupt_boundary(prev_visible, next_visible):
    if not prev_visible or not next_visible:
        return False

    return bool(
        trailing_alpha(prev_visible["text"])
        and leading_word(next_visible["text"])
    )


def iter_hidden_runs(fragments):
    i = 0

    while i < len(fragments):
        if fragments[i]["visible"]:
            i += 1
            continue

        start = i

        # HTML comments should be classified individually
        if fragments[i]["type"] == "html_comment":
            yield start, start + 1
            i += 1
            continue

        while (
            i < len(fragments)
            and not fragments[i]["visible"]
            and fragments[i]["type"] != "html_comment"
        ):
            i += 1

        end = i
        yield start, end

def get_prev_visible(fragments, start):
    for j in range(start - 1, -1, -1):
        if fragments[j]["visible"] and fragments[j]["clean"]:
            return j, fragments[j]
    return None, None


def get_next_visible(fragments, end):
    for j in range(end, len(fragments)):
        if fragments[j]["visible"] and fragments[j]["clean"]:
            return j, fragments[j]
    return None, None


def is_weak_disrupt_run(fragments, start, end):
    _, prev_visible = get_prev_visible(fragments, start)
    _, next_visible = get_next_visible(fragments, end)

    return weak_disrupt_boundary(prev_visible, next_visible)


#def collect_disrupt_chain(fragments, run_start, run_end):
    """
    Collect connected VIS [HID] VIS [HID] VIS blocks
    while the weak disrupt pattern persists.

    Returns:
        chain_runs: list of hidden-run ranges
        reconstructed_visible_word: visible text pieces joined together
    """

    chain_runs = [(run_start, run_end)]

    prev_idx, prev_visible = get_prev_visible(fragments, run_start)
    next_idx, next_visible = get_next_visible(fragments, run_end)

    if prev_visible is None or next_visible is None:
        return chain_runs, ""

    reconstructed_visible_word = (
        trailing_alpha(prev_visible["text"])
        + leading_alpha(next_visible["text"])
    )
    current_visible_idx = next_idx
    current_visible = next_visible

    while True:
        possible_hidden_start = current_visible_idx + 1

        if possible_hidden_start >= len(fragments):
            break

        if fragments[possible_hidden_start]["visible"]:
            break

        hidden_start = possible_hidden_start

        i = hidden_start
        while i < len(fragments) and not fragments[i]["visible"]:
            i += 1

        hidden_end = i

        next_visible_idx, next_visible = get_next_visible(fragments, hidden_end)

        if next_visible is None:
            break

        if not weak_disrupt_boundary(current_visible, next_visible):
            break

        chain_runs.append((hidden_start, hidden_end))
        reconstructed_visible_word += leading_alpha(next_visible["text"])
        current_visible_idx = next_visible_idx
        current_visible = next_visible

    return chain_runs, reconstructed_visible_word

def collect_disrupt_chain(fragments, run_start, run_end):
    chain_runs = [(run_start, run_end)]

    prev_idx, prev_visible = get_prev_visible(fragments, run_start)
    next_idx, next_visible = get_next_visible(fragments, run_end)

    if prev_visible is None or next_visible is None:
        return chain_runs, ""

    reconstructed_visible_word = trailing_alpha(prev_visible["text"])

    current_visible_idx = next_idx
    current_visible = next_visible

    while current_visible is not None:
        reconstructed_visible_word += leading_word(current_visible["text"])

        # Stop at the end of the visible word.
        # Example: "ted pending" contributes "ted", then stops.
        if has_word_boundary_after_leading_word(current_visible["text"]):
            break

        possible_hidden_start = current_visible_idx + 1

        if possible_hidden_start >= len(fragments):
            break

        if fragments[possible_hidden_start]["visible"]:
            break

        hidden_start = possible_hidden_start

        i = hidden_start
        while i < len(fragments) and not fragments[i]["visible"]:
            i += 1

        hidden_end = i

        next_visible_idx, next_visible = get_next_visible(fragments, hidden_end)

        if next_visible is None:
            break

        if not weak_disrupt_boundary(current_visible, next_visible):
            break

        chain_runs.append((hidden_start, hidden_end))

        current_visible_idx = next_visible_idx
        current_visible = next_visible

    return chain_runs, reconstructed_visible_word


def classify_hidden_run(fragments, start, end):
    run = fragments[start:end]

    # HTML comments special case
    if all(f["type"] == "html_comment" for f in run):
        text = " ".join(f["clean"] for f in run).lower()

        if (
            "if mso" in text
            or "if gte mso" in text
            or "if lte mso" in text
            or "endif" in text
            or "mso" in text
        ):
            return "mso_conditional_comment"

        return "html_comment"

    hidden_text = " ".join(f["clean"] for f in run)
    hidden_tokens = len(hidden_text.split())
    hidden_chars = len(hidden_text)

    if all(f.get("whitespace_only") for f in run):
        return "hidden_whitespace_only"

    # 1. First check disrupt-word structure
    if is_weak_disrupt_run(fragments, start, end):

        if hidden_chars >= PARAGRAPH_DISRUPTION_CHAR_THRESHOLD:
            return "add_paragraph"

        chain_runs, reconstructed_visible_word = collect_disrupt_chain(
            fragments,
            start,
            end
        )

        # print(
        #     "RECONSTRUCTED:",
        #     repr(reconstructed_visible_word),
        #     "HIDDEN:",S
        #     repr(hidden_text)
        # )



        if is_real_or_hyphenated_word(reconstructed_visible_word):
            return "disrupt_word"

    # 2. Not disrupt_word, so classify inserted hidden block
    _, prev_visible = get_prev_visible(fragments, start)
    _, next_visible = get_next_visible(fragments, end)

    return classify_non_disrupt_hidden_text(
        hidden_text,
        prev_visible,
        next_visible
    )


def build_fragments(rows):
    fragments = []

    for idx, row in enumerate(rows):
        text = row.get("text", "")

        if not text:
            continue

        is_whitespace_only = text.strip() == ""

        fragments.append({
            "idx": idx,
            "row": row,
            "text": text,
            "clean": normalize_space(text),
            "visible": row.get("visible", True),
            "type": row.get("type", "text"),
            "whitespace_only": is_whitespace_only,
        })

    return fragments


def classify_embedding_strategy(json_path):
    rows = load_jsonl(json_path)
    fragments = build_fragments(rows)

    hidden_fragments = [f for f in fragments if not f["visible"]]
    visible_fragments = [f for f in fragments if f["visible"]]

    counts = Counter({
        "add_paragraph": 0,
        "disrupt_word": 0,
        "insert_word": 0,
        "html_comment": 0,
        "mso_conditional_comment": 0,
        "hidden_whitespace_only": 0
    })

    examples = {
        "add_paragraph": [],
        "disrupt_word": [],
        "insert_word": [],
        "html_comment": [],
        "mso_conditional_comment": [],
        "hidden_whitespace_only": []
    }

    if not hidden_fragments:
        return {
            "dominant_category": "no_hidden_text",
            "counts": dict(counts),
            "examples": examples,
            "stats": {
                "visible_fragments": len(visible_fragments),
                "hidden_fragments": 0
            }
        }

    for start, end in iter_hidden_runs(fragments):
        category = classify_hidden_run(fragments, start, end)

        for frag in fragments[start:end]:
            counts[category] += 1

            if len(examples[category]) < 5:
                examples[category].append(frag["clean"])

    dominant = counts.most_common(1)[0][0]

    return {
        "dominant_category": dominant,
        "counts": dict(counts),
        "examples": examples,
        "stats": {
            "visible_fragments": len(visible_fragments),
            "hidden_fragments": len(hidden_fragments)
        }
    }


def classify_jsonl_file(input_path, output_path):
    rows = load_jsonl(input_path)
    fragments = build_fragments(rows)

    counts = Counter({
        "add_paragraph": 0,
        "disrupt_word": 0,
        "insert_word": 0,
        "html_comment": 0,
        "mso_conditional_comment": 0,
        "hidden_whitespace_only": 0
    })

    classified_hidden_rows = []

    processed_run_starts = set()

    hidden_runs = list(iter_hidden_runs(fragments))

    for start, end in hidden_runs:
        if start in processed_run_starts:
            continue

        category = classify_hidden_run(fragments, start, end)

        runs_to_output = [(start, end)]

        if category == "disrupt_word":
            chain_runs, _ = collect_disrupt_chain(fragments, start, end)
            runs_to_output = chain_runs

        for run_start, run_end in runs_to_output:
            processed_run_starts.add(run_start)

            for frag in fragments[run_start:run_end]:
                counts[category] += 1

                out_row = dict(frag["row"])
                out_row["embedding_strategy"] = category
                classified_hidden_rows.append(out_row)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in classified_hidden_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_file": input_path,
        "output_file": output_path,
        "counts": dict(counts),
        "stats": {
            "total_fragments": len(fragments),
            "hidden_fragments": len(classified_hidden_rows),
            "visible_fragments": len(fragments) - len(classified_hidden_rows)
        }
    }

    return classified_hidden_rows, summary


if __name__ == "__main__":

    input_path = sys.argv[1]

    output_dir = "test_outputs_classified"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(input_path)
    base_name = base_name.replace(".json", "_classified.jsonl")

    output_path = os.path.join(output_dir, base_name)

    classified_hidden_rows, summary = classify_jsonl_file(
        input_path,
        output_path
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))