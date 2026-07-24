import argparse
import re
from pathlib import Path
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup


KEEP_HEADERS = ["From", "To", "Subject"]


def parse_email(path):
    with open(path, "rb") as f:
        return BytesParser(policy=policy.default).parse(f)


def minimal_headers(msg):
    return "\n".join(
        f"{h}: {msg.get(h, '')}" for h in KEEP_HEADERS if msg.get(h)
    ) + "\n\n"


def decode_part(part):
    payload = part.get_payload(decode=True)

    if payload is None:
        raw = part.get_payload()
        return raw if isinstance(raw, str) else ""

    charset = part.get_content_charset() or "utf-8"

    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def attachment_summary(part):
    filename = part.get_filename() or "unknown"
    content_type = part.get_content_type()
    payload = part.get_payload(decode=True)
    size = len(payload) if payload else 0
    return f"[Attachment removed: {filename} | {content_type} | {size} bytes]"


def part_headers_without_encoding(part):
    lines = []

    for key, value in part.items():
        key_lower = key.lower()

        if key_lower == "content-transfer-encoding":
            continue

        lines.append(f"{key}: {value}")

    return "\n".join(lines)


def render_decoded_mime_body(part):
    if part.is_multipart():
        boundary = part.get_boundary()
        children = list(part.iter_parts())

        if not boundary:
            return "\n\n".join(render_decoded_mime_body(child) for child in children)

        output = []

        for child in children:
            output.append(f"--{boundary}")
            child_headers = part_headers_without_encoding(child)

            if child_headers:
                output.append(child_headers)
                output.append("")

            output.append(render_decoded_mime_body(child))

        output.append(f"--{boundary}--")
        return "\n".join(output)

    disposition = part.get_content_disposition()

    if disposition in ("attachment", "inline") and part.get_filename():
        return attachment_summary(part)

    content_type = part.get_content_type()

    if content_type in ("text/plain", "text/html"):
        return decode_part(part)

    return attachment_summary(part)


def make_p1_output(msg):
    return minimal_headers(msg) + render_decoded_mime_body(msg).strip() + "\n"


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "meta", "link", "noscript", "svg", "img", "title"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def collect_text_parts(msg):
    plain_parts = []
    html_parts = []
    attachments = []

    for part in msg.walk():
        if part.is_multipart():
            continue

        disposition = part.get_content_disposition()
        content_type = part.get_content_type()

        if disposition in ("attachment", "inline") and part.get_filename():
            attachments.append(attachment_summary(part))
            continue

        if content_type == "text/html":
            html_parts.append(decode_part(part))
        elif content_type == "text/plain":
            plain_parts.append(decode_part(part))

    return plain_parts, html_parts, attachments


def make_p2_output(msg):
    plain_parts, html_parts, attachments = collect_text_parts(msg)

    if html_parts:
        body = clean_html("\n\n".join(html_parts))
    else:
        body = "\n\n".join(plain_parts).strip()

    output = minimal_headers(msg) + body

    if attachments:
        output += "\n\nAttachments:\n" + "\n".join(attachments)

    return output.strip() + "\n"


def process_folder(input_dir, output_root):
    input_dir = Path(input_dir)
    output_root = Path(output_root)

    dataset_name = input_dir.name
    dataset_output = output_root / f"{dataset_name}_POut"

    p1_dir = dataset_output / "P1"
    p2_dir = dataset_output / "P2"

    p1_dir.mkdir(parents=True, exist_ok=True)
    p2_dir.mkdir(parents=True, exist_ok=True)

    for eml_path in input_dir.glob("*.eml"):
        try:
            msg1 = parse_email(eml_path)
            msg2 = parse_email(eml_path)

            p1_text = make_p1_output(msg1)
            p2_text = make_p2_output(msg2)

            (p1_dir / f"{eml_path.stem}_P1.txt").write_text(
                p1_text,
                encoding="utf-8"
            )

            (p2_dir / f"{eml_path.stem}_P2.txt").write_text(
                p2_text,
                encoding="utf-8"
            )

            print(f"Processed: {eml_path.name}")

        except Exception as e:
            print(f"Failed: {eml_path.name} | {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()