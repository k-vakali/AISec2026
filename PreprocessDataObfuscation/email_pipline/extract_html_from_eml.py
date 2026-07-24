import os
import email
from email.policy import default
import logging
from tqdm import tqdm  # Imported but not used in this script

logging.basicConfig(level=logging.INFO)


def extract_html_from_eml(eml_file_path, output_dir="test_outputs"):
    """
    Extract HTML content from a .eml file and save it as a .html file.

    Expected input path format:
        YEAR/MONTH/FILENAME.eml

    Example:
        2024/01/example.eml

    The script will read from:
        spamfiles/2024/01/example.eml

    And save to:
        htmls/2024/example.html

    :param eml_file_path: Relative path to the .eml file, such as "2024/01/email.eml"
    :return: True if HTML was successfully extracted and saved, otherwise False
    """

    # # Split the relative file path into year, month, and filename.
    # # This assumes the path has exactly three parts.
    # year, month, bsname = eml_file_path.split(os.sep)

    # # Separate the filename from its extension.
    # # Example: "email.eml" -> filename="email", extension=".eml"
    # filename, extension = os.path.splitext(bsname)

    # # Build the output path for the extracted HTML file.
    # # Note: this ignores the month in the output path.
    # output_html_path = os.path.join("htmls", year, filename + ".html")

    # # Build the full input path by adding the base spamfiles directory.
    # eml_file_path = os.path.join("spamfiles", eml_file_path)

    filename = os.path.splitext(os.path.basename(eml_file_path))[0]

    #output_dir = 'test_outputs'
    os.makedirs(output_dir, exist_ok=True)

    output_html_path = os.path.join(output_dir, filename + '.html')




    try:
        # Read and parse the email file as bytes.
        with open(eml_file_path, "rb") as f:
            msg = email.message_from_bytes(f.read(), policy=default)

        # Collect all HTML parts found in the email.
        html_parts = []

        for part in msg.walk():
            if part.get_content_type() == "text/html":
                # Detect the character encoding used by this email part.
                # If no charset is declared, fall back to UTF-8.
                charset = part.get_content_charset() or "utf-8"

                try:
                    # Decode the HTML content using the detected charset.
                    # Invalid characters are replaced instead of crashing.
                    html_content = part.get_payload(decode=True).decode(
                        charset,
                        errors="replace"
                    )
                    html_parts.append(html_content)

                except UnicodeDecodeError:
                    # If decoding fails, retry with UTF-8.
                    logging.warning(
                        f"Decoding failed with charset {charset}; using replacement characters."
                    )
                    html_content = part.get_payload(decode=True).decode(
                        "utf-8",
                        errors="replace"
                    )
                    html_parts.append(html_content)

        # Save the extracted HTML content if any HTML parts were found.
        if html_parts:
            # Ensure the output directory exists before writing the file.
            os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

            with open(output_html_path, "w", encoding="utf-8") as f:
                f.write("\n".join(html_parts))

            logging.info(
                f"Successfully saved {len(html_parts)} HTML part(s) to {output_html_path}"
            )
            return True

        else:
            logging.warning("No HTML content found.")
            return False

    except FileNotFoundError:
        logging.error(f"File does not exist: {eml_file_path}")
        return False

    except PermissionError:
        logging.error("Insufficient permissions to read or write the file.")
        return False

    except ValueError:
        logging.error(
            "Invalid eml_file_path format. Expected: YEAR/MONTH/FILENAME.eml"
        )
        return False

    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        return False


if __name__ == "__main__":
    # Example usage:
    #extract_html_from_eml(os.path.join("2024", "01", "example.eml"))
    extract_html_from_eml('test_eml/google.eml')