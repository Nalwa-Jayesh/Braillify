# src/utils.py
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import subprocess

def download_web_pages(url_list, output_dir="data/raw/html"):
    """
    Downloads HTML content from a list of URLs and saves each as a separate .html file.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, url in enumerate(url_list):
        try:
            response = requests.get(url)
            response.raise_for_status()
            filename = f"webpage_{i+1}.html"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded: {url} -> {filename}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def extract_visible_text_from_html(html_path):
    """
    Extracts visible text from an HTML file using BeautifulSoup.
    """
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def download_images_from_urls(image_urls, output_dir="data/raw/images"):
    """
    Downloads image files from a list of URLs and saves them in the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
            filename = f"image_{i+1}{ext}"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded image: {url} -> {filename}")
        except Exception as e:
            print(f"Failed to download image {url}: {e}")

def convert_to_braille(text, table="en-us-g2.ctb"):
    """
    Converts plain text to Braille using the `lou_translate` CLI tool from Liblouis.
    """
    try:
        result = subprocess.run(
            ["lou_translate", table, "-"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        print(f"Liblouis error: {e.stderr.decode()}")
        return None

# Optional test block
if __name__ == "__main__":
    sample_urls = [
        "https://www.gutenberg.org/files/1342/1342-h/1342-h.htm",
        "https://www.gutenberg.org/files/11/11-h/11-h.htm"
    ]
    sample_image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/3/38/Braille6dots.svg",
        "https://upload.wikimedia.org/wikipedia/commons/d/da/Braille_cell_6_dots.svg"
    ]

    download_web_pages(sample_urls)
    download_images_from_urls(sample_image_urls)

    sample_text = "This is a Braille test."
    braille_output = convert_to_braille(sample_text)
    if braille_output:
        print(f"Braille: {braille_output}")
