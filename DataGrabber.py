import requests
from bs4 import BeautifulSoup

def extract_pokepaste_links(urls: list[str], output_file: str):
    """Extract all Poképaste links from multiple web pages and save them to a file."""
    all_pokepaste_links = set()  # use a set to avoid duplicates

    for url in urls:
        try:
            # Fetch page content
            response = requests.get(url)
            response.raise_for_status()
            html = response.text

            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Find and filter links
            links = [a["href"] for a in soup.find_all("a", href=True)]
            pokepaste_links = [link for link in links if "pokepast.es" in link]

            all_pokepaste_links.update(pokepaste_links)
            print(f"✅ Found {len(pokepaste_links)} links on {url}")

        except requests.RequestException as e:
            print(f"⚠️ Failed to fetch {url}: {e}")

    # Write all unique links to file
    with open(output_file, "w", encoding="utf-8") as f:
        for link in sorted(all_pokepaste_links):
            f.write(link + "\n")

    print(f"\n✅ Extracted total {len(all_pokepaste_links)} unique Poképaste links to {output_file}")


source_urls = [
    #2026 Reg H data No Longer works with this format
    # "https://www.nimbasacitypost.com/2025/10/milwaukee-regional-2026.html",
    # "https://www.nimbasacitypost.com/2025/09/pittsburgh-regional-2026.html",
    # "https://www.nimbasacitypost.com/2025/09/frankfurt-regional-2026.html",
    # "https://www.nimbasacitypost.com/2025/10/lille-regional-2026.html",
    # "https://www.nimbasacitypost.com/2025/11/gdansk-regional-2026.html",
    # "https://www.nimbasacitypost.com/2025/09/monterrey-regional-2026.html",
    # "https://www.nimbasacitypost.com/2025/10/belo-horizonte-regional-2026.html",
    # "https://www.nimbasacitypost.com/2025/11/brisbane-regional-2026.html",
    # "https://www.nimbasacitypost.com/2025/11/las-vegas-regional-2026.html",
    # "https://www.nimbasacitypost.com/2025/11/buenos-aires-special-2026.html",
    # "https://www.nimbasacitypost.com/2025/11/latin-america-international-2026.html",
    # "https://www.nimbasacitypost.com/2025/11/stuttgart-regional-2026.html"

    #2027 M-C Data
    "https://www.nimbasacitypost.com/2026/09/baltimore-regional-2027.html",
    "https://www.nimbasacitypost.com/2026/10/louisville-regional-2027.html",
    "https://www.nimbasacitypost.com/2026/09/frankfurt-regional-2027.html",
    "https://www.nimbasacitypost.com/2026/10/nice-regional-2027.html",
    "https://www.nimbasacitypost.com/2026/11/gdansk-regional-2027.html",
    "https://www.nimbasacitypost.com/2026/10/recife-regional-2027.html",
    "https://www.nimbasacitypost.com/2026/10/puebla-regional-2027.html",
    "https://www.nimbasacitypost.com/2026/09/brisbane-regional-2027.html"
]

extract_pokepaste_links(source_urls, "data.txt")
