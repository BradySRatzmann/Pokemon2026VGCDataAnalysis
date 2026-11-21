import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os
import re

from data_collection.team_parser import parse_team_metadata

# Asynchronous helper function to fetch a single Poképaste page
async def fetch_page(session, url: str):
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, "lxml")

                paragraphs = soup.find_all("article")

                # Find the page title
                title_candidates = []

                heading = soup.find("h1")
                if heading:
                    title_candidates.append(heading.get_text(" ", strip=True))

                og_title = soup.find("meta", property="og:title")
                if og_title and og_title.get("content"):
                    title_candidates.append(og_title["content"])

                if soup.title:
                    title_candidates.append(soup.title.get_text(" ", strip=True))

                # Extract metadata from the first matching title
                metadata = {
                    "player_name": "Unknown",
                    "placement": "Unknown",
                    "event_name": "Unknown"
                }

                for title in title_candidates:
                    parsed = parse_team_metadata(title)

                    if parsed["player_name"] != "Unknown":
                        metadata = parsed
                        break

                return {
                    "paragraphs": [
                        para.get_text("\n") for para in paragraphs
                    ],
                    "metadata": metadata
                }

                return {
                    "paragraphs": [
                        para.get_text("\n") for para in paragraphs
                    ],
                    "metadata": metadata
                }

            else:
                print(f"⚠️ Failed to fetch {url} (status {resp.status})")
                return {"paragraphs": [], "metadata": {}}

    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return {"paragraphs": [], "metadata": {}}


# Function to concurrently fetch multiple Poképaste pages
async def read_all_urls(urls: list[str]):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, url) for url in urls]
        return await asyncio.gather(*tasks)