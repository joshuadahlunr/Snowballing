from fuzzywuzzy import fuzz
from typing import List
import logging
import asyncio
from aiohttp import ClientSession, TCPConnector
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DOIBodySearcher:
    def __init__(self):
        self.base_url = "https://api.crossref.org/works/"
        self.headers = {'User-Agent': 'YourApp/1.0 (mailto:knightjosephsnow@gmail.com)'}
        self._compiled_keywords = None
        self.semaphore = asyncio.Semaphore(9)
        self.last_request_time = time.time()

    async def get_body(self, session: ClientSession, doi: str) -> str:
        async with self.semaphore:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < 0.11:
                await asyncio.sleep(0.11 - elapsed)
            try:
                async with session.get(f"{self.base_url}{doi}", headers=self.headers) as response:
                    self.last_request_time = time.time()

                    if response.status != 200:
                        return f"Error: HTTP {response.status}"

                    data = await response.json()
                    # Try to get the full text content from different possible fields
                    body = data.get('message', {}).get('full-text-content', '')
                    if not body:
                        body = data.get('message', {}).get('body', '')
                    if not body:
                        body = data.get('message', {}).get('content', '')
                    
                    return body if body else "No body content found for this DOI"
            except Exception as e:
                logging.error(f"Error retrieving body for DOI {doi}: {str(e)}")
                return f"Error retrieving body: {str(e)}"

    def fuzzy_search_keywords(self, text: str, keywords: List[str], threshold: int = 80, accept_threshold: int = 1) -> bool:
        if not text or not keywords:
            return False

        words = set(text.lower().split())
        keywords_set = {k.lower() for k in keywords}
        
        # Count how many keywords match above the threshold
        matched_keywords = sum(
            1 for keyword in keywords_set
            if any(fuzz.ratio(keyword, word) >= threshold for word in words)
        )
        
        # Return True if we found enough matching keywords
        return matched_keywords >= accept_threshold

async def process_dois(dois: List[str],keywords: List[str]) -> None:
    # Keywords specific to compiler implementation details
    searcher = DOIBodySearcher()

    connector = TCPConnector(limit=9)
    async with ClientSession(connector=connector) as session:
        with open('Filtered_Body_DOI.txt', 'w') as output_file:
            for i in range(0, len(dois), 9):
                batch = dois[i:i + 9]
                try:
                    bodies = await asyncio.gather(
                        *[searcher.get_body(session, doi) for doi in batch]
                    )
                    for doi, body in zip(batch, bodies):
                        if searcher.fuzzy_search_keywords(body, keywords):
                            output_file.write(f"{doi}\n")
                        logging.info(f"Processed: {doi}")

                    await asyncio.sleep(0.1)

                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    continue

def read_filtered_dois() -> List[str]:
    try:
        with open('Filtered_Abstract_DOI.txt', 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        logging.error("Filtered_Abstract_DOI.txt not found")
        return []

def main():
    keywords = ["Intermediate Representation", "Code Generation", "Optimization"]
    try:
        filtered_dois = read_filtered_dois()
        total_dois = len(filtered_dois)
        logging.info(f"Found total of {total_dois} DOIs to process")

        asyncio.run(process_dois(filtered_dois, keywords))

        try:
            with open('Filtered_DOI.txt', 'r') as results_file:
                result_dois = len(results_file.readlines())

            print(f"\nProcessing Summary:")
            print(f"Keywords being searched In the Body of DOI: {', '.join(keywords)}")
            print(f"Total DOIs processed: {total_dois}")
            print(f"DOIs matching criteria: {result_dois}")
            print(f"Match rate: {(result_dois/total_dois*100):.2f}%")

        except FileNotFoundError:
            logging.error("Results file not found. No DOIs may have matched the criteria.")
            print("\nProcessing Summary:")
            print(f"Keywords being searched: LLVM, Compiler, Binary, Assembly, Machine Code")
            print(f"Total DOIs processed: {total_dois}")
            print("DOIs matching criteria: 0")
            print("Match rate: 0.00%")

    except Exception as e:
        logging.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()