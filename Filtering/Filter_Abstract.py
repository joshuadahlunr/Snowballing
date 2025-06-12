from fuzzywuzzy import fuzz
from typing import List, Dict, Set
import logging
from pathlib import Path
import asyncio
from aiohttp import ClientSession, TCPConnector
import time

ONE_SECOND = 1
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DOIAbstractSearcher:
    def __init__(self):
        self.base_url = "https://api.crossref.org/works/" # The API endpoint
        self.headers = {'User-Agent': 'YourApp/1.0 (mailto:knightjosephsnow@gmail.com)'} # If you want to be emailed
        self._compiled_keywords = None # Compile keywords once and reuse
        self.semaphore = asyncio.Semaphore(9)  # Limit concurrent requests to 9
        self.last_request_time = time.time() # To help with filtering



    async def get_abstract(self, session: ClientSession, doi: str) -> str:
        """
        - This is an asynchronous method definition takes 3 parameters
        :param self: ( class instance)
        :param session: `session` ( for HTTP requests)
        :param doi: and `doi` (the DOI string)
        :return: `-> str` indicates it returns a string
        """
        async with self.semaphore:  # Uses a semaphore to limit how many concurrent requests can run to prevent overloading the server with too many simultaneous requests
            current_time = time.time()  #Gets current time and calculates how long since last request Used for rate limiting
            elapsed = current_time - self.last_request_time
            if elapsed < 0.11:  # Add small buffer to be safe
                await asyncio.sleep(0.11 - elapsed) # If less than 0.11 seconds passed since last request Waits for remaining time to ensure at least 0.11 seconds between requests
            try:
                # Makes HTTP GET request to the API, which Combines base URL with DOI and Includes headers for authentication
                async with session.get(f"{self.base_url}{doi}", headers=self.headers) as response:
                    self.last_request_time = time.time() # Records when this request was made for rate limiting

                    if response.status != 200: # Checks if request was successful (200 is success code) Returns error message if not successful
                        return f"Error: HTTP {response.status}"
                    # Converts response to JSON, then  safely extracts abstract from nested JSON structure and make sure an empty string if not found
                    data = await response.json()
                    abstract = data.get('message', {}).get('abstract', '')
                    return abstract if abstract else "No abstract found for this DOI"
            # Catches any errors during the process and logs the error with details then returns error message to caller
            except Exception as e:
                logging.error(f"Error retrieving abstract for DOI {doi}: {str(e)}")
                return f"Error retrieving abstract: {str(e)}"

    def compile_keywords(self, keywords: List[str]) -> Set[str]:
        """
        Method definition that takes a list of strings as `keywords`
        :param keywords: Is a list of key word strings to filter DOI's using fuzzywuzzy
        :return: we return those keywords as a set of lowercase strings
        Method definition that takes a list of strings as `keywords` checks if keywords have been compiled before.
        This is a form of caching/memoization only compiles keywords once to improve performance
        """
        # Returns either the newly compiled set or the previously cached set
        if self._compiled_keywords is None:
            self._compiled_keywords = {k.lower() for k in keywords}
        return self._compiled_keywords

    def fuzzy_search_keywords(self, text: str, keywords: List[str], threshold: int = 80) -> bool:
        """
        # Method that performs fuzzy (approximate) text matching.Takes three parameters:
        :param Text: The string to search in
        :param Keywords: List of keywords to search for
        :param Threshold: Minimum similarity score (default 80%)
        :return: Returns a boolean (True/False)
        """
        # Early exit if either text or keywords is empty and prevents processing empty data
        if not text or not keywords:
            return False

        words = set(text.lower().split())
        # Uses the previously explained compile_keywords method and gets lowercase set of keywords to search for
        keywords_set = self.compile_keywords(keywords)
        # Nested loop using `any()` function for each keyword, checks against each word
        # `fuzz.ratio()` calculates similarity between words (0-100) and returns True if any word matches any keyword
        # Above threshold Example: "testing" might match "test" if similarity > 80%
        return any(
            any(fuzz.ratio(keyword, word) >= threshold for word in words)
            for keyword in keywords_set
        )

def read_all_doi_files(directory_path: str) -> List[str]:
    """
    Reads all DOI files from a directory and combines them into a single list.
    Args:
        directory_path (str): Path to the directory containing DOI files  
    Returns:
        List[str]: Combined list of DOIs from all files
    """
    all_dois = []
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Iterate through all .txt files in the directory
    for doi_file in directory.glob('*.txt'):
        try:
            # Read content of each file
            with doi_file.open('r') as file:
                # Read lines, strip whitespace, and filter out empty lines
                file_dois = [line.strip() for line in file if line.strip()]
                all_dois.extend(file_dois)  # Add DOIs from this file to main list
                logging.info(f"Read {len(file_dois)} DOIs from {doi_file.name}")
        except Exception as e:
            logging.error(f"Error reading {doi_file.name}: {str(e)}")
            continue
    return all_dois

async def process_dois(dois: List[str],keywords: List[str] ) -> None:
    searcher = DOIAbstractSearcher()

    # Use TCPConnector with limit to prevent too many connections
    connector = TCPConnector(limit=9)
    async with ClientSession(connector=connector) as session:
        with open('Filtered_Abstract_DOI.txt', 'w') as output_file:
            for i in range(0, len(dois), 9):  # Process in batches of 9
                batch = dois[i:i + 9]
                try:
                    # Process batch of DOIs
                    abstracts = await asyncio.gather(
                        *[searcher.get_abstract(session, doi) for doi in batch]
                    )

                    # Process results
                    for doi, abstract in zip(batch, abstracts):
                        if searcher.fuzzy_search_keywords(abstract, keywords):
                            output_file.write(f"{doi}\n")
                        logging.info(f"Processed: {doi}")

                    # Small delay between batches to ensure rate limit
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    continue


def main():
    try:
        # Put the keywords, you would like to search for here.
        keywords = ["Intermediate Representation", "Code Generation", "Optimization"]
        all_dois = read_all_doi_files('Unfiltered_DOI_Files') # Read all DOI files
        total_dois = len(all_dois)
        logging.info(f"Found total of {total_dois} DOIs to process")

        # Process DOIs
        asyncio.run(process_dois(all_dois, keywords))

        try:
            # Count results
            with open('Filtered_Abstract_DOI.txt', 'r') as results_file:
                filtered_dois = len(results_file.readlines())

            print(f"\nProcessing Summary:")
            print(f"Total DOIs processed: {total_dois}")
            print(f"DOIs matching criteria: {filtered_dois}")
            print(f"Match rate: {(filtered_dois/total_dois*100):.2f}%")

        except FileNotFoundError:
            logging.error("Results file not found. No DOIs may have matched the criteria.")
            print("\nProcessing Summary:")
            print(f"Total DOIs processed: {total_dois}")
            print("DOIs matching criteria: 0")
            print("Match rate: 0.00%")

    except Exception as e:
        logging.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()