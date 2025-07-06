# Standard library imports
import asyncio
import logging
import re
import time
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Third-party imports
import aiohttp
from aiohttp import ClientSession, TCPConnector
from rapidfuzz import fuzz

# From other files
from combind_dois import collect_and_append_dois_multithreaded


# Setup logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class ThematicGroups:
    """
    Contains thematic groups of keywords related to intermediate representations (IR).
    Provides a flattened structure separating single-word and multi-word keywords
    for optimized matching against abstracts.
    """

    @dataclass
    class ThematicGroups:
        """
        Contains thematic groups of keywords related to intermediate representations (IR).
        Provides a flattened structure separating single-word and multi-word keywords.
        """
        IR_GROUPS = {
            "general_ir": {
                "intermediate representation", "intermediate code", "compiler ir",
                "ir", "intermediate form", "code representation", "program representation",
                "low-level ir", "high-level ir", "intermediate-level code",
                "middle-end", "ir format", "intermediate abstraction", "custom ir"
            },
            "ssa_forms": {
                "ssa", "single static assignment", "ssa form", "gsa", "gated single assignment",
                "in gated single-assignment form", "ssa graph", "thinned gated single assignment",
                "gamma function", "ssa construction", "phi node", "phi function", "def-use chain",
                "use-def chain", "ssa renaming"
            },
            "three_address_and_linear": {
                "three address code", "3ac", "quadruples", "linear ir", "tac", "temporary variable",
                "symbolic temporaries", "assignment form", "three address form"
            },
            "graph_based_ir": {
                "dag", "directed acyclic graph", "expression dag", "value numbering", "common subexpression",
                "graphical form", "theta graph", "dataflow graph", "dfg", "vdg", "value state dependence graph",
                "vsdg", "control flow graph", "cfg", "program dependence graph", "pdg", "data dependence",
                "control dependence", "program dependence web", "dependence graph", "control/dataflow hybrids"
            },
            "parser_structures": {
                "ast", "abstract syntax tree", "parse tree", "syntax tree", "polish notation",
                "prefix notation", "reverse polish notation", "postfix notation", "rpn"
            },
            "ir_implementations_and_frameworks": {
                "llvm", "clang", "swift", "ghc", "solang", "mlir", "codon", "gimple", "gimplifier",
                "gcc gimple", "xark", "graal", "kimble", "htg", "suif", "llhd", "tapir",
                "pegasus", "uncol", "spir", "spir-v", "spirev", "wam", "logical ir", "lambda calculus",
                "mathematical ir", "functional ir", "parallelizing llvm"
            },
            "optimization_support": {
                "dead code elimination", "constant folding", "loop optimization", "loop unrolling",
                "strength reduction", "code motion", "peephole optimization", "inlining",
                "branch prediction", "loop fusion", "code specialization", "global value numbering",
                "automatic vectorization", "automatic parallelization", "range analysis"
            },
            "code_generation": {
                "code generation", "target code generation", "ir to machine code", "llvm codegen",
                "code emitter", "backend", "well-formedness constraint", "ir lowering",
                "intermediate code generation", "machine-independent code"
            },
            "interaction_nets": {
                "interaction nets", "interaction combinators", "substitution rules",
                "five of the six interaction net substitution rules"
            },
            "fuzzing_and_testing": {
                "ir fuzzing", "spir-v fuzzing", "bytecode fuzzing", "intermediate representation fuzzing"
            }
        }

    def __init__(self):
        # On initialization, flatten keywords into single- and multi-word sets
        self.groups = self._flatten_groups()

    def _flatten_groups(self):
        """
        Splits IR_GROUPS keywords into two dictionaries:
        - single_word: groups with single-word keywords
        - multi_word: groups with multi-word phrases
        This enables optimized matching strategies later.
        """
        single_word = {}
        multi_word = {}
        for group, keywords in self.IR_GROUPS.items():
            for kw in keywords:
                if len(kw.split()) == 1:
                    single_word.setdefault(group, set()).add(kw)
                else:
                    multi_word.setdefault(group, set()).add(kw)
        return {"single_word": single_word, "multi_word": multi_word}


class MetadataValidator:
    """
    Validates the metadata of a CrossRef DOI record based on:
    - Publication year (minimum threshold)
    - Document type (must be in allowed types)
    - Language (optional; currently disabled)
    - Publisher (blacklist checking)
    """

    def __init__(self):
        self.min_year = 1910
        self.valid_types = {
            "journal-article",
            "conference-paper",
            "proceedings-article",
            "book-chapter",
            "dissertation",
            "review-article",
            "report",
            "reference-entry",
            "monograph"
        }
        self.valid_languages = {None, "en"}
        self.blacklisted_publishers = set()  # Add predatory publishers if needed

    def extract_year(self, metadata: dict) -> int:
        """
        Attempts to extract the publication year from possible date fields:
        'published-print', 'published-online', or 'issued'.
        Returns 0 if no valid year is found.
        """
        for field in ('published-print', 'published-online', 'issued'):
            parts = metadata.get(field, {}).get('date-parts', [])
            if parts and parts[0]:
                try:
                    return int(parts[0][0])
                except (ValueError, TypeError):
                    continue
        return 0

    def is_valid(self, metadata: dict) -> bool:
        """
        Runs validation checks against the metadata:
        - Year is after min_year
        - Type is in valid types
        - Publisher is not blacklisted
        Returns True if all checks pass, else False.
        """
        try:
            year = self.extract_year(metadata)
            if year < self.min_year:
                return False

            if metadata.get('type') not in self.valid_types:
                return False

            # Language validation is currently commented out:
            # lang = metadata.get('language')
            # if lang is not None and not any(lang.startswith(valid_lang) for valid_lang in self.valid_languages):
            #     return False

            if metadata.get('publisher', '').lower() in self.blacklisted_publishers:
                return False

            return True

        except Exception as e:
            logging.error(f"Metadata validation error: {e}")
            return False


class DOIFilter:
    """
    Main class responsible for fetching DOI metadata from CrossRef,
    validating metadata, analyzing abstracts for thematic keyword matches,
    and returning DOIs that pass all filters.
    """

    def __init__(self):
        self.base_url = "https://api.crossref.org/works/"
        self.headers = {'User-Agent': 'YourApp/1.0 (mailto:knightjosephsnow@gmail.com)'}
        self.semaphore = asyncio.Semaphore(9)  # Limit concurrent requests
        self.last_request_time = time.time()

        self.thematic_groups = ThematicGroups()
        self.metadata_validator = MetadataValidator()

        # Thresholds controlling filtering behavior
        self.abstract_group_threshold = 1          # Minimum thematic groups matched to accept
        self.abstract_match_threshold = 80         # RapidFuzz similarity score threshold

    def count_matching_groups(self, abstract: str) -> int:
        """
        Counts how many thematic groups have keywords that appear in the abstract.
        Single-word keywords are matched by exact token presence.
        Multi-word keywords are matched using fuzzy partial matching.
        Returns the number of matched groups.
        """
        if not abstract:
            return 0

        abstract_lower = abstract.lower()
        tokens = set(re.findall(r'\b\w+\b', abstract_lower))

        match_count = 0

        # Check single-word keyword matches
        for group, keywords in self.thematic_groups.groups["single_word"].items():
            if any(kw in tokens for kw in keywords):
                match_count += 1
                if match_count >= self.abstract_group_threshold:
                    return match_count

        # Check multi-word keyword matches with fuzzy matching
        for group, keywords in self.thematic_groups.groups["multi_word"].items():
            if any(fuzz.partial_ratio(kw, abstract_lower) >= self.abstract_match_threshold for kw in keywords):
                match_count += 1
                if match_count >= self.abstract_group_threshold:
                    return match_count

        return match_count

    async def fetch_with_retry(
        self,
        session: ClientSession,
        url: str,
        headers: dict,
        retries: int = 3,
        base_delay: float = 0.5
    ) -> Optional[dict]:
        """
        Makes an HTTP GET request to the specified URL with headers, retrying on
        rate-limit or server errors with exponential backoff.
        Returns the JSON response dict if successful, otherwise None.
        """
        for attempt in range(retries):
            try:
                # Enforce minimum delay between requests to respect rate limits
                elapsed = time.time() - self.last_request_time
                if elapsed < 0.11:
                    await asyncio.sleep(0.11 - elapsed)

                async with session.get(url, headers=headers) as response:
                    self.last_request_time = time.time()

                    if response.status == 200:
                        return await response.json()

                    if response.status in (429, 500, 502, 503, 504):
                        delay = base_delay * (2 ** attempt)
                        logging.warning(f"Retryable HTTP {response.status} for {url} - retry {attempt + 1} in {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue

                    logging.error(f"Non-retryable HTTP {response.status} for {url}")
                    return None

            except aiohttp.ClientError as e:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Network error for {url}: {e} - retry {attempt + 1} in {delay:.2f}s")
                await asyncio.sleep(delay)
                continue
            except Exception as e:
                logging.error(f"Unexpected error for {url}: {e}")
                return None

        logging.error(f"Failed after {retries} retries for {url}")
        return None

    async def process_doi(self, session: ClientSession, doi: str) -> Optional[str]:
        """
        Fetches metadata for a single DOI, validates it, and checks the abstract
        for thematic keyword matches.
        Returns the DOI if it passes all filters; otherwise, returns None.
        """
        try:
            data = await self.fetch_with_retry(session, f"{self.base_url}{doi}", self.headers)
            if not data:
                return None

            message = data.get('message', {})

            # Stage 1: Validate metadata
            if not self.metadata_validator.is_valid(message):
                year = self.metadata_validator.extract_year(message)
                doc_type = message.get('type')
                lang = message.get('language')
                logging.debug(f"Rejected (metadata) DOI {doi} - Year: {year}, Type: {doc_type}, Language: {lang}")
                return None

            # Stage 2: Analyze abstract for thematic matches
            raw_abstract = message.get('abstract', '')
            matched_groups = self.count_matching_groups(raw_abstract)

            if matched_groups >= self.abstract_group_threshold:
                logging.info(f"Accepted DOI {doi} - Matched {matched_groups} thematic groups")
                return doi

            logging.debug(f"Rejected (abstract) DOI {doi} - Matched {matched_groups} groups (threshold: {self.abstract_group_threshold})")
            return None

        except Exception as e:
            logging.error(f"Error processing DOI {doi}: {e}")
            return None


async def process_dois(dois: List[str]) -> None:
    """
    Processes a list of DOIs concurrently with a maximum concurrency limit.
    Writes the filtered DOIs that pass the checks to an output file.
    """
    doi_filter = DOIFilter()
    connector = TCPConnector(limit=9)  # Max concurrent HTTP connections

    output_file_path = '../filtering/doi_results/filtered_by_abstract_dois.txt'
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    async with ClientSession(connector=connector) as session:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            # Process DOIs in batches of 9 to respect concurrency limit
            for i in range(0, len(dois), 9):
                batch = dois[i:i + 9]
                try:
                    # Run async process_doi tasks concurrently
                    results = await asyncio.gather(*[doi_filter.process_doi(session, doi) for doi in batch])

                    # Write accepted DOIs to output file
                    for accepted_doi in filter(None, results):
                        output_file.write(f"{accepted_doi}\n")

                    # Small delay between batches to respect API rate limits
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logging.error(f"Batch processing error: {e}")


def prepare_and_read_combined_dois() -> List[str]:
    """
    Runs the DOI combining function on raw DOI files,
    then reads and returns a deduplicated list of DOIs from the combined file.
    """
    input_folder = "../filtering/unfiltered_doi_files"
    combined_file_path = "../filtering/unfiltered_doi_combind/combined_dois.txt"

    # Combine multiple DOI files into one deduplicated file
    collect_and_append_dois_multithreaded(input_folder, combined_file_path)

    combined_path = Path(combined_file_path)
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined DOI file not found: {combined_file_path}")

    # Read all DOIs from the combined file, stripping whitespace
    with combined_path.open('r', encoding='utf-8') as f:
        dois = [line.strip() for line in f if line.strip()]

    return dois


def main():
    """
    Main function to orchestrate the workflow:
    1. Combine raw DOI files and read the combined DOIs.
    2. Filter DOIs by fetching metadata and analyzing abstracts.
    3. Write filtered DOIs to output file.
    4. Print a summary of the results.
    """
    try:
        # Step 1: Prepare combined DOI list
        dois = prepare_and_read_combined_dois()
        total_dois = len(dois)
        logging.info(f"Found {total_dois} DOIs to process from combined file")

        # Step 2: Run the async processing pipeline
        asyncio.run(process_dois(dois))

        # Step 3: After processing, read and summarize results
        output_file_path = '../filtering/doi_results/filtered_by_abstract_dois.txt'
        try:
            with open(output_file_path, 'r', encoding='utf-8') as results_file:
                filtered_dois = len(results_file.readlines())

            print("\nProcessing Summary:")
            print(f"Total DOIs processed: {total_dois}")
            print(f"DOIs matching criteria: {filtered_dois}")
            print(f"Match rate: {(filtered_dois / total_dois * 100):.2f}%")

        except FileNotFoundError:
            logging.error("Results file not found after processing")
            print("\nProcessing Summary:")
            print(f"Total DOIs processed: {total_dois}")
            print("DOIs matching criteria: 0")
            print("Match rate: 0.00%")

    except Exception as e:
        logging.error(f"Application error: {e}")


if __name__ == "__main__":
    main()
