from dataclasses import dataclass
from fuzzywuzzy import fuzz
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from aiohttp import ClientSession, TCPConnector
from pathlib import Path
import logging
import asyncio
import aiohttp
import time
import re

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class ThematicGroups:
    # Stage 1: Title keywords (early rejection)
    TITLE_KEYWORDS = {
        # Core IR terms
        "intermediate representation", "ir", "ssa", "static single assignment", "tac",
        "three address code", "three address form", "bytecode", "intermediate code",
        "intermediate form", "intermediate language", "internal representation",

        # Optimization and analysis
        "dead code elimination", "constant folding", "loop optimization", "strength reduction",
        "code motion", "peephole optimization", "compiler optimization", "code generation",

        # Control/data flow & structure
        "control_flow", "cfg", "dag", "directed acyclic graph", "program dependence graph", "pdg",

        # Tools & platforms
        "llvm", "mlir", "cranelift", "graalvm", "compcert", "clang frontend", "rustc",
        "gcc gimple", "wat", "wasm", "webassembly",

        # IR Levels & forms
        "high-level ir", "mid-level ir", "low-level ir", "hir", "mir", "lir", "sea of nodes",

        # Parsing-related
        "abstract syntax tree", "syntax tree", "parse tree",

        # Prefix/Postfix
        "prefix notation", "polish notation", "postfix notation", "reverse polish notation", "rpn"
    }

    IR_GROUPS = {
        "ssa": {
            "phi node", "phi functions", "single assignment", "dominator tree", "ssa form",
            "static single assignment", "def-use chain", "use-def chain", "variable renaming",
            "ssa renaming", "ssa construction"
        },
        "tac": {
            "three address code", "quadruples", "intermediate variable", "temporary variable",
            "temp variables", "symbolic temporaries", "assignment form", "three address form"
        },
        "bytecode": {
            "jvm bytecode", "ir bytecode", "stack-based ir", "virtual machine code",
            "bytecode verification", "typed bytecode", "webassembly", "wasm",
            "bytecode interpreter", "stack machine", "virtual instruction set"
        },
        "control_flow": {
            "cfg", "basic blocks", "data flow", "control flow graph", "dominance frontier",
            "dominance analysis", "loop analysis", "region analysis", "loop nesting tree",
            "loop unrolling", "dominator frontier"
        },
        "dag": {
            "directed acyclic graph", "dag representation", "expression tree", "common subexpression",
            "operator dag", "value numbering", "expression dag", "subexpression elimination",
            "common subexpression elimination", "ssa dag"
        },
        "dependency": {
            "program dependence graph", "pdg", "data dependence", "control dependence",
            "dependency analysis", "dependence graph"
        },
        "memory": {
            "memory ssa", "alias analysis", "points-to analysis", "memory dependence",
            "heap analysis", "pointer analysis", "alias set", "escape analysis",
            "memory graph", "load-store analysis"
        },
        "llvm": {
            "llvm ir", "llvm pass", "llvm optimization", "llvm bitcode", "llvm analysis",
            "llvm transformation", "llvm dialect", "mlir dialect", "llvm ir semantics"
        },
        "analysis": {
            "static analysis", "dataflow analysis", "abstract interpretation",
            "reaching definition", "liveness analysis", "constant propagation",
            "range analysis", "nullness analysis", "interval analysis", "shape analysis"
        },
        "optimization": {
            "dead code elimination", "constant folding", "loop optimization", "strength reduction",
            "code motion", "peephole optimization", "inlining", "branch prediction",
            "loop fusion", "code specialization", "loop invariant code motion"
        },
        "ast": {
            "abstract syntax tree", "syntax tree", "parse tree"
        },
        "prefix_postfix": {
            "prefix notation", "polish notation", "postfix notation",
            "reverse polish notation", "rpn"
        }
    }


class MetadataValidator:
    """Stage 2: Metadata validation"""

    def __init__(self):
        self.min_year = 1910
        self.valid_types = {
            "journal-article",  # Research articles in academic journals
            "conference-paper",  # Papers presented at conferences
            "proceedings-article",  # Articles published in conference proceedings
            "book-chapter",  # Chapters in academic books about compilers/IR
            "dissertation",  # PhD theses often contain detailed IR research
            "review-article",  # Survey papers about IR implementations
            "report",  # Technical reports from universities/research labs
            "reference-entry",  # Encyclopedia/handbook entries about IR
            "monograph"  # Detailed books focused on IR topics
        }
        self.valid_languages = {None, "en"}
        self.blacklisted_publishers = set()  # Add predatory publishers here

    def extract_year(self, metadata: dict) -> int:
        """Extract year from various possible metadata fields"""
        for field in ('published-print', 'published-online', 'issued'):
            parts = metadata.get(field, {}).get('date-parts', [])
            if parts and parts[0]:
                try:
                    return int(parts[0][0])
                except (ValueError, TypeError):
                    continue
        return 0

    def is_valid(self, metadata: dict) -> bool:
        try:
            # Date validation with improved year extraction
            year = self.extract_year(metadata)
            if year < self.min_year:
                return False

            # Type validation
            if metadata.get('type') not in self.valid_types:
                return False

            # Language validation
            # lang = metadata.get('language')
            # if lang is not None and not any(lang.startswith(valid_lang) for valid_lang in self.valid_languages):
            # return False

            # Publisher validation
            if metadata.get('publisher', '').lower() in self.blacklisted_publishers:
                return False

            return True
        except Exception as e:
            logging.error(f"Metadata validation error: {str(e)}")
            return False


class DOIFilter:
    def __init__(self):
        self.base_url = "https://api.crossref.org/works/"
        self.headers = {'User-Agent': 'YourApp/1.0 (mailto:knightjosephsnow@gmail.com)'}
        self.semaphore = asyncio.Semaphore(9)
        self.last_request_time = time.time()

        # Initialize components
        self.thematic_groups = ThematicGroups()
        self.metadata_validator = MetadataValidator()

        # Thresholds
        self.title_threshold = 80
        self.abstract_group_threshold = 1
        self.abstract_match_threshold = 70

    async def check_title(self, title: str) -> bool:
        """Stage 1: Title-based early rejection"""
        if not title:
            return False

        title_lower = title.lower()
        return any(
            fuzz.token_set_ratio(keyword, title_lower) >= self.title_threshold
            for keyword in self.thematic_groups.TITLE_KEYWORDS
        )

    def count_matching_groups(self, abstract: str) -> int:
        """Stage 3: Count matching thematic groups in abstract.
        Returns early if enough matches are found for acceptance."""
        if not abstract:
            return 0

        abstract_lower = abstract.lower()
        matching_groups = 0

        for group_keywords in self.thematic_groups.IR_GROUPS.values():
            if any(
                    fuzz.partial_ratio(keyword, abstract_lower) >= self.abstract_match_threshold
                    for keyword in group_keywords
            ):
                matching_groups += 1
                if matching_groups >= self.abstract_group_threshold:
                    return matching_groups  # Early return once we have enough matches

        return matching_groups

    async def fetch_with_retry(self, session: ClientSession, url: str, headers: dict,
                               retries: int = 3, base_delay: float = 0.5) -> Optional[dict]:
        """Fetch data with exponential backoff retry logic"""
        for attempt in range(retries):
            try:
                current_time = time.time()
                elapsed = current_time - self.last_request_time
                if elapsed < 0.11:  # Rate limiting
                    await asyncio.sleep(0.11 - elapsed)

                async with session.get(url, headers=headers) as response:
                    self.last_request_time = time.time()

                    if response.status == 200:
                        return await response.json()

                    if response.status in (429, 500, 502, 503, 504):
                        delay = base_delay * (2 ** attempt)
                        logging.warning(f"Retryable error {response.status} for {url}, "
                                        f"waiting {delay:.2f}s before retry {attempt + 1}/{retries}")
                        await asyncio.sleep(delay)
                        continue

                    logging.error(f"Non-retryable HTTP {response.status} for {url}")
                    return None

            except aiohttp.ClientError as e:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Network error for {url}: {e}, "
                                f"waiting {delay:.2f}s before retry {attempt + 1}/{retries}")
                await asyncio.sleep(delay)
                continue
            except Exception as e:
                logging.error(f"Unexpected error for {url}: {e}")
                return None

        logging.error(f"Failed after {retries} retries for {url}")
        return None

    async def process_doi(self, session: ClientSession, doi: str) -> Optional[str]:
        """Process a single DOI through all filtering stages"""
        try:
            data = await self.fetch_with_retry(session, f"{self.base_url}{doi}", self.headers)
            if not data:
                return None

            message = data.get('message', {})

            # Stage 1: Title Check
            title = message.get('title', [''])[0]
            if not await self.check_title(title):
                logging.debug(f"Rejected (title) for DOI: {doi} - Title: {title}")
                return None

            # Stage 2: Metadata Validation
            if not self.metadata_validator.is_valid(message):
                year = self.metadata_validator.extract_year(message)
                doc_type = message.get('type')
                lang = message.get('language')
                logging.debug(f"Rejected (metadata) for DOI: {doi} - "
                              f"Year: {year}, Type: {doc_type}, Language: {lang}")
                return None

            # Stage 3: Abstract Analysis with cleaned text
            raw_abstract = message.get('abstract', '')  # Clean the abstract

            matching_groups = self.count_matching_groups(raw_abstract)
            if matching_groups >= self.abstract_group_threshold:
                logging.info(f"Accepted DOI: {doi} - Matched {matching_groups} groups")
                return doi

            logging.debug(f"Rejected (abstract) for DOI: {doi} - "
                          f"Matched only {matching_groups}/{self.abstract_group_threshold} required groups")
            return None

        except Exception as e:
            logging.error(f"Error processing DOI {doi}: {str(e)}")
            return None


async def process_dois(dois: List[str]) -> None:
    doi_filter = DOIFilter()
    connector = TCPConnector(limit=9)

    async with ClientSession(connector=connector) as session:
        with open('DOI_Results/Filtered_Abstract_DOI.txt', 'w') as output_file:  # Updated path
            for i in range(0, len(dois), 9):
                batch = dois[i:i + 9]
                try:
                    results = await asyncio.gather(
                        *[doi_filter.process_doi(session, doi) for doi in batch]
                    )

                    # Write accepted DOIs to file
                    for doi in filter(None, results):
                        output_file.write(f"{doi}\n")

                    await asyncio.sleep(0.1)

                except Exception as e:
                    logging.error(f"Batch processing error: {str(e)}")
                    continue


def read_all_doi_files(directory_path: str = 'Unfiltered_DOI_Files') -> List[str]:
    """Reads all DOI files from a directory and combines them into a single list."""
    all_dois = set()  # Using set to remove duplicates while reading
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    for doi_file in directory.glob('*.txt'):
        try:
            with doi_file.open('r') as file:
                file_dois = {line.strip() for line in file if line.strip()}
                all_dois.update(file_dois)
                logging.info(f"Read {len(file_dois)} DOIs from {doi_file.name}")
        except Exception as e:
            logging.error(f"Error reading {doi_file.name}: {str(e)}")

    return list(all_dois)


def main():
    try:
        # Use read_all_doi_files instead of reading a single file
        dois = read_all_doi_files('Unfiltered_DOI_Files')
        total_dois = len(dois)
        logging.info(f"Found {total_dois} DOIs to process from multiple files")

        # Process DOIs
        asyncio.run(process_dois(dois))

        # Count results
        try:
            with open('DOI_Results/Filtered_Abstract_DOI.txt', 'r') as results_file:
                filtered_dois = len(results_file.readlines())

            print(f"\nProcessing Summary:")
            print(f"Total DOIs processed: {total_dois}")
            print(f"DOIs matching criteria: {filtered_dois}")
            print(f"Match rate: {(filtered_dois / total_dois * 100):.2f}%")

        except FileNotFoundError:
            logging.error("No results file found")
            print("\nProcessing Summary:")
            print(f"Total DOIs processed: {total_dois}")
            print("DOIs matching criteria: 0")
            print("Match rate: 0.00%")

    except Exception as e:
        logging.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()