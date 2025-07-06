# === Standard Library Imports ===
import asyncio
import functools
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

# === Third-party Imports ===
import aiohttp
import PyPDF2
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

# === Custom Import ===
from generate_manual_review_files import generate_manual_review_files

# === Base Paths ===
BASE_DIR = Path("../filtering").resolve()
RESULTS_DIR = BASE_DIR / "doi_results"
PROCESSING_DIR = BASE_DIR / "processing"
PDF_DIR = PROCESSING_DIR / "pdfs"
SNIPPET_DIR = PROCESSING_DIR / "snippets"
MANUAL_REVIEW_DIR = BASE_DIR / "doi_manual_review"

# === Directory Creation ===
def ensure_directories():
    for path in [RESULTS_DIR, PROCESSING_DIR, PDF_DIR, SNIPPET_DIR, MANUAL_REVIEW_DIR]:
        path.mkdir(parents=True, exist_ok=True)

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROCESSING_DIR / "body_fetch.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class APIConfig:
    url: str
    rate_limit: float
    email: str = ''
    last_request: float = field(default_factory=time.time)

@dataclass
class ContentChecker:
    thematic_groups: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "core_ir": {
            "intermediate representation": 1.0, "ir": 1.0, "ir form": 0.9,
            "ssa": 1.0, "ssa form": 1.0, "single static assignment": 1.0,
            "three address code": 0.9, "3ac": 0.9, "quadruples": 0.8,
            "cfg": 0.9, "control flow graph": 1.0,
            "ast": 0.9, "abstract syntax tree": 1.0, "parse tree": 0.8, "syntax tree": 0.8,
            "dag": 0.8, "directed acyclic graph": 1.0,
            "pdg": 0.9, "program dependence graph": 1.0, "dataflow graph": 0.9, "dfg": 0.9,
            "vsdg": 0.9, "value state dependence graph": 1.0,
            "vdg": 0.8, "dependence web": 0.8, "dependence graph": 0.9,
            "gsa": 0.9, "gamma function": 0.9,
            "thinned gated single assignment": 1.0,
            "control-dataflow hybrid": 0.8
        },
        "named_irs": {
            "suif": 1.0, "llhd": 1.0, "tapir": 1.0, "pegasus": 0.9, "uncol": 0.9,
            "spir": 0.9, "spirv": 1.0, "spire": 0.9,
            "gral": 0.8, "graal": 0.9, "htg": 0.9, "kimble": 0.8, "xark": 0.8,
            "theta graph": 0.8, "click ir": 0.8,
            "mlir": 1.0, "mlir dialect": 1.0, "llvm dialect": 0.9, "mlir pass": 0.9,
            "codon": 0.9, "solang": 0.9, "swift ir": 0.8, "ghc core": 0.8,
            "logical ir": 0.8, "wam": 0.8
        },
        "optimizations": {
            "optimization": 0.8,
            "global value numbering": 1.0, "gvn": 0.9,
            "automatic parallelization": 1.0, "parallelism": 0.9, "thread extraction": 0.8,
            "automatic vectorization": 1.0, "vectorization": 1.0, "vector ir": 0.9,
            "loop unrolling": 1.0, "loop transformation": 1.0,
            "constant folding": 1.0, "dead code elimination": 1.0, "dce": 0.9,
            "strength reduction": 0.9, "instruction scheduling": 1.0,
            "common subexpression elimination": 0.9, "cse": 0.9,
            "code motion": 0.9, "loop fusion": 0.8, "inlining": 0.8,
            "peephole optimization": 0.9, "range analysis": 0.8,
            "reaching definition": 0.8, "liveness analysis": 0.8,
            "nullness analysis": 0.7, "alias analysis": 0.8,
            "dataflow analysis": 0.9, "shape analysis": 0.7,
            "escape analysis": 0.8, "control dependence": 0.8, "data dependence": 0.9
        },
        "code_generation": {
            "code generation": 1.0, "codegen": 1.0,
            "target code generation": 0.9, "ir lowering": 0.9,
            "machine code": 1.0, "assembly code": 0.9,
            "code emitter": 0.9, "backend": 1.0, "llvm backend": 1.0,
            "interpreter": 0.8, "translator": 0.8,
            "bytecode": 0.9, "bytecode generation": 1.0,
            "virtual machine": 0.7, "vm code": 0.8,
            "llvm codegen": 1.0, "gimple": 0.9, "gimplifier": 0.8
        },
        "program_analysis": {
            "static analysis": 1.0, "abstract interpretation": 1.0,
            "data flow": 0.9, "control flow": 0.9,
            "symbolic execution": 0.8, "taint analysis": 0.8,
            "program slicing": 0.8, "semantic analysis": 0.8,
            "value tracking": 0.7, "dependence analysis": 0.9
        },
        "representations": {
            "ssa graph": 1.0, "phi node": 1.0, "phi function": 1.0,
            "use-def chain": 0.9, "def-use chain": 0.9,
            "three address": 1.0, "tac": 0.9, "temp variable": 0.9,
            "quad form": 0.8, "symbolic temporaries": 0.8,
            "dag representation": 0.9, "expression tree": 0.9,
            "operator dag": 0.8, "value numbering": 0.9,
            "expression dag": 0.8, "subexpression elimination": 0.8
        },
        "notations_and_forms": {
            "prefix notation": 0.8, "postfix notation": 0.8,
            "polish notation": 0.9, "reverse polish notation": 0.9,
            "rpn": 0.8, "linear ir": 0.8, "mathematical ir": 0.8,
            "functional ir": 0.8, "lambda calculus": 0.9, "logical ir": 0.9
        },
        "fuzzing_and_testing": {
            "ir fuzzing": 1.0, "bytecode fuzzing": 0.9,
            "spirv fuzzing": 1.0, "fuzz testing": 0.9,
            "ir verification": 0.9, "bytecode verification": 0.9
        },
        "toolchains": {
            "llvm": 1.0, "clang": 1.0, "gcc": 1.0, "mlir": 1.0,
            "ghc": 0.9, "codon": 0.9, "solang": 0.9,
            "swift": 0.9, "jvm": 0.8, "vm": 0.8,
            "intermediate compiler": 0.9, "compiler framework": 1.0
        },
        "exotic_and_experimental": {
            "interaction nets": 1.0, "interaction combinators": 1.0,
            "substitution rules": 0.9, "five of the six interaction net substitution rules": 1.0,
            "theta graph": 1.0, "well-formedness constraint": 1.0
        }
    })

    min_group_matches: int = 2  # Minimum number of groups that must match
    min_group_score: float = 0.2  # Minimum score within each group
    min_total_score: float = 0.15  # Minimum overall relevance score
    pdf_cache: Dict[str, str] = field(default_factory=dict)
    text_cache: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.apis = {
            'unpaywall': APIConfig("https://api.unpaywall.org/v2/", 1.0, "your@email.com"),
            'semantic_scholar': APIConfig("https://api.semanticscholar.org/graph/v1/paper/", 1.0),
            'europepmc': APIConfig("https://www.ebi.ac.uk/europepmc/webservices/rest/article", 0.5),
            'doaj': APIConfig("https://doaj.org/api/v2/search/articles/doi:", 1.0)
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def get_file_path(self, doi: str, filetype: str) -> Path:
        safe_name = doi.replace("/", "_").replace(".", "_")
        filetype = filetype.lower()
        if filetype == 'pdf':
            return PDF_DIR / f"{safe_name}.pdf"
        elif filetype == 'snippet':
            return SNIPPET_DIR / f"{safe_name}.txt"
        return PROCESSING_DIR / f"{safe_name}.{filetype}"

    def check_relevance(self, text: str) -> Tuple[bool, float, Dict[str, List[str]]]:
        """Check if a document's text is relevant based on thematic group matches."""

        if not text:
            return False, 0.0, {}

        text_lower = text.lower()
        matched_keywords_by_group: Dict[str, List[str]] = {}
        total_score = 0.0
        groups_above_threshold = 0

        for group_name, keywords in self.thematic_groups.items():
            matched_keywords: List[str] = []
            group_score = 0.0

            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    matched_keywords.append(keyword)
                    group_score += weight

            if matched_keywords:
                normalized_group_score = group_score / len(keywords)
                if normalized_group_score >= self.min_group_score:
                    matched_keywords_by_group[group_name] = matched_keywords
                    total_score += normalized_group_score
                    groups_above_threshold += 1

        if not matched_keywords_by_group:
            return False, 0.0, {}

        # Normalize total score by number of thematic groups
        total_score /= len(self.thematic_groups)

        is_relevant = (
                groups_above_threshold >= self.min_group_matches and
                total_score >= self.min_total_score
        )

        return is_relevant, total_score, matched_keywords_by_group

    async def check_rate_limit(self, api_name: str):
        api = self.apis[api_name]
        elapsed = time.time() - api.last_request
        if elapsed < api.rate_limit:
            await asyncio.sleep(api.rate_limit - elapsed)
        api.last_request = time.time()

    async def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        pdf_path_str = str(pdf_path)
        if pdf_path_str in self.text_cache:
            return self.text_cache[pdf_path_str]

        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                self.thread_pool,
                self._extract_text_sync,
                pdf_path_str
            )
            if text:
                self.text_cache[pdf_path_str] = text
            return text
        except Exception as e:
            logging.error(f"PDF extraction error: {str(e)}")
            return None

    def _extract_text_sync(self, pdf_path: str) -> Optional[str]:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            logging.error(f"PDF sync extraction error: {str(e)}")
            # Delete corrupt PDF file to avoid retrying a bad file
            try:
                Path(pdf_path).unlink()
                logging.info(f"Deleted corrupt PDF file {pdf_path}")
            except Exception as del_e:
                logging.warning(f"Failed to delete corrupt PDF file {pdf_path}: {del_e}")
            return None

    async def download_pdf(self, session: aiohttp.ClientSession, url: str, doi: str) -> Optional[str]:
        if doi in self.pdf_cache:
            return self.pdf_cache[doi]

        pdf_path = self.get_file_path(doi, "pdf")
        if pdf_path.exists():
            self.pdf_cache[doi] = str(pdf_path)
            return str(pdf_path)

        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                content = await response.read()
                pdf_path.write_bytes(content)
                self.pdf_cache[doi] = str(pdf_path)
                return str(pdf_path)
        except Exception as e:
            logging.error(f"Download error for {doi}: {str(e)}")
            return None

    async def process_paper(self, session: aiohttp.ClientSession, doi: str, metadata: dict) -> Tuple[
        bool, Optional[dict], str]:
        """Process single paper with optimized flow"""
        if not metadata.get('is_oa'):
            return False, None, "no_pdf"

        pdf_url = metadata.get('best_oa_location', {}).get('url_for_pdf')
        if not pdf_url:
            return False, None, "no_pdf"

        pdf_path = await self.download_pdf(session, pdf_url, doi)
        if not pdf_path:
            return False, None, "no_pdf"

        text = await self.extract_text_from_pdf(pdf_path)
        if not text:
            return False, None, "no_pdf"

        is_relevant, score, matches = self.check_relevance(text)
        if not is_relevant:
            return False, None, "irrelevant"

        return True, {
            'doi': doi,
            'relevance_score': score,
            'text_length': len(text),
            'pdf_url': pdf_url,
            'matching_keywords': matches
        }, "relevant"

    async def try_unpaywall(self, session: aiohttp.ClientSession, doi: str) -> Tuple[bool, Optional[dict], str]:
        await self.check_rate_limit('unpaywall')
        api = self.apis['unpaywall']

        try:
            url = f"{api.url}{doi}?email={api.email}"
            async with session.get(url) as response:
                if response.status != 200:
                    return False, None, "error"

                data = await response.json()
                return await self.process_paper(session, doi, data)

        except Exception as e:
            logging.error(f"Unpaywall error for {doi}: {str(e)}")
            return False, None, "error"

    async def try_semantic_scholar(self, session, doi):
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,openAccessPdf"
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("openAccessPdf") and data["openAccessPdf"].get("url"):
                        return True, {"doi": doi, "pdf_url": data["openAccessPdf"]["url"]}, "success"
                    return False, None, "no_pdf"
                return False, None, "error"
        except Exception as e:
            logging.error(f"Semantic Scholar error for {doi}: {e}")
            return False, None, "error"


    async def try_europepmc(self, session: aiohttp.ClientSession, doi: str) -> Tuple[bool, Optional[dict], str]:
        await self.check_rate_limit('europepmc')

        try:
            params = {'query': f"DOI:{doi}", 'format': 'json', 'resultType': 'core'}
            async with session.get(self.apis['europepmc'].url, params=params) as response:
                if response.status != 200:
                    return False, None, "error"

                data = await response.json()
                results = data.get('resultList', {}).get('result', [])

                for result in results:
                    if result.get('inEPMC') == 'Y' and result.get('hasPDF') == 'Y':
                        metadata = {
                            'is_oa': True,
                            'best_oa_location': {
                                'url_for_pdf': f"https://europepmc.org/article/PMC/{result['pmcid']}/pdf"
                            }
                        }
                        return await self.process_paper(session, doi, metadata)

                return False, None, "no_pdf"

        except Exception as e:
            logging.error(f"Europe PMC error for {doi}: {str(e)}")
            return False, None, "error"


    async def try_doaj(self, session: aiohttp.ClientSession, doi: str) -> Tuple[bool, Optional[dict], str]:
        await self.check_rate_limit('doaj')
        api = self.apis['doaj']
        url = f"{api.url}{doi}"

        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return False, None, "error"

                data = await response.json()
                if 'bibjson' in data and 'link' in data['bibjson']:
                    for link in data['bibjson']['link']:
                        if link.get('type') == 'fulltext' and link.get('url', '').endswith('.pdf'):
                            pdf_url = link['url']
                            return await self.process_paper(session, doi, {
                                'is_oa': True,
                                'best_oa_location': {'url_for_pdf': pdf_url}
                            })

                return False, None, "no_pdf"
        except Exception as e:
            logging.error(f"DOAJ error for {doi}: {e}")
            return False, None, "error"

async def process_dois_parallel(dois: List[str], batch_size: int = 5):
    checker = ContentChecker()
    results = []
    no_pdfs = set()

    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)

    async with aiohttp.ClientSession(connector=connector) as session:
        progress = tqdm(total=len(dois), desc="Processing DOIs")

        for i in range(0, len(dois), batch_size):
            batch = dois[i:i + batch_size]
            tasks = []

            async def process_with_fallback(doi):
                for func in [
                    functools.partial(checker.try_unpaywall, session),
                    functools.partial(checker.try_semantic_scholar, session),
                    functools.partial(checker.try_europepmc, session),
                    functools.partial(checker.try_doaj, session),
                ]:
                    success, data, status = await func(doi)
                    if success or status == "no_pdf":
                        return success, data, status
                return False, None, "error"

            # Inside process_dois_parallel, replace the loop with:
            tasks = [process_with_fallback(doi) for doi in batch]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(batch_results):
                doi = batch[idx]
                if isinstance(result, tuple):
                    success, data, status = result
                    if success and data:
                        results.append(data)
                    elif status == "no_pdf":
                        no_pdfs.add(doi)

            progress.update(len(batch))
            await asyncio.sleep(0.5)

    # === Save Results ===
    with open(RESULTS_DIR / "filtered_by_body.txt", 'w') as f:
        for result in results:
            f.write(f"{result['doi']}\n")

    with open(PROCESSING_DIR / "content_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_DIR / "dois_no_pdfs.txt", 'w') as f:
        for doi in sorted(no_pdfs):
            f.write(f"{doi}\n")

    print(f"\nProcessing Summary:")
    print(f"Total DOIs processed: {len(dois)}")
    print(f"Relevant papers found: {len(results)}")
    print(f"Papers with no accessible PDFs: {len(no_pdfs)}")
    print(f"Match rate: {(len(results) / len(dois) * 100):.2f}%")

    return results

async def main_async():
    try:
        with open(RESULTS_DIR / "filtered_by_abstract_dois.txt", "r") as f:
            dois = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error("Input file not found!")
        return

    logging.info(f"Processing {len(dois)} DOIs...")
    await process_dois_parallel(dois)

    # Final manual review file generation
    generate_manual_review_files(
        json_path=PROCESSING_DIR / "content_analysis.json",
        output_path=MANUAL_REVIEW_DIR / "review_doi.txt"
    )


def main():
    ensure_directories()
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
