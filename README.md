# **C**ompiler and **A**cademic **T**ext **G**athering **I**ntelligent **R**esearch **L**ibrary
C.A.T.G.I.R.L is A Python-based tool for filtering Digital Object Identifiers (DOIs) of academic papers related to Intermediate Representation (IR) and compiler technology. The tool processes DOIs through multiple filtering stages to identify relevant academic content.
We may nyot be pawfessionals in hydwodynamics, but we suwe know how to surf a sea of citations~! 💖 Whether it’s SSA or TAC, we’ll nevah compiwomise on pwoper wesearch etqwette, nya~!”
## Features

- Multi-stage filtering process for academic paper DOIs
- Asynchronous processing with rate limiting
- Fuzzy text matching for content relevance
- Comprehensive thematic group categorization
- Metadata validation for academic content
- Batch processing with error handling
- Detailed logging system

## Requirements

- Python 3.12+
- Required packages:
  - `fuzzywuzzy`: For fuzzy text matching
  - `aiohttp`: For asynchronous HTTP requests
  - `requests`: For HTTP operations

## Installation

1. Clone this repository
2. Install required packages:

## Directory Structure
1. After running the snowballing.py files results should be stored in `Unfiltered_DOI_Files`
2. Place any additional DOI files (`.txt` format) in the `Unfiltered_DOI_Files` directory 
3. The resulting files from `Filter_Abstract.py` and `Filter_Body` will be placed in `DOI_Results`

## Filtering Process

The tool employs a three-stage filtering process:

1. **Title Keywords Check**: Early rejection based on paper titles
2. **Metadata Validation**: Checks publication type, year, and language
3. **Abstract Analysis**: Content relevance checking using thematic groups

### Thematic Groups

The system includes comprehensive keyword sets for various IR-related topics:
- Core IR terms
- Optimization and analysis
- Control/data flow
- Tools and platforms
- IR Levels and forms
- Parsing-related concepts
- And more...

### Configuration

- Default minimum year: 1910
- Title match threshold: 80%
- Abstract match threshold: 70%
- Rate limiting: 9 concurrent requests
- Request delay: 0.11 seconds between requests

### Output

The script generates:
- Filtered DOIs in `DOI_Results/Filtered_Abstract_DOI.txt`
- Processing summary with:
  - Total DOIs processed
  - Number of matching DOIs
  - Match rate percentage
- Detailed logs with rejection reasons

### Error Handling

- Automatic retry mechanism for failed requests
- Exponential backoff for rate limiting
- Comprehensive error logging
- Exception handling for file operations
## Stage 2: Full-Text (Body) Retrieval and Filtering

A comprehensive second-stage analysis that goes beyond abstracts to improve filtering accuracy and precision in academic paper selection.

### Purpose

This phase enhances paper filtering by:
- Analyzing complete paper content for higher precision
- Capturing relevant discussions not mentioned in abstracts
- Providing comprehensive metadata for survey inclusion

### 🛠Implementation Details

1. **Open Access Detection**
   - Utilizes Unpaywall & OpenAlex APIs
   - Checks publisher metadata
   - Explores preprint repository availability

2. **Full-Text Download and Parsing**
   - Retrieves PDF/HTML/plain text versions
   - Processes PDFs using pdfminer.six
   - Parses HTML content with BeautifulSoup

3. **Thematic Content Filtering**
   - Applies multi-stage keyword analysis
   - Section-specific content evaluation
   - Minimum match score requirements

###  Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| MIN_BODY_MATCH_PERCENT | 65% | Minimum content match threshold |
| MAX_DOWNLOAD_RETRIES | 3 | Number of retry attempts |
| PDF_TEXT_EXTRACTOR | pdfminer | PDF text extraction library |
| USE_UNPAYWALL | True | Unpaywall API usage flag |
| REQUEST_DELAY | 0.11s | Inter-request delay |

### Output Structure

- `DOI_Results/Filtered_Body_DOI.txt`: Final filtered DOIs
- `Matched_Body_Snippets/`: Relevant text extracts
- `Body_Fetch_Log.txt`: Processing logs and statistics


### Troubleshooting Guide

Common issues and solutions:
- Inaccessible full text: Logged and skipped
- PDF parsing challenges: OCR integration planned
- Rate limiting: Automatic request throttling
- Need to be below 100,000 API request per day for full body read


## Stage 3: Experimental Filtering Methodologies

### DOI Battle Royale
Our machine learning model simulates academic "combat" between papers based on:
- Citation impact factors
- Methodology robustness
- Implementation complexity
- Real-world applications

### Quantum-Inspired Hypercube Filter
Leveraging principles from quantum computing and 4D visualization:
- Maps papers onto a 4D hypercube based on relevance metrics
- Performs quantum-inspired state superposition of selection criteria
- Collapses the probability wave function to identify optimal papers
- Achieves O($#%^) complexity because why not?

### Stochastic Selection Protocol
Implements a controlled chaos approach:
- Random selection with weighted probabilities
- Monte Carlo simulation of paper relevance
- Entropy-based filtering thresholds
- Statistical significance testing of randomness

### Biomimetic Selection Algorithm
Inspired by evolutionary computing processes:
- Deploys a flock of "chickens"
- Each chicken follows specialized selection heuristics
- Papers selected through emergent swarm behavior
- Self-organizing paper clusters form naturally

### Crowdsourced Quantum Number Theory
Combines public participation with advanced mathematics:
- Live-streamed paper selection events
- Audience input converted to quantum states
- Real-time coherence checking
- Collective intelligence optimization

## If Catgirls dont like water? How could they like building snowballs.
"Nyaa~! We’re nyot just your avewage papew-chasews, teehee~! (≧◡≦) ♡ We awe the MEOWWW-TASTIC catgiwls of cowd, cawcuwated data~! 
Even if we don't wike watew—ewww soggy paws!—we've got the purr-fect system fow handwing academic papews, nyan! (ฅ’ω’ฅ)♪
we wills sniffff out knowwedge~! We're nyot afwaid to get ouww pawwies a wittwe frosty to cowwect and owganize aww that bwoody wovewy weseawch data! ( •̀ ω •́ )✧
we wuv to carefuwwy wown up each impawtant papew into one neat and tidy cowwection, nyaaa~! It’s a totaw meow-vement of wuv and wuogic~!Let's go make academia nyaneriffic, nyaaaa~!!💖💖💖💻🎀✨

![Quinn](RandomFiles(Delete%20Later)/Quinn.png)