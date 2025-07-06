import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def read_dois_from_file(file_path):
    dois = set()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            doi = line.strip()
            if doi:
                dois.add(doi)
    return dois

def collect_and_append_dois_multithreaded(input_folder, output_file_path, max_threads=8):
    all_dois = set()

    # Load existing DOIs from output file (to avoid re-adding duplicates)
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as outfile:
            for line in outfile:
                doi = line.strip()
                if doi:
                    all_dois.add(doi)

    input_folder = Path(input_folder)
    files = [
        str(f) for f in input_folder.glob("*.txt")
        if os.path.abspath(f) != os.path.abspath(output_file_path)
    ]

    # Read files in parallel using threads
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for file_dois in executor.map(read_dois_from_file, files):
            all_dois.update(file_dois)

    # Write final deduplicated set to output
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for doi in sorted(all_dois):
            outfile.write(doi + "\n")

    print(f"[Multithreaded] Combined {len(all_dois)} unique DOIs into {output_file_path}")

# Only run the combine if called directly as a script
if __name__ == "__main__":
    collect_and_append_dois_multithreaded(
        input_folder="../filtering/unfiltered_doi_files",
        output_file_path="../filtering/unfiltered_doi_combind/combined_dois.txt",
        max_threads=8
    )