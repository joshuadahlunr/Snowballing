# file: generate_manual_review_files.py

import json
from pathlib import Path
from typing import Union

def generate_manual_review_files(
    json_path: Union[str, Path],
    output_path: Union[str, Path] = "../filtering/doi_manual_review/review_doi.txt"
):
    """
    Reads a content_analysis.json file and writes a simplified DOI review list.

    Args:
        json_path (Union[str, Path]): Path to the JSON file containing analysis results.
        output_path (Union[str, Path], optional): Where to write the review summary.
            Defaults to "../filtering/doi_manual_review/review_doi.txt".
    """
    # Convert inputs to Path objects
    json_path = Path(json_path)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load content_analysis.json
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ JSON file not found: {json_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format in file: {json_path}")
        return

    # Sort papers by relevance score
    sorted_data = sorted(data, key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Write simplified DOI review output
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sorted_data:
            doi = item.get("doi", "N/A")
            score = item.get("relevance_score", 0.0)
            groups = ', '.join(item.get("matching_keywords", {}).keys())
            f.write(f"{doi} | score: {score:.3f} | groups: {groups}\n")

    print(f"✅ Review file saved to: {output_path.resolve()}")
