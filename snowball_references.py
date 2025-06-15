import requests
from ratelimit import limits, sleep_and_retry

class ThrowSeen(Exception):
    seen = set()

ONE_SECOND = 1

@sleep_and_retry
@limits(calls=9, period=ONE_SECOND)
def get_referenced_dois(doi):
    # Construct the API URL
    base_url = "https://opencitations.net/index/api/v1/references/"
    url = f"{base_url}{doi}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    
    try:
        data = response.json()
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
        return []

    referenced_dois = [entry['cited'] for entry in data if 'cited' in entry]
    return referenced_dois

def recursive_search(start_doi, max_depth=4, seen=None, found=None, current_depth=0):
    try:
        if seen is None:
            seen = set()

        if found is None:
            found = set()

        if current_depth > max_depth:
            return seen

        if start_doi in seen:
            return seen

        print(f"Depth {current_depth}: Looking up references for {start_doi}")
        seen.add(start_doi)

        referenced_dois = get_referenced_dois(start_doi)
        found.update(referenced_dois)
        print(len(found))

        for citing_doi in referenced_dois:
            if citing_doi not in seen:
                recursive_search(citing_doi, max_depth, seen, found, current_depth + 1)
    except KeyboardInterrupt:
        s = ThrowSeen()
        s.seen = seen.union(found)
        raise s

    return seen.union(found)

if __name__ == "__main__":

    # Example DOI (you can replace this with any DOI)
    target_doi = "10.1145/2480741.2480743"

    try:
        referenced_dois = recursive_search(target_doi)
    except ThrowSeen as s:
        print("\nKeyboardInterrupt received. Stopping early...")
        referenced_dois = s.seen


    print(f"\n{len(referenced_dois)} DOIs that cite {target_doi}:\n")
    with open("dois.txt", "w") as f:
        for doi in referenced_dois:
            f.write(f"{doi}\n")
