import snowball_citations
import snowball_references

# Example DOI (you can replace this with any DOI)
target_doi = "10.1145/2480741.2480743"

found = set()

try:
    back2 = snowball_references.recursive_search(target_doi, 2)
    found.update(back2)
    for doi in back2:
        found.update(snowball_citations.recursive_search(doi, 6))
except snowball_references.ThrowSeen as s:
    print("\nKeyboardInterrupt received. Stopping early...")
    found.update(s.seen)
except snowball_citations.ThrowSeen as s:
    print("\nKeyboardInterrupt received. Stopping early...")
    found.update(s.seen)

with open("dois.txt", "w") as f:
    for doi in found:
        f.write(f"{doi}\n")
