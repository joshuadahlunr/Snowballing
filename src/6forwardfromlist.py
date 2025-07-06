import snowball_citations

with open("dois.txt", "r") as f:
    dois = f.read().splitlines()

for i, doi in enumerate(dois):
    try:
        found = snowball_citations.recursive_search(doi, 6)
    except snowball_citations.ThrowSeen as s:
        print("\nKeyboardInterrupt received. Stopping early...")
        found = s.seen
        break
    finally:
        with open(f"dois{i}.txt", "w") as f:
            for doi in found:
                f.write(f"{doi}\n")