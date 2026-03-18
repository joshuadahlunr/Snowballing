import snowball_citations
import glob, copy, os, re, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=None, help="Override start index")
parser.add_argument("--end", type=int, default=None, help="Override end index")
args = parser.parse_args()

with open("dois.list", "r") as f:
    dois = f.read().splitlines()

start_index = 0
already_explored = []
for file in glob.glob("doi*.txt"):
    with open(file, "r") as f:
        already_explored.extend(f.read().splitlines())
    match = re.fullmatch(r"dois(\d+)\.txt", os.path.basename(file))
    if match: start_index = max(start_index, int(match.group(1)))
already_explored = set(already_explored).difference(set(dois))
print(len(already_explored))

if args.start is not None:
    start_index = args.start
elif start_index > 0:
    start_index += 1
print(f"Resuming from index {start_index}")

last_found = []
found = []
seen = copy.deepcopy(already_explored)
for i, doi in enumerate(dois):
    if args.end is not None and i >= args.end: break
    if i < start_index: continue
    # if doi in already_explored: continue
    try:
        found = snowball_citations.recursive_search(doi, 2, seen=seen)
    except snowball_citations.ThrowSeen as s:
        print("\nKeyboardInterrupt received. Stopping early...")
        found = s.seen
        break
    finally:
        tmp = copy.deepcopy(found)
        found.difference_update(last_found)
        last_found = tmp

        with open(f"dois{i}.txt", "w") as f:
            for doi in found:
                if doi in already_explored: continue
                f.write(f"{doi}\n")