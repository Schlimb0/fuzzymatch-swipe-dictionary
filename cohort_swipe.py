"""Swipe cohort8 pairs (cohort8_count == 2): → dupe, ← not a dupe, q quit. No dependencies.
Usage: python cohort_swipe.py products.csv   (decisions saved to products_cohort_decisions.csv; re-run to resume)"""
import csv, os, sys
from collections import defaultdict

try:
    import msvcrt
    def key():
        k = msvcrt.getwch()
        return msvcrt.getwch() if k in '\x00\xe0' else k  # arrows: M right, K left
except ImportError:
    import termios, tty
    def key():
        fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
        try: tty.setraw(fd); return os.read(fd, 3).decode()[-1]  # arrows: C right, D left
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old)

src = sys.argv[1]
out = os.path.splitext(src)[0] + '_cohort_decisions.csv'
cohorts = defaultdict(list)
with open(src, newline='', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if float(r['cohort8_count'] or 0) == 2: cohorts[r['cohort8']].append(r)
done = set()
if os.path.exists(out):
    with open(out, newline='', encoding='utf-8') as f: done = {r['cohort8'] for r in csv.DictReader(f)}
todo = [(c, rs) for c, rs in cohorts.items() if len(rs) == 2 and c not in done]

with open(out, 'a', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    if not done: w.writerow(['cohort8', 'desc_a', 'desc_b', 'dupe'])
    for i, (c, (a, b)) in enumerate(todo, 1):
        print(f"\033[2J\033[H[{i}/{len(todo)}] cohort8 {c}\n\n  {a['MATL_SHRT_DESC']}\n  {b['MATL_SHRT_DESC']}\n\n→ dupe   ← not dupe   q quit")
        while (k := key()) not in 'CMldDKhaq': pass
        if k == 'q': break
        w.writerow([c, a['MATL_SHRT_DESC'], b['MATL_SHRT_DESC'], k in 'CMld']); f.flush()
print(f"Saved to {out}")
