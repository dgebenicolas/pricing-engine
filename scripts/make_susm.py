"""One-off: write the Standard Ultimate Survival Model (SUSM) to CSV.

DHW Appendix D: Makeham's law, mu_x = A + B c^x, so
    q_x = 1 - exp(-A - B/ln(c) * c^x * (c - 1)).
Closed at omega = 130 with q_130 = 1 (closing at 100 misses Table D.4).
The engine never runs this; it only reads the CSV it produces.
"""

import csv
import math
from pathlib import Path

A, B, C = 0.00022, 2.7e-6, 1.124
MIN_AGE, OMEGA = 20, 130
OUT = Path(__file__).resolve().parents[1] / "pricing_engine" / "config" / "mortality" / "susm.csv"


def makeham_q(x: int) -> float:
    return 1 - math.exp(-A - B / math.log(C) * C**x * (C - 1))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["age", "q"])
        for x in range(MIN_AGE, OMEGA):
            writer.writerow([x, repr(makeham_q(x))])
        writer.writerow([OMEGA, 1.0])


if __name__ == "__main__":
    main()
