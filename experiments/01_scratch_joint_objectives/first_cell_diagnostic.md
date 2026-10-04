# First-cell math diagnostic

Recorded 10-04-2026 after the frozen sequential seed 0 terminal evaluation.

**The first scratch cell's nonzero math score comes from answer collapse.**
All 1,319 generated responses yield the extracted numeric answer `8`; none
contains the `####` answer delimiter, and all reach the fixed 128-token generation
limit. Exactly 30 gold answers match that number. The frozen last-number extraction
rule therefore reports 30/1,319 exact matches: 2.27% [1.60%, 3.23%], a 95% Wilson
item interval conditional on this model; p-val=n/a (no hypothesis test).

This is a descriptive audit of already generated terminal outputs. It does not
establish reasoning ability, choose any setting, authorize another run, or alter
the frozen matrix. The experiment continues with unchanged models, schedules,
budgets and decoding. WikiText negative log likelihood remains the scratch primary;
math is the predeclared sparse secondary diagnostic. Other methods and seeds remain
pending, so this observation does not support a comparative verdict.

The cell has all 564 text blocks, 256 preference pairs, 2,376 multiple-choice items
and 1,319 math items, with unique identities checked against the frozen splits.
Its terminal summary SHA-256 (Secure Hash Algorithm, 256-bit) is
`5d1a9744d3b907d7a044626b106203c14461b6d867c5953a646c25814ec2f939`.
Full item-level responses remain in ignored runtime storage. Aggregate results
first landed in [commit 3b84582](https://github.com/brando90/unified-training/commit/3b845821f26baafd27ec45c6599b8fd5d68db522).
