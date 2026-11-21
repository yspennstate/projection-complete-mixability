# Line projections do not determine complete mixability

Yitzchak Shmalo

A distribution F on R^d is n-completely mixable if there are n random vectors with law F whose sum is almost surely
constant. Linear images of n-completely mixable distributions are n-completely mixable, and Wang asked when the
converse holds. We show that for every n >= 3 there is a distribution on the plane, with a smooth density positive on
a disk or with an everywhere positive real-analytic density, whose projections onto all lines are n-completely
mixable while the distribution itself is not. For n = 2 the converse holds when the mean is finite and fails without
a mean. In the other direction, one generic projection decides complete mixability for distributions carried by
compact sets of small box dimension.

## Contents

- `main.tex`, `main.pdf`: the paper.
- `code/check_constants.py`: recomputes the constants of Propositions 2.2 and 2.3 in exact rational arithmetic and
  checks Lemma 3.2 and the inequality nu <= p of Section 3 numerically on a grid. Needs Python 3 with numpy and scipy.
  Run `python code/check_constants.py`; its output is in `code/check_constants.out`.
