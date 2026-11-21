"""Checks for 'Line projections do not determine complete mixability'.

1. Four copies: the identity of Lemma 2.1 on random zero-sum quadruples, and the constants of
   Propositions 2.2 and 2.3 in exact rational arithmetic.
2. Three copies: Lemma 3.2 (the equation for f, the closed form, the bound |f| <= (25/3) y a) and the
   inequality nu <= p of the proof of Proposition 3.1, on a grid, for c = 1, -1 and 0.3.
Run: python check_constants.py   (needs numpy and scipy)
"""
import warnings
from fractions import Fraction as Fr
import numpy as np
from scipy.integrate import IntegrationWarning, quad

# The integrands in part 2 carry the factor 1/sqrt(1 - x^2), which is integrable at x = 1; quad still
# warns there. For Lemma 3.2 the integral is compared with its closed form (agreement about 1e-7).
warnings.simplefilter("ignore", IntegrationWarning)

rng = np.random.default_rng(1)

# ---- 1. four copies -------------------------------------------------------------------------
worst = 0.0
for _ in range(20000):
    w = rng.normal(size=3) + 1j * rng.normal(size=3)
    w = np.append(w, -w.sum())
    lhs = (w ** 3).sum()
    rhs = 3 * sum((1 - abs(w[i]) ** 2) * np.prod(np.delete(w, i)) for i in range(4))
    worst = max(worst, abs(lhs - rhs) / (1 + abs(lhs)))
print("Lemma 2.1 identity, max relative error on 20000 random quadruples:", worst)

eps = Fr(1, 10)
floor = Fr(9, 10) * Fr(1, 4) / Fr(11, 20)
assert floor == Fr(9, 22) and floor > Fr(3, 8)
assert Fr(9, 10) * (1 - Fr(1, 2)) == Fr(9, 20)          # mass of h
print("Prop 2.2: remainder floor", floor, "> 3/8")

delta, eta = Fr(1, 200), Fr(1, 1000)
R = 1 + delta
lhs = (1 - eta) * eps / 2
assert 2 * delta + delta ** 2 == Fr(401, 40000)
rhs = 3 * R ** 3 * ((1 - eta) * (2 * delta + delta ** 2) + eta * 1)
assert R ** 2 - 1 < 1
print("Prop 2.3: E W^3 =", lhs, "=", float(lhs), " bound =", float(rhs), " bound < 0.0336:", rhs < Fr(336, 10000))
assert rhs < Fr(336, 10000) < lhs

# ---- 2. three copies ------------------------------------------------------------------------
beta, eps3 = 0.5, 0.01
a = lambda y: 1.0 / (np.pi * np.sqrt(1.0 - y * y))
T5 = lambda y: 16 * y ** 5 - 20 * y ** 3 + 5 * y
h = lambda y: a(y) * T5(y)
f = lambda y: a(y) * (32 * y ** 5 - 34 * y ** 3 + 5 * y) / 3.0
ys = np.linspace(0.002, 0.995, 400)
e1 = max(abs(f(y) - 2 * quad(lambda x: f(x) / x, y, 1, limit=200)[0] - h(y)) / a(y) for y in ys)
F_num = lambda y: quad(lambda x: x * h(x), y, 1, limit=200)[0] / y ** 2
e2 = max(abs(F_num(y) - (8 * y ** 3 - 5 * y) * np.sqrt(1 - y * y) / (3 * np.pi)) for y in ys)
ratio = max(abs(f(y)) / (y * a(y)) for y in ys)
print("Lemma 3.2: equation error", e1, " closed-form error", e2, " max |f|/(y a) =", ratio, "(<= 25/3)")
assert e1 < 1e-6 and e2 < 1e-6 and ratio <= 25 / 3
for c in (1.0, -1.0, 0.3):
    g = lambda y: 2 * beta * eps3 * c * f(y)
    gp = lambda y: max(g(y), 0.0)
    gm = lambda y: max(-g(y), 0.0)
    nu_pos = np.array([gp(y) + 2 * quad(lambda x: gm(x) / x, y, 1, limit=200)[0] for y in ys])
    nu_neg = np.array([gm(y) + 2 * quad(lambda x: gp(x) / x, y, 1, limit=200)[0] for y in ys])
    p_pos = np.array([beta * a(y) * (1 + eps3 * c * T5(y)) for y in ys])
    p_neg = np.array([beta * a(y) * (1 - eps3 * c * T5(y)) for y in ys])
    res_pos, res_neg = p_pos - nu_pos, p_neg - nu_neg
    mass_g = quad(lambda y: abs(g(y)), 0, 1, limit=200)[0]
    r = beta - 3 * mass_g
    print(f"Prop 3.1, c={c}: min (p-nu)/p = {min((res_pos / p_pos).min(), (res_neg / p_neg).min()):.4f},"
          f" residual oddness {np.abs(res_pos - res_neg).max() / p_pos.max():.1e}, r = {r:.4f},"
          f" atom left = {1 - beta - r / 2:.4f}")
    assert (res_pos > 0).all() and (res_neg > 0).all() and 1 - beta - r / 2 >= 0
print("all checks passed")
