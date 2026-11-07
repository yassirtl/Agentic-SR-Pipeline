"""Meta-analysis maths, in plain Python (no SciPy, no R): every formula is written out so
it can be checked by hand (tests/test_synthesis.py does, on a 3-study example).

Continuous outcomes
  Hedges' g (standardised mean difference, intervention minus control), Borenstein et al.
  2009, ch. 4:
    s_pooled = sqrt(((n1-1) s1^2 + (n2-1) s2^2) / (n1 + n2 - 2))
    d = (m1 - m2) / s_pooled,   V_d = (n1 + n2) / (n1 n2) + d^2 / (2 (n1 + n2))
    J = 1 - 3 / (4 (n1 + n2 - 2) - 1),   g = J d,   V_g = J^2 V_d
  The SMD puts different instruments (GDS-15, BDI, PHQ-9) on one scale.

Random effects, DerSimonian-Laird (1986):
    w_i = 1 / v_i,  fixed = sum(w y) / sum(w),  Q = sum(w (y - fixed)^2),  df = k - 1
    C = sum(w) - sum(w^2) / sum(w),  tau^2 = max(0, (Q - df) / C)
    w*_i = 1 / (v_i + tau^2),  pooled = sum(w* y) / sum(w*),  SE = sqrt(1 / sum(w*))
    95% CI = pooled +- 1.96 SE,  I^2 = max(0, (Q - df) / Q)

Diagnostic accuracy
  Per study: sensitivity = TP / (TP + FN), specificity = TN / (TN + FP), each with the
  Wilson score 95% interval (good coverage near 0 and 1, unlike the Wald interval).
  Pooled (>= 3 studies): univariate random effects on the logit scale, sensitivity and
  specificity separately (logit p, variance 1/x + 1/(n - x); 0.5 added to x and n - x when
  either is 0), DerSimonian-Laird, back-transformed. The standard model is the bivariate
  one (Reitsma 2005), which also models the correlation between sensitivity and
  specificity; it needs a mixed-model fit (R mada / metafor). The univariate model is a
  stated limitation.

Vote counting by direction of effect (SWiM, Campbell 2020): two-sided sign test on the
studies favouring each side (studies with no difference left out).
"""

import math

Z95 = 1.959963984540054  # standard normal 97.5% quantile


def normal_p(z: float) -> float:
    """Two-sided p value of a standard normal z."""
    return math.erfc(abs(z) / math.sqrt(2))


def hedges_g(n1: float, m1: float, s1: float, n2: float, m2: float, s2: float
             ) -> tuple[float, float]:
    """(g, variance of g): arm 1 = intervention, arm 2 = control."""
    if min(n1, n2) < 2 or s1 < 0 or s2 < 0:
        raise ValueError("each arm needs n >= 2 and a non-negative SD")
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if sp == 0:
        raise ValueError("pooled SD is 0")
    d = (m1 - m2) / sp
    vd = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    return j * d, j ** 2 * vd


def ci(est: float, var: float) -> tuple[float, float]:
    se = math.sqrt(var)
    return est - Z95 * se, est + Z95 * se


def dersimonian_laird(effects: list[float], variances: list[float]) -> dict:
    """Random-effects pooled estimate (see module docstring)."""
    k = len(effects)
    if k == 0 or k != len(variances) or any(v <= 0 for v in variances):
        raise ValueError("need k >= 1 effects with positive variances")
    w = [1 / v for v in variances]
    sw = sum(w)
    fixed = sum(wi * y for wi, y in zip(w, effects)) / sw
    q = sum(wi * (y - fixed) ** 2 for wi, y in zip(w, effects))
    df = k - 1
    c = sw - sum(wi ** 2 for wi in w) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    ws = [1 / (v + tau2) for v in variances]
    sws = sum(ws)
    pooled = sum(wi * y for wi, y in zip(ws, effects)) / sws
    se = math.sqrt(1 / sws)
    z = pooled / se
    return {"k": k, "pooled": pooled, "se": se, "ci_low": pooled - Z95 * se,
            "ci_high": pooled + Z95 * se, "z": z, "p": normal_p(z), "q": q, "df": df,
            "tau2": tau2, "i2": max(0.0, (q - df) / q) if q > 0 else 0.0,
            "weights": [x / sws for x in ws], "fixed": fixed}


def wilson(x: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """(proportion, lower, upper): Wilson score interval."""
    if n <= 0 or not 0 <= x <= n:
        raise ValueError("need 0 <= x <= n and n > 0")
    p = x / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def expit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def pooled_proportion(events: list[int], totals: list[int]) -> dict:
    """Univariate random-effects pooling of proportions on the logit scale."""
    ys, vs = [], []
    for x, n in zip(events, totals):
        a, b = float(x), float(n - x)
        if a == 0 or b == 0:  # continuity correction, only here (never in the 2x2 table)
            a, b = a + 0.5, b + 0.5
        ys.append(math.log(a / b))
        vs.append(1 / a + 1 / b)
    res = dersimonian_laird(ys, vs)
    return {**res, "logit_pooled": res["pooled"], "pooled": expit(res["pooled"]),
            "ci_low": expit(res["ci_low"]), "ci_high": expit(res["ci_high"])}


def sign_test(positive: int, negative: int) -> float | None:
    """Two-sided exact binomial (p = 0.5) test on the direction counts; None if no study."""
    n = positive + negative
    if n == 0:
        return None
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
