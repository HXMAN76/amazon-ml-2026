"""The France-style copy and decoy operators of stages/xenc_fr.py (train records only)."""

import numpy as np

from ber.stages.xenc_fr import NOISE, glue, initials, noise_swap, spell, swap


def test_copy_operators():
    assert initials("lille club") == "lc"
    assert initials("professionnel elementaire ecole maternelle") == "pee"
    assert spell("acme builders llc") == "acme builders l l c"
    assert spell("acme builders") is None
    rng = np.random.default_rng(0)
    assert glue("ecole maternelle sarl", "ecole maternelle", rng) == "ecolematernelle sarl"
    assert glue("acme llc", "acme", rng) is None


def test_swap_uses_common_words_and_never_noise():
    rng = np.random.default_rng(1)
    common = np.array(["club", "school", "clinic", "center"])
    out = {swap("nje club llc", {"nje", "club"}, common, set(common), rng) for _ in range(50)}
    assert out <= {"nje school llc", "nje clinic llc"} and out
    assert swap("nje services", {"nje", "services"}, common, set(common) | NOISE, rng) is None  # noise words are true-copy noise in train


def test_noise_swap_is_a_true_copy_operator():
    rng = np.random.default_rng(2)
    out = {noise_swap("nje club llc", {"nje", "club"}, {"club"}, rng) for _ in range(30)}
    assert out and out <= {f"nje {w} llc" for w in NOISE}
