import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ber.decoys import make_decoy  # noqa: E402


def test_decoy_changes_name_or_number_but_keeps_shape():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n, a = make_decoy("jarlent states llc", "312 main street chicago il", rng)
        assert (n, a) != ("jarlent states llc", "312 main street chicago il")
        assert len(n) == len("jarlent states llc")  # letters replaced in place, never inserted or dropped
        assert n.split()[1:] == ["states", "llc"] or n.split()[0] == "jarlent"  # only the longest word is edited
        num = a.split()[0]
        assert num.isdigit() and abs(len(num) - 3) <= 1 and a.split()[1:] == "main street chicago il".split()


def test_decoy_without_number_or_long_word():
    rng = np.random.default_rng(1)
    n, a = make_decoy("ab cd", "main street", rng)
    assert (n, a) == ("ab cd", "main street")  # nothing safe to change: returned unchanged
    n, a = make_decoy("ab cd", "7 main street", rng)
    assert n == "ab cd" and a != "7 main street" and a.endswith(" main street")
