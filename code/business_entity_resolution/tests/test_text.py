import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import text  # noqa: E402


def test_html_entities_and_ampersand():
    assert text.parse_name("Elite &amp; Co").name1 == text.parse_name("Elite & Co").name1 == "elite and company"


def test_leetspeak_only_in_mixed_tokens():
    assert text.parse_name("Elite  + C0mpany").name1.endswith("company")
    assert "secure" in text.parse_name("Cabrera 5ecure Sciences LP").name1
    assert text.parse_name("3M Company").name1.startswith("3m")  # real alphanumerics untouched
    assert "24x7" in text.parse_name("Shop 24x7").name1


def test_dba_split_gives_alias():
    p = text.parse_name("Quoavi Co doing business as Asset Building Committee")
    assert p.has_alias and p.name1.startswith("quoavi") and p.name2 == "asset building committee"
    p = text.parse_name("Arcbrixx dba Rays Office")
    assert p.name1 == "arcbrixx" and p.name2 == "rays office"


def test_domain_names():
    p = text.parse_name("ipower.com")
    assert p.is_domain and p.name1 == "ipower"
    p = text.parse_name("Cabrera Secure Sciences | www.cabreras.com")
    assert not p.is_domain and p.name1 == "cabrera secure sciences" and p.name2 == "cabreras"
    assert text.parse_name("láwrenceventures.com").name1 == "lawrenceventures"
    p = text.parse_name("Vision Health of Davis Center Inc | [www.visionhea.com]")
    assert p.name2 == "visionhea" and not p.is_domain


def test_legal_forms_moved_to_legal_field():
    p = text.parse_name("Lawrence Ventures Private Limited")
    assert p.core1 == "lawrence ventures" and p.legal == "limited private"
    assert text.parse_name("Marina Ecole France Sarl").legal == "sarl"
    assert text.parse_name("Limited").core1 == "limited"  # never empty


def test_latin_accents_stripped_indic_preserved():
    assert text.parse_name("Léarning Café").name1 == "learning cafe"
    hi = "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert text.parse_name(hi).name1 == hi
    assert text.nonlatin_frac(hi) > 0.99 and text.nonlatin_frac("Ram Marketing") == 0.0


def test_address_normalisation():
    assert text.norm_address("1344 MCDONALD HILL RD, CHICAGOCDP, IL") == "1344 mcdonald hill road chicago il"
    assert text.norm_address("Door No 467 401, Omega Business Park") == "467 401 omega business park"
    assert text.norm_address("दिल्ली, JD-36B, PITAMPURA") == "दिल्ली jd 36b pitampura"
    assert text.norm_address("") == "" and text.norm_address(None) == ""


def test_country_aliases_open_set():
    assert text.norm_country("USA") == "us" and text.norm_country("France") == "france"
    assert text.norm_country("Germany") == "germany"  # unseen labels pass through


def test_ascii_fast_path_equals_slow_path():
    import random
    import string

    rng = random.Random(0)
    for _ in range(20000):
        s = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 40)))
        assert text._tokens(s) == text._tokens_slow(s), repr(s)


def test_street_and_saint_share_one_token():
    a = text.norm_address("12 Rue St Jean, Lille")
    assert a == text.norm_address("12 Rue Saint Jean, Lille") == text.norm_address("12 Rue Street Jean, Lille")
