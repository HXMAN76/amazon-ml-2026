"""French-style copy of the training data, labels kept. Usage: python src/scripts/frenchify.py OUT_DATA_DIR [--sample N]
Rewrites every record of $BER_DATA/train/train_source{1,2,3}.tsv into French form with a fixed, hand-written dictionary (domain knowledge,
no test data): legal forms (LLC -> SARL, Inc -> SAS, spaced forms kept spaced), category words (bakery -> boulangerie, school -> ecole, ...)
with the category word moved to the front as in French names, connectors (and -> et), noise-type words (sons -> fils, group -> groupe),
and addresses (123 main street -> 123 rue main; state code -> a department, state name -> its region, as French records mix them). The
same word always maps the same way, so a true copy stays a copy and a sibling stays a sibling: the ground truth is copied unchanged.
Entity ids and row order are kept; country becomes France. Run `prepare` with BER_DATA=OUT_DATA_DIR and a separate BER_WORK afterwards.
--sample N prints N rewritten true pairs instead of writing files."""

import multiprocessing as mp
import re
import shutil
import sys
from pathlib import Path

import polars as pl

from ber import config

LEGAL = [(r"private limited|pvt\.? ltd\.?|pvt limited|pvt\.? limited", "sasu"), (r"l\.l\.c\.?", "s.a.r.l."), (r"l l c", "s a r l"), (r"llc", "sarl"),
         (r"l l p", "s n c"), (r"llp", "snc"), (r"pllc", "sasu"), (r"incorporated|inc\.?", "sas"), (r"i n c", "s a s"),
         (r"corporation|corp\.?", "sa"), (r"limited|ltd\.?", "eurl"), (r"lp", "sci"), (r"pc", "ei"), (r"company", "compagnie"), (r"co\.?", "cie")]
MULTI = {"high school": "lycee", "middle school": "college", "elementary school": "ecole elementaire", "primary school": "ecole primaire",
         "medical center": "centre medical", "health center": "centre de sante", "real estate": "immobilier", "day care": "creche",
         "law firm": "cabinet avocats", "car wash": "lavage auto", "auto repair": "garage", "senior center": "foyer", "sports club": "club sportif"}
WORDS = dict(pair.split(":") for pair in (
    "school:ecole academy:academie university:universite institute:institut clinic:clinique hospital:hopital pharmacy:pharmacie "
    "dental:dentaire health:sante care:soins center:centre centre:centre association:association society:societe foundation:fondation "
    "federation:federation committee:comite council:conseil church:eglise bakery:boulangerie kitchen:cuisine pizza:pizzeria inn:auberge "
    "market:marche store:magasin shop:boutique fitness:forme gym:gymnase sports:sportive dance:danse music:musique theater:theatre "
    "theatre:theatre studio:atelier gallery:galerie motors:automobiles repair:reparation construction:construction builders:batiment "
    "roofing:toiture plumbing:plomberie electric:electricite electrical:electricite cleaning:nettoyage transportation:transports "
    "trucking:transports logistics:logistique travel:voyages insurance:assurance financial:finance bank:banque investments:investissements "
    "realty:immobilier properties:proprietes law:droit legal:juridique consulting:conseil management:gestion media:medias "
    "technologies:technologies technology:technologie tech:technologie software:logiciel systems:systemes engineering:ingenierie "
    "industries:industries manufacturing:fabrication supply:fournitures trading:negoce exports:exportations imports:importations "
    "farm:ferme farms:fermes agro:agricole agricultural:agricole foods:alimentation food:alimentation catering:traiteur books:librairie "
    "printing:imprimerie textiles:textiles fashion:mode jewelers:bijouterie jewellers:bijouterie optical:optique veterinary:veterinaire "
    "nursing:ehpad home:maison homes:maison house:maison apartments:residence youth:jeunes kids:enfants children:enfants parents:parents "
    "friends:amis family:famille community:communaute regional:regionale local:locale city:ville united:unis group:groupe "
    "services:services service:services development:developpement sons:fils brothers:freres bros:freres associates:associes "
    "partners:partenaires enterprises:entreprises enterprise:entreprise holdings:holding and:et of:de the:le").split())
CATEGORY = set(WORDS.values()) - {"et", "de", "le", "groupe", "services", "developpement", "fils", "freres", "associes", "partenaires",
                                  "entreprises", "entreprise", "holding", "unis", "ville", "locale", "regionale"}
CATEGORY |= {v.split()[0] for v in MULTI.values()}
FR_LEGAL = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "cie", "compagnie"}
STREET = {"street": "rue", "st": "rue", "avenue": "avenue", "ave": "avenue", "av": "avenue", "road": "route", "rd": "route", "boulevard": "boulevard",
          "blvd": "boulevard", "drive": "allee", "dr": "allee", "lane": "chemin", "ln": "chemin", "court": "impasse", "ct": "impasse", "place": "place",
          "pl": "place", "way": "voie", "circle": "rond point", "cir": "rond point", "parkway": "cours", "pkwy": "cours", "highway": "route nationale",
          "hwy": "route nationale", "square": "square", "sq": "square", "terrace": "residence", "trail": "sentier", "marg": "rue", "nagar": "quartier"}
ADDRW = {"north": "nord", "south": "sud", "east": "est", "west": "ouest", "unit": "appartement", "apt": "appartement", "apartment": "appartement",
         "suite": "bat", "ste": "bat", "floor": "etage", "usa": "france", "india": "france", "united states": "france"}
REGIONS = [("hauts de france", ["nord", "pas de calais", "somme", "oise", "aisne"]), ("ile de france", ["paris", "yvelines", "essonne", "val d oise"]),
           ("nouvelle aquitaine", ["gironde", "landes", "dordogne", "vienne"]), ("pays de la loire", ["loire atlantique", "sarthe", "vendee", "maine et loire"]),
           ("occitanie", ["haute garonne", "herault", "gard", "aude"]), ("bretagne", ["finistere", "morbihan", "ille et vilaine", "cotes d armor"]),
           ("grand est", ["bas rhin", "haut rhin", "moselle", "marne"]), ("normandie", ["calvados", "manche", "seine maritime", "eure"]),
           ("auvergne rhone alpes", ["rhone", "isere", "savoie", "puy de dome"]), ("provence alpes cote d azur", ["var", "vaucluse", "alpes maritimes"]),
           ("bourgogne franche comte", ["cote d or", "doubs", "jura", "yonne"]), ("centre val de loire", ["loiret", "cher", "indre", "eure et loir"]),
           ("corse", ["corse du sud", "haute corse"])]
STATES = ("al alabama,ak alaska,az arizona,ar arkansas,ca california,co colorado,ct connecticut,de delaware,fl florida,ga georgia,hi hawaii,"
          "id idaho,il illinois,in indiana,ia iowa,ks kansas,ky kentucky,la louisiana,me maine,md maryland,ma massachusetts,mi michigan,"
          "mn minnesota,ms mississippi,mo missouri,mt montana,ne nebraska,nv nevada,nh new hampshire,nj new jersey,nm new mexico,ny new york,"
          "nc north carolina,nd north dakota,oh ohio,ok oklahoma,or oregon,pa pennsylvania,ri rhode island,sc south carolina,sd south dakota,"
          "tn tennessee,tx texas,ut utah,vt vermont,va virginia,wa washington,wv west virginia,wi wisconsin,wy wyoming,dc district of columbia,"
          "mh maharashtra,dl delhi,ka karnataka,tn tamil nadu,up uttar pradesh,gj gujarat,wb west bengal,rj rajasthan,kl kerala,hr haryana,"
          "pb punjab,ap andhra pradesh,ts telangana,mp madhya pradesh,br bihar,or odisha")
STATE_FR: dict[str, str] = {}
for i, s in enumerate(STATES.split(",")):
    code, name = s.split(" ", 1)
    reg, deps = REGIONS[i % len(REGIONS)]
    STATE_FR.setdefault(code, deps[(i // len(REGIONS)) % len(deps)])
    STATE_FR.setdefault(name, reg)
_LEGAL_RE = [(re.compile(rf"(?<![a-z0-9]){p}(?![a-z0-9])"), r) for p, r in LEGAL]
_MULTI_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, MULTI), key=len, reverse=True)) + r")\b")
_WORD_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, WORDS), key=len, reverse=True)) + r")\b")
_ST_RE = re.compile(r"^(\d+[a-z]?)\s+(.+?)\s+(" + "|".join(sorted(STREET, key=len, reverse=True)) + r")\b\.?(.*)$")
_STREET_RE = re.compile(r"\b(" + "|".join(sorted(STREET, key=len, reverse=True)) + r")\b")
_ADDRW_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ADDRW), key=len, reverse=True)) + r")\b")
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATE_FR), key=len, reverse=True)) + r")\b")


def fr_name(raw: str) -> str:
    s = raw.lower().replace("&", " and ")
    for rx, rep in _LEGAL_RE:
        s = rx.sub(rep, s)
    s = _MULTI_RE.sub(lambda m: MULTI[m.group(1)], s)
    s = _WORD_RE.sub(lambda m: WORDS[m.group(1)], s)
    t = s.split()
    core = [w for w in t if w.strip(".,") not in FR_LEGAL]
    if len(core) >= 2 and core[-1] in CATEGORY:  # French order: the category word leads ("boulangerie smith")
        cat = core[-1]
        i = len(t) - 1 - t[::-1].index(cat)
        t = [cat] + t[:i] + t[i + 1:]
    return " ".join(t)


def fr_addr(raw: str) -> str:
    s = raw.lower()
    m = _ST_RE.match(s)
    if m:
        s = f"{m.group(1)} {STREET[m.group(3)]} {m.group(2)}{m.group(4)}"
    else:
        s = _STREET_RE.sub(lambda m: STREET[m.group(1)], s)
    s = _ADDRW_RE.sub(lambda m: ADDRW[m.group(1)], s)
    return _STATE_RE.sub(lambda m: STATE_FR[m.group(1)], s)


def _rows(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(fr_name(n or ""), fr_addr(a or "")) for n, a in rows]


def rewrite(src: Path, dst: Path) -> int:
    d = pl.read_csv(src, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.all().fill_null(""))
    rows = list(zip(d["business_name"].to_list(), d["business_address"].to_list()))
    step = 200_000
    with mp.get_context("fork").Pool(60) as pool:
        out = [r for part in pool.map(_rows, [rows[i:i + step] for i in range(0, len(rows), step)]) for r in part]
    d = d.with_columns(pl.Series("business_name", [r[0] for r in out]), pl.Series("business_address", [r[1] for r in out]), pl.lit("France").alias("country"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    d.write_csv(dst, separator="\t", quote_style="never")
    return d.height


def main() -> None:
    out = Path(sys.argv[1])
    P = config.paths()
    src = P["data"] / "train"
    if "--sample" in sys.argv:
        n = int(sys.argv[sys.argv.index("--sample") + 1])
        pq = P["parquet"] / "train"
        s1 = pl.read_parquet(pq / "source1.parquet", columns=["rid", "business_name", "business_address"]).rename({"rid": "s1_rid"})
        lab = pl.read_parquet(pq / "labels.parquet").sample(n, seed=1)
        for r in lab.join(s1, on="s1_rid").iter_rows(named=True):
            o = pl.read_parquet(pq / f"source{r['src']}.parquet", columns=["rid", "business_name", "business_address"]).filter(pl.col("rid") == r["other_rid"]).row(0, named=True)
            print(f"S1  {r['business_name']} | {r['business_address']}\n  -> {fr_name(r['business_name'] or '')} | {fr_addr(r['business_address'] or '')}")
            print(f"S{r['src']}  {o['business_name']} | {o['business_address']}\n  -> {fr_name(o['business_name'] or '')} | {fr_addr(o['business_address'] or '')}\n")
        return
    for i in (1, 2, 3):
        n = rewrite(src / f"train_source{i}.tsv", out / "train" / f"train_source{i}.tsv")
        print(f"train_source{i}: {n} rows rewritten", flush=True)
    shutil.copy(src / "train_ground_truth.tsv", out / "train" / "train_ground_truth.tsv")


if __name__ == "__main__":
    main()
