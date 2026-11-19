"""Build data/kenya_places.txt, the place-name list used by redact.py.

Sources (nothing is hand-picked from any test set):
  - the county / district / sub-county columns of the FarmerChat Kenya dataset, EXCLUDING the 50 rows in our
    labeled dev set (data/farmerchat_kenya_sample.jsonl), so that dev set stays a fair test
  - the 47 Kenyan counties (a public list)

Run:  python tools/build_place_list.py
"""
import json
import os
import re

from datasets import load_dataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SAMPLE = os.path.join(ROOT, "data", "farmerchat_kenya_sample.jsonl")
OUT = os.path.join(ROOT, "data", "kenya_places.txt")

COUNTIES = ["Mombasa", "Kwale", "Kilifi", "Tana River", "Lamu", "Taita-Taveta", "Garissa", "Wajir", "Mandera",
            "Marsabit", "Isiolo", "Meru", "Tharaka-Nithi", "Embu", "Kitui", "Machakos", "Makueni", "Nyandarua",
            "Nyeri", "Kirinyaga", "Murang'a", "Kiambu", "Turkana", "West Pokot", "Samburu", "Trans-Nzoia",
            "Uasin Gishu", "Elgeyo-Marakwet", "Nandi", "Baringo", "Laikipia", "Nakuru", "Narok", "Kajiado",
            "Kericho", "Bomet", "Kakamega", "Vihiga", "Bungoma", "Busia", "Siaya", "Kisumu", "Homa Bay", "Migori",
            "Kisii", "Nyamira", "Nairobi"]

# Generic words that appear in geo columns but would wrongly redact ordinary text
GENERIC = {"central", "east", "west", "north", "south", "town", "market", "soy", "county", "kenya"}


def clean(value):
    value = (value or "").strip()
    if not value or len(value) > 40 or re.search(r"\d", value):
        return None
    return value


def main():
    with open(SAMPLE, encoding="utf-8") as f:
        held_out = {json.loads(line)["row_index"] for line in f if line.strip()}

    ds = load_dataset("DigiGreen/farmerchat-queries", "kenya", split="train")
    places = set(COUNTIES)
    for i, row in enumerate(ds):
        if i in held_out:
            continue
        for col in ("user_geo_level2", "user_geo_level3"):
            value = clean(row.get(col))
            if not value:
                continue
            places.add(value)
            if value.endswith(" County"):
                places.add(value[: -len(" County")])  # also match the bare name, e.g. "Nyeri"

    places = {p for p in places if len(p) >= 4 and p.lower() not in GENERIC}
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(places)) + "\n")
    print(f"Wrote {len(places)} place names to {OUT} (held out {len(held_out)} dev-set rows)")


if __name__ == "__main__":
    main()
