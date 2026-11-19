"""Generate the labeled dev sets: data/relevance_dev.jsonl and data/extraction_dev.jsonl.

The messages are fictional WhatsApp messages / voice-note transcripts from a community group, written by a local
instruction model (default: Qwen/Qwen2.5-3B-Instruct) from the scenarios below. The scenarios come from the
label guide and Dandelion Africa's program areas.

relevance   Event / Chit-chat messages (label guide below), for relevance.py's threshold.
extraction  Event messages only, each about one ACTIVITIES item and either an exact head count (in digits, in words,
            or next to distracting numbers such as a time and an amount of money) or no number at all. Labels:
            "activity" (the ACTIVITIES item), "category" (its extract.ACTIVITY_CATEGORIES name) and
            "participants" (the count, or null).

Labels:
  Event     : the message reports a field activity that has ALREADY TAKEN PLACE (a meeting, training, outreach,
              screening, distribution, home visit, demonstration...), with or without numbers, even if it also
              contains greetings or thanks.
  Chit-chat : everything else: greetings, thanks, congratulations, condolences, prayers, jokes, personal news;
              plans, reminders or questions about FUTURE activities; apologies for missing an activity; logistics,
              money and transport coordination; requests for reports or photos.

Since we don't want to be reporting on future activities, those should ideally be classified as chit-chat.

Each row's labels come from its scenario ("label_source": "scenario"). We can manually review the labels and set "reviewed_by_human": true

Run:  python tools/make_dev_sets.py relevance
      python tools/make_dev_sets.py extraction
      (a few minutes each on a GPU, tested on a 4090)
"""
import argparse
import itertools
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# AI-generated categories/activities/other topics that may be relevant to Dandelion's activities
ACTIVITIES = [
    "a family planning outreach", "a maternal health talk for pregnant mothers", "a menstrual hygiene session at a school",
    "a gender-based violence awareness meeting", "a table banking (savings group) meeting", "a kitchen garden training",
    "a farmer training on drought-tolerant seeds", "a youth mentorship session", "HIV testing and counselling",
    "a child immunization day", "a nutrition and cooking demonstration", "a community clean-up",
    "a tree planting day", "home visits to new mothers by a community health volunteer",
    "a distribution of sanitary pads", "a business skills training for women", "a borehole committee meeting",
    "a poultry keeping training",
]
# The category of each activity in extract.ACTIVITY_CATEGORIES (the extraction set's "category" label)
ACTIVITY_CATEGORY = {
    "a family planning outreach": "Family planning outreach",
    "a maternal health talk for pregnant mothers": "Maternal health talk",
    "a menstrual hygiene session at a school": "Menstrual hygiene session",
    "a gender-based violence awareness meeting": "Gender-based violence awareness session",
    "a table banking (savings group) meeting": "Table banking (savings) group meeting",
    "a kitchen garden training": "Kitchen garden training",
    "a farmer training on drought-tolerant seeds": "Farmer training",
    "a youth mentorship session": "Youth mentorship session",
    "HIV testing and counselling": "HIV testing and counselling",
    "a child immunization day": "Immunization outreach",
    "a nutrition and cooking demonstration": "Nutrition session",
    "a community clean-up": "Community clean-up",
    "a tree planting day": "Tree planting",
    "home visits to new mothers by a community health volunteer": "Home visits",
    "a distribution of sanitary pads": "Distribution of supplies",
    "a business skills training for women": "Business skills training",
    "a borehole committee meeting": "Water committee meeting",
    "a poultry keeping training": "Livestock and poultry training",
}
EVENT_STYLES = [
    "gives the number of people who came", "gives no numbers at all", "is short and casual, one or two sentences",
    "starts with a greeting and ends with thanks to the team", "mentions a problem that happened during it",
    "mixes in a few Swahili words",
]
CHITCHAT_SCENARIOS = [
    # plain chit-chat
    "a good-morning greeting to the group with a blessing", "congratulating a member on a new baby",
    "congratulating a member's child on passing exams", "condolences to a member whose relative passed away",
    "thanking everyone for their hard work this month, without describing any activity", "a light joke about the rain",
    "sharing personal news about a family wedding", "asking for prayers for a sick member",
    # coordination and questions
    "asking who has the receipt book", "coordinating money contributions for a gift via M-Pesa",
    "asking for help sending photos on WhatsApp", "asking the coordinator to send the monthly report template",
    "asking whether the transport money has been sent",
    # hard cases: about an activity, but not a report of one that took place
    "announcing {a} that will happen next week and asking people to come",
    "reminding members about {a} tomorrow morning and what to bring",
    "apologizing for missing {a} because of a family emergency",
    "asking whether {a} is still happening on Saturday",
    "suggesting that the group should organize {a} next month",
    "asking how many people came to {a} because they could not attend",
]
# Extraction set: how the head count appears ({n} = the count, {w} = the count in words)
COUNT_VARIANTS = [
    ("says that exactly {n} people came, writing the number {n} in digits", True),
    ("says that exactly {w} people came, writing the number in words ({w})", True),
    ("does not say how many people came and contains no numbers at all", False),
    ("says that exactly {n} people came (in digits), and also mentions the time it started (in digits) and "
     "how many shillings were collected or spent", True),
]
EXTRACTION_STYLES = ["is short and casual", "starts with a greeting and ends with thanks to the team",
                     "mentions a problem that happened during it", "is written by the person who led it"]

PROMPT = (
    "Write one WhatsApp message posted in a group chat of community health volunteers and women's group members in "
    "rural Kenya. It is a transcript of a voice note, in simple spoken English. Invent the names and places. "
    "The message {what}.\nWrite only the message itself, 1 to 5 sentences, no quotation marks."
)

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    """1-999 in words: 42 -> 'forty-two', 130 -> 'one hundred and thirty'."""
    if n >= 100:
        rest = n % 100
        return f"{_ONES[n // 100]} hundred" + (f" and {number_words(rest)}" if rest else "")
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else "")


def relevance_scenarios(seed: int, n_events: int = 60, n_chitchat: int = 60) -> list[dict]:
    rng = random.Random(seed)
    events = [{"what": f"reports on {a} that already took place earlier today or this week, and {s}", "label": "Event"}
              for a, s in itertools.product(ACTIVITIES, EVENT_STYLES)]
    chats = []  # each plain scenario twice (sampling makes the two messages differ), each hard case with 6 activities
    for sc in CHITCHAT_SCENARIOS:
        if "{a}" in sc:
            chats += [{"what": f"is {sc.format(a=a)}", "label": "Chit-chat"} for a in rng.sample(ACTIVITIES, 6)]
        else:
            chats += [{"what": f"is {sc}", "label": "Chit-chat"}] * 2
    return rng.sample(events, n_events) + rng.sample(chats, min(n_chitchat, len(chats)))


def extraction_scenarios(seed: int) -> list[dict]:
    """Every activity with every count variant (18 x 4 = 72 Event messages)."""
    rng = random.Random(seed)
    out = []
    for activity, (variant, has_count) in itertools.product(ACTIVITIES, COUNT_VARIANTS):
        n = rng.randint(8, 150)
        count = variant.format(n=n, w=number_words(n))
        what = (f"reports on {activity} that already took place earlier today or this week, "
                f"{rng.choice(EXTRACTION_STYLES)}, and {count}")
        out.append({"what": what, "label": "Event", "activity": activity, "category": ACTIVITY_CATEGORY[activity],
                    "participants": n if has_count else None})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("set", choices=["relevance", "extraction"])
    p.add_argument("-m", "--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-o", "--output", help="Default: data/<set>_dev.jsonl")
    args = p.parse_args()
    output = args.output or os.path.join(DATA, f"{args.set}_dev.jsonl")
    scenarios = relevance_scenarios(args.seed) if args.set == "relevance" else extraction_scenarios(args.seed)

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for i, sc in enumerate(scenarios):
        prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(what=sc["what"])}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=160, do_sample=True, temperature=0.9, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True).strip().strip('"')
        labels = {k: v for k, v in sc.items() if k != "what"}
        rows.append({"id": i + 1, "text": text, **labels, "scenario": sc["what"], "label_source": "scenario",
                     "reviewed_by_human": False})
        print(f"[{i + 1}] {labels}: {text[:90]}")

    with open(output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
