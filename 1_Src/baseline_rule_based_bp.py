"""
event_identification.py
复现 Jurkiewicz & Nawrocki (2015) 的自动事件识别方法（规则版）。
依赖: spaCy (pip install spacy; python -m spacy download en_core_web_sm)
"""

from typing import List, Dict, Tuple, Any
from imports import StanfordCoreNLP
from collections import defaultdict
nlp = StanfordCoreNLP(r'../0_Data/stanford-corenlp-4.0.0', memory='8g')
# -----------------------
# 1) Observed set (Table 4 + Table 7)
# Format: (Performer, Activity, Property, EventType)
# Use '*' for wildcard property (valid for any property)
# -----------------------
OBSERVED = [
    # Table 4 entries (converted)
    ("USER", "ENTER", "SET", "WrongData"),
    ("USER", "ENTER", "COMPOUND", "Incomplete"),
    ("USER", "ENTER", "SET", "AlreadyExist"),
    ("USER", "ENTER", "REMOTE", "ConnectionProblem"),
    ("USER", "ENTER", "AT_LEAST", "TooLittleLeft"),
    ("SYSTEM", "DISPLAY", "SET", "NoDataDefined"),
    ("USER", "SELECT", "SET", "NoDataDefined"),
    ("USER", "SELECT", "SET", "AlreadySelected"),
    ("USER", "SELECT", "NO_MORE_THAN", "TooMuchSelected"),
    ("USER", "SELECT", "*", "NoObjectSelected"),
    ("USER", "DELETE", "*", "DoesNotExist"),
    ("USER", "READ", "*", "DoesNotExist"),
    ("USER", "LINK", "*", "Incomplete"),
    ("USER", "SELECT", "AT_LEAST", "TooLittleSelected"),
    ("USER", "CONFIRM", "*", "LackOfConfirmation"),
    ("SYSTEM", "ADD", "LONG_LASTING", "WantsToInterrupt"),
    ("USER", "ADD", "TIMEOUTABLE", "TooLateForDoingThis"),
    # Table 7 additions
    ("SYSTEM", "FETCH", "REMOTE", "ConnectionProblem"),
    ("USER", "UPDATE", "SET", "WrongData"),
    ("USER", "ENTER", "COMPOUND", "Incomplete"),  # duplicate but OK
    ("USER", "DELETE", "AT_LEAST", "TooLittleLeft"),
]

# Normalize Observed to a list of dicts for readability
OBSERVED = [tuple(o) for o in OBSERVED]

# -----------------------
# 2) Event description templates (based on Table 5)
# -----------------------
EVENT_TEMPLATES = {
    "WrongData": "{subject} {verb_past} wrong data of {object}",
    "TooLittleLeft": "Not enough {object_plural} left",
    "Incomplete": "{subject} {verb_past}, incomplete data of {object}",
    "AlreadyExist": "{object} already exists",
    "ConnectionProblem": "Connection problem occurred",
    "TooMuchSelected": "Too many {object} selected by {subject}",
    "NoObjectSelected": "No {object} chosen by {subject}",
    "AlreadySelected": "{object} has already been selected by {subject}",
    "TooLittleSelected": "Too little {object_plural} chosen by {subject}",
    "DoesNotExist": "Given {object} does not exist",
    "LackOfConfirmation": "{subject} has not confirmed",
    "NoDataDefined": "No {object} has been defined",
    "WantsToInterrupt": "Operation canceled",
    "TooLateForDoingThis": "Too late for this operation",
}

# -----------------------
# 3) Activity mapping heuristics (verb lemma -> activity type)
#    This is a heuristic map; 可根据需要扩展
# -----------------------
VERB_TO_ACTIVITY = {
    "add": "ADD",
    "create": "ADD",
    "insert": "ADD",
    "enter": "ENTER",
    "type": "ENTER",
    "write": "ENTER",
    "link": "LINK",
    "attach": "LINK",
    "read": "READ",
    "view": "READ",
    "show": "DISPLAY",
    "display": "DISPLAY",
    "update": "UPDATE",
    "change": "UPDATE",
    "modify": "UPDATE",
    "delete": "DELETE",
    "remove": "DELETE",
    "select": "SELECT",
    "choose": "SELECT",
    "pick": "SELECT",
    "validate": "VALIDATE",
    "check": "VALIDATE",
    "confirm": "CONFIRM",
    "accept": "CONFIRM",
    "fetch": "FETCH",
    "import": "FETCH",
}

# -----------------------
# 4) Helper functions
# -----------------------
def extract_svo_corenlp(step_text: str):
    """
    用 Stanford CoreNLP 抽取 SVO（含异常处理）。
    返回 {'subject', 'verb', 'verb_past', 'object', 'object_plural'}
    """
    tokens = nlp.word_tokenize(step_text)
    pos_tags = nlp.pos_tag(step_text)
    deps = nlp.dependency_parse(step_text)

    # 1. 找 root 动词
    root_verb = None
    for rel, gov, dep in deps:
        if rel == 'root':
            if 0 < dep <= len(tokens):
                root_verb = tokens[dep - 1]
            break

    # 2. fallback: 如果没找到 root_verb，则尝试用第一个动词
    if not root_verb:
        for word, pos in pos_tags:
            if pos.startswith("VB"):
                root_verb = word
                break

    # 3. 找 nsubj / dobj
    subj, obj = None, None
    for rel, gov, dep in deps:
        if rel in ('nsubj', 'nsubjpass') and 0 < gov <= len(tokens) and tokens[gov - 1] == root_verb:
            subj = tokens[dep - 1]
        if rel in ('dobj', 'obj', 'pobj') and 0 < gov <= len(tokens) and tokens[gov - 1] == root_verb:
            obj = tokens[dep - 1]

    # 4. 生成过去式（安全）
    verb_past = ""
    if root_verb:
        verb_past = root_verb  # 默认
        for word, pos in pos_tags:
            if word == root_verb:
                if pos in ('VBD', 'VBN'):
                    verb_past = word
                else:
                    # 简单过去式近似
                    if not root_verb.endswith("ed"):
                        verb_past = root_verb + "ed"
                break

    return {
        "subject": subj or "",
        "verb": (root_verb or "").lower(),
        "verb_past": verb_past or "",
        "object": obj or "",
        "object_plural": (obj + "s") if obj else ""
    }


def map_verb_to_activity(verb_lemma: str) -> str:
    return VERB_TO_ACTIVITY.get(verb_lemma.lower(), None)

# -----------------------
# 5) Inference engine: simple & compound rules
# -----------------------
def match_observed(perf: str, activity: str, obj_properties: List[str]) -> List[str]:
    """
    Apply simple inference rule:
    If there exists (perf, activity, P, E) in OBSERVED and P in obj_properties (or P == '*'),
    then E is a candidate.
    """
    events = []
    for (R, A, P, E) in OBSERVED:
        if R != perf:
            continue
        if A != activity:
            continue
        if P == "*" or P in obj_properties:
            events.append(E)
    return events

def match_compound(perf: str, activity: str, obj: str, asp_map: Dict[Tuple[str,str], List[str]]) -> List[str]:
    """
    Compound rule:
    If ASP(obj) contains (P,Activity) and Observed contains (R,Activity,P,E) -> E
    Here asp_map: keys (object_lower, activity) -> list of properties (like TIMEOUTABLE, LONG_LASTING)
    """
    events = []
    key = (obj.lower(), activity)
    if key not in asp_map:
        return events
    asp_props = asp_map[key]  # properties that are activity-sensitive for this object+activity
    for prop in asp_props:
        for (R, A, P, E) in OBSERVED:
            if R == perf and A == activity and P == prop:
                events.append(E)
    return events

# -----------------------
# 6) Main processing function
# -----------------------
def identify_events_for_step(step_text: str,
                             actors: List[str],
                             info_objects: List[str],
                             afp_map: Dict[str, List[str]],
                             asp_map: Dict[Tuple[str,str], List[str]],
                             prefer_actor_detection: bool = True) -> List[Dict[str,str]]:
    """
    step_text: the use-case step in natural language
    actors: list of actor names (strings), e.g. ["Author", "System", "Customer"]
    info_objects: list of info object names, e.g. ["Post", "Article", "Credit card"]
    afp_map: Activity-Free Properties: { "post": ["COMPOUND", "SET"], ... }
    asp_map: Activity-Sensitive Properties: { ("post","ENTER"): ["TIMEOUTABLE"], ... }
    Returns list of detected events with metadata.
    """
    svo = extract_svo_corenlp(step_text)
    subj = svo["subject"]
    verb = svo["verb"]
    activity = map_verb_to_activity(verb) or "UNKNOWN"
    obj = svo["object"]

    # Map to performer type: if subject matches System -> SYSTEM else USER
    perf = "USER"
    if subj:
        if subj.lower() in [a.lower() for a in actors]:
            # if the subject is "System" or matches system-like -> SYSTEM
            if subj.lower() == "system":
                perf = "SYSTEM"
            else:
                perf = "USER"
        else:
            # heuristics: if subject contains 'system' -> SYSTEM
            if "system" in subj.lower():
                perf = "SYSTEM"
            else:
                perf = "USER"
    else:
        # default assume USER (paper assumes steps alternate)
        perf = "USER"

    # Normalize object name and find declared info object closest match
    matched_obj_name = None
    for io in info_objects:
        if io.lower() in obj.lower() or obj.lower() in io.lower():
            matched_obj_name = io
            break
    if not matched_obj_name and info_objects:
        # fallback: take first info_object if object not detected
        matched_obj_name = info_objects[0]

    # gather properties: AFP from afp_map
    props = afp_map.get(matched_obj_name.lower(), []) if matched_obj_name else []
    # Compound rule uses asp_map
    simple_events = match_observed(perf, activity, props)
    compound_events = match_compound(perf, activity, matched_obj_name or "", asp_map)

    all_events = list(dict.fromkeys(simple_events + compound_events))  # deduplicate preserving order

    results = []
    for ev in all_events:
        template = EVENT_TEMPLATES.get(ev, "{subject} {verb_past} {object} -> " + ev)
        generated = template.format(subject=svo["subject"] or perf.title(),
                                    verb_past=svo["verb_past"] or svo["verb"],
                                    object=svo["object"] or (matched_obj_name or ""),
                                    object_plural=svo["object_plural"] or (matched_obj_name + "s" if matched_obj_name else ""))
        results.append({"step": step_text, "performer": perf, "activity": activity, "object": matched_obj_name or svo["object"], "event_type": ev, "event_text": generated})
    return results

# -----------------------
# 7) Example / Demo
# -----------------------
if __name__ == "__main__":
    # Example declarations (你应当替换成你自己的声明/数据集)
    actors = ["Author", "System", "Student", "Customer"]
    info_objects = ["Post", "Article", "Credit card", "Book", "Subject"]

    # AFP: activity-free properties for each info object (lowercased keys)
    afp_map = {
        "post": ["COMPOUND"],
        "article": ["SET"],
        "credit card": ["REMOTE"],
        "book": ["SET"],
        "subject": ["SET", "AT_LEAST", "NO_MORE_THAN"]
    }

    # ASP: activity-sensitive properties: keys are (object_lower, activity) -> list of properties
    asp_map = {
        ("post", "ENTER"): ["TIMEOUTABLE"],  # e.g. post is timeoutable when ENTER
        ("invoice", "IMPORT"): ["LONG_LASTING"]
    }

    # Example use-case steps (替换为你的数据集中的步骤)
    steps = [
        "Author chooses a post.",
        "Author chooses the edit option.",
        "System prompts for entering the changes.",
        "Author enters the changes to the post.",
        "Author confirms the changes.",
        "System stores the changes.",
        # additional examples
        "Customer enters credit card details.",
        "Student selects books.",
        "Author deletes an article."
    ]

    # Process steps
    all_results = []
    for s in steps:
        evs = identify_events_for_step(s, actors, info_objects, afp_map, asp_map)
        if evs:
            for e in evs:
                print(f"[STEP] {e['step']}")
                print(f"  Performer={e['performer']}, Activity={e['activity']}, Object={e['object']}")
                print(f"  -> EventType={e['event_type']}; Text: {e['event_text']}")
                print()
                all_results.append(e)
        else:
            print(f"[STEP] {steps.index(s)}: {s}")
            print("  -> No event inferred by rules.\n")
