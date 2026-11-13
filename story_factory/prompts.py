"""System prompts and task prompts for every writers' room role.

Each task's output shape is enforced separately by `schemas`, so prompts focus
on judgment and craft rather than formatting.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Sequence

CHANNEL = """\
Hollow Lines is an animated horror storytelling channel for a global audience of teens and
adults (not made for kids). Every video is told in still pictures that look like frames of a
dark, slow-burn horror film (real-looking people and places, low light, deep shadows, a
muted cold colour grade, film grain and a single blood-red accent) over voices and music. The host, The Storyteller, a tall man in a long coat and wide-brimmed hat
carrying a red-flamed lantern, opens and closes long episodes, speaking directly to the
viewer. The story itself is usually told in the first person by the one it happened to (a
first-person teller gets their own voice); characters speak their quoted lines in their own
voices.

Rules that are never broken:
- Fiction or folklore only; never claim a story is true or happened to real people.
- No real living or recently deceased people, real crimes, or real companies as villains.
- No gore, torture or lingering injury; violence is implied, brief and off-screen.
- No sexual content. No harm to children; children may appear only safely.
- No strong profanity: the channel stays advertiser-friendly.
- No suicide or self-harm methods.
- No copyrighted characters, creatures or plots (famous creepypastas, film, TV or game monsters).
- Treat every culture's folklore with accuracy and respect; no stereotypes.
"""

AUDIENCE_NEEDS = """\
What a great Hollow Lines story gives its viewer (the psychology we write for):
1. Curiosity: a concrete, unanswered question within the first ten seconds, and new questions
   every minute or so. Answer an old question only after a new one is open.
2. A person, not a job: we care before anything goes wrong because the hero wants something,
   carries a wound, and has one small human detail we recognize.
3. Agency: the hero makes choices, including one understandable mistake the viewer wants to
   shout at. Things happen because of the hero, not only to them.
4. Dread over surprise: the viewer knows or suspects more than the hero, and waits for it.
5. Anticipation: the moment before the reveal is the scariest. Linger there; keep the threat
   mostly unseen and let the listener's imagination do the work.
6. Rhythm: fear climbs, but breathes. A moment of warmth or quiet humor makes the next
   scare land harder.
7. Fair play: the threat obeys its own rules, every planted clue pays off, and the twist
   recolors what came before so the story rewards a second watch.
8. Peak and end: viewers remember the worst moment and the last line most; spend the best
   writing there.
9. Meaning: beneath the fear lies something true about people (grief, guilt, loneliness,
   love), and the ending answers it without stating it.
10. Aftertaste: the fear follows them home, attached to something ordinary they will notice
   tonight.
11. Something to argue about: one deliberate, fair ambiguity for the comments.
12. "That could be me": an ordinary person in a place the viewer knows.
"""

CRAFT = """\
Writing for the ear:
- This is narration heard once, at speaking pace, over simple drawings, by a global audience
  for whom English is often a second language. Write sentences a listener can follow without
  rereading: mostly plain words, an average sentence under about 16 words, no tangled clauses.
  Vary rhythm: longer sentences to settle in, very short ones as dread tightens.
- Make it concrete: specific sounds, smells, objects, times and numbers. One precise detail
  beats three adjectives. Show fear through the body and actions; never name the feeling
  ("she felt terrified").
- Write for sound: a recurring sound can be a clue, and silence is a sound.
- Use a refrain: a short line or image that returns and means something worse the second time.
- Keep dialogue sparse and always clear about who is speaking.
- Every moment must make one strong still picture: few people, clear silhouettes, darkness, one red accent.
- Never explain the twist or the theme. Trust the listener.
- Avoid worn horror phrasing: "a chill ran down", "blood ran cold", "little did they know",
  "suddenly", "out of nowhere", "it was all a dream", "heart pounding in my chest",
  "something was off", "eerie silence".
- Avoid machine-sounding phrasing: "a sense of dread", "the air grew thick", "a testament to",
  "tapestry", "palpable", "sent shivers down", "couldn't shake the feeling", "deafening
  silence", "in that moment", "time seemed to stop", "every fiber of", "hung in the air",
  "for what felt like an eternity". No lists of three adjectives. Few dashes.
"""

STORYTELLER = """\
The Storyteller (the voice of every intro and outro):
- An old, warm, unhurried voice; never shouts, never hypes, never asks viewers to subscribe.
- Speaks to the viewer as "you", like someone sharing a fire. Faintly amused by the dark,
  tender toward the people in the stories. Short sentences.
- Never explains what a story meant.
- Cold open, then ritual: every long episode's intro opens with one or two gripping
  sentences flashed forward from the story (present tense, a question without a spoiler),
  then exactly "The lantern is lit.", then at most one short line. Never a slow mood-setting
  preamble: viewers decide in seconds. Shorts never use the ritual: their first words are
  the hook. The outro lets the story's fear follow the viewer
  home, and ends with exactly one question to the viewer.
"""

STYLE_SAMPLES = """\
House style, by example (original samples: match their qualities, never copy their words).

GOOD (a hook that raises a concrete question):
"Marisol had driven the 11:40 bus for nine years, and in nine years nobody had ever pressed
the stop bell between Calder Road and the reservoir. There is nothing out there. No houses.
No shelter. Just a pole with a timetable nobody updates. Tonight, at six minutes past
midnight, the bell rang. She checked the mirror. Every seat was empty. The bell rang again."

GOOD (a person, and dread the viewer sees before the hero does):
"He hums when he is nervous, always the same three notes of a song his mother sang, and he is
humming now as he locks the back door. Check, then check again. That is his rule. What he
does not see, what you can see if you look at the kitchen window, is the handprint in the
mist on the glass. On the inside."

GOOD (an ending that answers the emotional core and leaves an aftertaste):
"The voicemail was still saved under Don't Delete. She played it one last time. Hi, love.
I'm on my way. And after the beep, very close to the phone, a second voice none of them
had ever noticed: So am I. She keeps her phone in the kitchen now, face down. You might
want to listen to your old voicemails tonight. Or you might not."

NEVER LIKE THIS:
"A sense of dread washed over her as the air grew thick with an eerie silence. Her heart
pounded in her chest. Suddenly, out of nowhere, a shadowy figure appeared, sending shivers
down her spine. She felt terrified."
Why: it names feelings instead of causing them, every phrase is borrowed, and there is
nothing a listener can picture.
Instead: "The fridge stopped humming. In the quiet she heard a second breath, slower than
hers, coming from the hallway she had just walked down."
"""

SYSTEM_WRITER = (
    "You are the head writer of Hollow Lines, a world-class horror storyteller and TV writer "
    "known for characters people love, dread that builds, and endings people still think about "
    "days later. You understand what audiences need psychologically and you write to give it "
    "to them.\n\n"
    + CHANNEL + "\n" + AUDIENCE_NEEDS + "\n" + CRAFT + "\n" + STORYTELLER + "\n" + STYLE_SAMPLES
)

SYSTEM_EDITOR = (
    "You are a demanding story editor for Hollow Lines who has read thousands of horror stories "
    "and greenlights only the exceptional. You are specific, honest and hard to impress: a "
    "competent but familiar story scores 6, not 8. You judge only what is on the page.\n\n"
    + CHANNEL + "\n" + AUDIENCE_NEEDS
)

SYSTEM_RESEARCHER = (
    "You are a folklore researcher preparing accurate notes for a horror storyteller. Prefer "
    "scholarly and cultural sources over entertainment sites, separate the traditional legend "
    "from modern pop-culture versions, and note anything that would be disrespectful to get wrong."
)

# What each editor score means, so different judges score on the same scale.
REVIEW_GUIDE = """\
Scoring guide (a competent, publishable story is 6-7; an 8 needs a reason you can quote):
- hook: 4 opens on weather or setting; 6 something odd but no real question; 8 a concrete
  question within two sentences that you need answered; 10 impossible to leave.
- tension: 4 flat or repetitive; 6 escalates predictably or stops raising new questions; 8 each
  escalation worse than the last, new open loops keep pulling, dread through what the viewer
  knows; 10 relentless, yet it breathes.
- character: 4 a job title; 6 sympathetic but passive or generic; 8 a specific want, wound and
  human detail, and choices that drive the story; 10 a person you will not forget.
- emotional_payoff: 4 arbitrary or explained ending; 6 makes sense but does not land; 8 earned,
  recolors the story, answers its emotional core and leaves an aftertaste; 10 haunts for days.
- originality: 4 familiar internet-horror beats; 6 familiar premise with a fresh detail; 8 a
  central image or turn you have not seen; 10 a new idea.
- prose_for_the_ear: 4 cliches, named feelings, tangled sentences; 6 clean but generic; 8
  specific sensory detail, varied rhythm, easy to follow in one hearing; 10 every line lands aloud.
- drawability: 4 needs crowds or complex action to make sense; 6 mostly works as still pictures;
  8 strong simple compositions and one iconic image; 10 an unforgettable red-accent image.
- follows_recipe: 4 ignores the ingredients; 6 uses them on the surface; 8 all used naturally;
  10 the recipe feels inevitable.
"""

PITCH_GUIDE = """\
Scoring guide (a competent pitch is 6; 8 needs a reason you can name): 8+ on click_appeal and hook
means a specific, strange image or question a stranger would stop for; 8+ on emotional_core
means a real human need or wound, not just fear; 8+ on originality means you have not seen this
central image or turn before.
"""


def system_audience(persona: str) -> str:
    return (
        f"You are {persona}. You are watching a horror story video on YouTube. React honestly as "
        "this person would, not as a critic. You have many other videos you could watch instead."
    )


def _recipe(recipe: Any, formats: dict[str, str], endings: dict[str, str]) -> str:
    return (f"- setting: {recipe.setting}\n- core fear: {recipe.core_fear}\n"
            f"- protagonist: {recipe.protagonist}\n- format: {formats[recipe.format]}\n"
            f"- twist: {recipe.twist}\n- ending: {endings[recipe.ending]}\n")


def _pitch(p: dict[str, Any]) -> str:
    return (f"Title: {p['title']}\nLogline: {p['logline']}\nSynopsis: {p['synopsis']}\n"
            f"Opening line: {p.get('hook_line', '')}\nThumbnail: {p.get('thumbnail_concept', '')}")


def _world(entities: Sequence[sqlite3.Row]) -> str:
    if not entities:
        return ""
    lines = "\n".join(f"- {e['name']} ({e['kind']}): {e['description']}" for e in entities)
    return ("\nShared world canon (never contradict it). Loyal viewers love small connections: you "
            "may weave in at most two existing entries as subtle details, and introduce at most two "
            f"new named characters, places or objects.\n{lines}\n")


def _legend(legend: sqlite3.Row | None, dossier: dict | None) -> str:
    if legend is None:
        return ""
    notes = json.dumps(dossier, ensure_ascii=False) if dossier else (legend["summary"] or "")
    return (f"\nLegend to retell: {legend['title']} ({legend['region']}).\nResearch notes: {notes}\n"
            "Keep the legend's traditional core accurate and respectful; the framing story about the "
            "protagonist is original. The legend's being keeps exactly its traditional nature, "
            "appearance, powers and role: never invent new ones (in a story bible, the threat is the "
            "being as tradition describes it). The episode's recipe (setting, twist, ending) shapes "
            "only the modern frame story; reinterpret any ingredient that would bend the legend. Use "
            "your own words, never the notes' wording.\n")


def _bible(bible: dict | None) -> str:
    if not bible:
        return ""
    return ("Story bible (the decisions every draft must honor):\n"
            + json.dumps(bible, ensure_ascii=False) + "\n")


def _script(script: dict[str, Any]) -> str:
    return json.dumps({k: script.get(k, "") for k in ("intro", "body", "outro")}, ensure_ascii=False)


# --- Research ----------------------------------------------------------------

def dossier_research(legend: sqlite3.Row) -> str:
    return (f"Research the traditional legend \"{legend['title']}\" from {legend['region']}. "
            f"Starting notes: {legend['summary']}\n\nFind: the core traditional story and what the "
            "entity is and does; how versions differ by region or era; cultural meaning and context; "
            "details that would be inaccurate or disrespectful to get wrong; and which well-known "
            "versions are modern copyrighted works rather than tradition. Cite your sources.")


def dossier_structure(legend: sqlite3.Row, research: str) -> str:
    return (f"Turn these research notes on \"{legend['title']}\" into a storyteller's dossier. Keep "
            f"only what the notes support.\n\n{research}")


# --- Pitching ----------------------------------------------------------------

def pitches(*, series: Any, recipe: Any, formats: dict[str, str], endings: dict[str, str], count: int,
            legend: sqlite3.Row | None, dossier: dict | None, sparks: Sequence[sqlite3.Row],
            world: Sequence[sqlite3.Row], avoid: Sequence[sqlite3.Row], trends: Sequence[sqlite3.Row],
            learnings: Sequence[str]) -> str:
    parts = [f"Series: {series.title}. {series.description}",
             ("Suggested ingredients for the modern frame story (drop or adapt any that would change "
              "the legend's nature, place or role; the legend comes first):\n" if legend else
              "Recipe for this episode (use every ingredient):\n") + _recipe(recipe, formats, endings),
             _legend(legend, dossier), _world(world)]
    if sparks:
        parts.append("Folklore motifs you may borrow one abstract element from (a creature's nature, "
                     "a superstition), never a plot or a name:\n"
                     + "\n".join(f"- {s['title']} ({s['region']}): {(s['summary'] or '')[:300]}" for s in sparks))
    if learnings:
        parts.append("What our audience has responded to so far (from real analytics):\n"
                     + "\n".join(f"- {i}" for i in learnings))
    if trends:
        parts.append("Popular horror video titles right now, to understand appetite only; never copy "
                     "their titles or plots:\n" + "\n".join(f"- {t['title']}" for t in trends))
    if avoid:
        parts.append("Episodes we already have; every pitch must be clearly different from all of "
                     "these:\n" + "\n".join(f"- {s['title']}: {s['logline'] or ''}" for s in avoid))
    parts.append(
        f"Pitch {count} genuinely different episodes built from the recipe: different central images, "
        "different hooks, different emotional cores. Give each a hero with a want and a wound, a "
        "threat with its own logic, and an ending that answers something human beneath the fear. "
        "Before writing each, ask whether a viewer "
        "scrolling past would stop for its title and thumbnail. Titles: at most 60 characters, "
        "intriguing, honest about the story. Synopsis: 80-120 words including the ending. "
        "Thumbnail concept: one simple drawable image with no text.")
    return "\n\n".join(p for p in parts if p)


def score_pitch(series: Any, pitch: dict[str, Any], legend: sqlite3.Row | None = None,
                dossier: dict | None = None) -> str:
    fidelity = ""
    if legend is not None:
        notes = json.dumps(dossier, ensure_ascii=False) if dossier else (legend["summary"] or "")
        fidelity = (f"This retells the legend {legend['title']} ({legend['region']}). Research notes: {notes}\n"
                    "series_fit: score 1-3 if the pitch changes the being's nature, appearance, powers, "
                    "place or role from these notes.\n\n")
    return (f"Series: {series.title}. {series.description}\n\nPitch:\n{_pitch(pitch)}\n\n{fidelity}{PITCH_GUIDE}\n"
            "Score 1-10 on: click_appeal (would a viewer stop scrolling for this title and thumbnail), "
            "hook, originality (have you seen this before?), emotional_core, series_fit and "
            "talkability (will viewers debate it in the comments). Then give a one-line verdict.")


def compare_pitches(series: Any, a: dict[str, Any], b: dict[str, Any]) -> str:
    return (f"Series: {series.title}. We can only make one of these episodes.\n\nPITCH A:\n{_pitch(a)}"
            f"\n\nPITCH B:\n{_pitch(b)}\n\nWhich would more viewers click, watch to the very end and "
            "talk about afterwards? Reason briefly, then choose.")


def judge_duplicate(premise: dict[str, Any], existing: sqlite3.Row) -> str:
    return ("Decide whether two horror premises are essentially the same story: the same central "
            "situation, threat and twist, even if names, places or wording differ. Sharing only a "
            "genre, a setting type or a common trope does NOT make them the same.\n\n"
            f"STORY A: {existing['title']}. {existing['logline'] or ''} {existing['synopsis'] or ''}\n\n"
            f"STORY B: {premise['title']}. {premise['logline']} {premise['synopsis']}")


# --- Story bible -------------------------------------------------------------

def story_bible(*, series: Any, kind: str, recipe: Any, formats: dict[str, str], endings: dict[str, str],
                pitch: dict[str, Any], legend: sqlite3.Row | None, dossier: dict | None,
                world: Sequence[sqlite3.Row], recent_heroes: Sequence[str] = ()) -> str:
    size = ("" if kind == "long" else
            " This is a 60-second Short: keep every field to one short line, give one or two clues, "
            "two or three insider_details, and leave breath_moment, refrain and refrain_turn empty "
            "if there is no room.")
    return "\n\n".join(p for p in [
        f"Series: {series.title}. {series.description}",
        "Recipe:\n" + _recipe(recipe, formats, endings),
        "Approved pitch:\n" + _pitch(pitch), _legend(legend, dossier), _world(world),
        ("Heroes of our recent episodes. Make this hero clearly different from all of them in age, "
         "gender, background, culture and life situation, and never reuse their names:\n"
         + "\n".join(f"- {h}" for h in recent_heroes)) if recent_heroes else "",
        "Before any plotting, make the storyteller's decisions for this episode, as a story bible.\n"
        "hero: a specific person, not a job. name (invented, and not the name of any famous real "
        "person); who (age, life, in one line); want (what they are "
        "trying to get tonight); wound (the past hurt that shapes them); flaw (what will lead them "
        "into a mistake); false_belief (what they wrongly believe, which the ending tests); "
        "human_detail (one small, specific habit or object that makes us love them); voice (how "
        "they talk, in a few words).\n"
        "threat: what_it_is; what_it_wants; rules (2-4 laws it always obeys); limits (what it cannot "
        "do); tell (the sign that it is near, ideally a sound or small detail).\n"
        "theme: what the story is really about, in a few words. theme_question: the human question "
        "the ending answers without stating it.\n"
        "hero_choice: the key decision the hero makes, ideally an understandable mistake viewers "
        "will shout at. dramatic_irony: what the viewer will know or suspect before the hero. "
        "breath_moment: a moment of warmth or quiet humor before the worst.\n"
        "refrain: a short line or image (at most 12 words) that returns; refrain_turn: how its "
        "meaning darkens when it returns. sound_motif: the recurring sound. red_accent: the one "
        "thing drawn in red and why it matters to this story.\n"
        "open_detail: one deliberate, fair ambiguity viewers will argue about. aftertaste: the "
        "ordinary thing in the viewer's own life the fear will attach to tonight.\n"
        "clues: 2-4 early details, each with the later payoff it sets up.\n"
        "insider_details: 6-8 specific details only someone who has really done the hero's job or "
        "lived in this place would know: routines, jargon, objects, sounds, smells, small rules "
        "(for a legend, the region's real customs, places and words). They make the story feel "
        "lived-in; never invent ones that sound fake." + size] if p)


def review_bible(kind: str, pitch: dict[str, Any], bible: dict[str, Any], dossier: dict | None = None) -> str:
    fidelity = ("This retells a traditional legend. Also require that the threat is the legend's being "
                "exactly as these research notes describe it, with no invented appearance, powers, place "
                "or role (interpreting its motives or inner life is allowed), and that the frame story "
                "fits where the legend can happen:\n"
                f"{json.dumps(dossier, ensure_ascii=False)}\n\n" if dossier else "")
    return (f"Pitch:\n{_pitch(pitch)}\n\n{_bible(bible)}\n{fidelity}Review this story bible for a "
            f"{'long episode' if kind == 'long' else '60-second Short'}. Approve only if: the hero is "
            "a specific person we would care about (not a type), with a want, wound and human detail "
            "that matter to the plot; the threat's rules are consistent and its limits create "
            "suspense; the theme and the ending's answer feel true; the hero's choice is active and "
            "understandable; the clues are fair and their payoffs surprising; the insider details "
            "ring true; nothing is generic or cliched; and no character shares the name of a famous "
            "real person. " + PLAN_SCORING)


PLAN_SCORING = ("Score the plan from 1 to 10: 5 generic or broken, 7 solid and ready to write, 8 "
                "exceptional with a reason you can name, 10 unforgettable. List at most five problems, "
                "the ones that would most improve the story, most important first, each specific "
                "enough to act on. approved is true when you would write this plan as it is.")


def revise_bible(bible: dict[str, Any], problems: list[str]) -> str:
    return (f"{_bible(bible)}\nEditor's problems:\n" + "\n".join(f"- {p}" for p in problems)
            + "\n\nRevise the story bible to solve every problem while keeping what works.")


def legend_fit(legend: sqlite3.Row, dossier: dict | None, kind: str) -> str:
    notes = json.dumps(dossier, ensure_ascii=False) if dossier else (legend["summary"] or "")
    length = "an 8-minute episode" if kind == "long" else "a 60-second Short"
    return (f"Legend: {legend['title']} ({legend['region']}).\nResearch notes: {notes}\n\nCould this "
            f"legend carry {length} of genuine horror on our channel while its being keeps exactly its "
            "traditional nature, appearance and role? Answer fit false if the being is harmless or "
            "comic, too thinly documented to frighten without inventing traits it does not have, or if "
            "retelling it as horror would disrespect a sacred figure or living religious practice.")


def bible_check(bible: dict[str, Any], items: Sequence[tuple[str, str]], script: dict[str, Any],
                dossier: dict | None = None) -> str:
    listed = "\n".join(f"- [{item_id}] {requirement}" for item_id, requirement in items)
    legend = ("\n\nThis retells a traditional legend. Also list, among the rule breaks, every place where "
              "the script gives the legend's being an appearance, power, place or role these research "
              "notes do not support. Interpreting its motives, feelings or inner life is what a retelling "
              "does and is allowed; do not list that. Research notes:\n"
              f"{json.dumps(dossier, ensure_ascii=False)}" if dossier else "")
    return (f"STORY BIBLE:\n{json.dumps(bible, ensure_ascii=False)}\n\nSCRIPT:\n{_script(script)}\n\n"
            f"TASK: Check whether the script delivers each item below.\n{listed}\n\nFor each item "
            "give its id, whether it is clearly present on the page, and a short exact quote (5-15 "
            "words, copied character for character from the script) that shows it; use an empty "
            "quote when it is missing. Then list every place where the threat breaks one of its own "
            f"rules or limits (empty if none).{legend}")


def confirm_breaks(bible: dict[str, Any], script: dict[str, Any], claimed: Sequence[str],
                   dossier: dict | None = None) -> str:
    listed = "\n".join(f"[{n}] {c}" for n, c in enumerate(claimed, 1))
    notes = (f"\nLegend research notes:\n{json.dumps(dossier, ensure_ascii=False)}\n" if dossier else "")
    return (f"STORY BIBLE:\n{json.dumps(bible, ensure_ascii=False)}\n{notes}\nSCRIPT:\n{_script(script)}\n\n"
            f"Another editor claims the script breaks the threat's own rules (or, for a legend, gives "
            f"its being traits the research does not support):\n{listed}\n\nFor each numbered claim, "
            "decide whether it is real: the script clearly shows the threat doing what its rules or the "
            "research forbid, or gives a legend's being an appearance, power, place or role the research "
            "does not support. Mark it not real if it is a stretch, about something the script never "
            "shows, or about the being's motives, feelings or inner life (a retelling may interpret "
            "those). Give its number, your verdict and a one-line reason.")


# --- Outline and drafts ------------------------------------------------------

def beat_sheet(*, series: Any, kind: str, recipe: Any, formats: dict[str, str], endings: dict[str, str],
               pitch: dict[str, Any], legend: sqlite3.Row | None, dossier: dict | None,
               world: Sequence[sqlite3.Row], target_seconds: int, bible: dict | None = None) -> str:
    if kind == "long":
        shape = ("a Storyteller intro; a hook in the first sentence; ordinary life where we come to "
                 "care about the hero (show the want, the wound and the human detail); the first "
                 "wrong detail; three escalations each worse than the last, with a reversal at the "
                 "midpoint that changes what the hero is fighting; the breath moment; the hero's "
                 "choice; a false relief; the climax, lingering on the moment before the reveal; the "
                 "ending that answers the theme question; and a Storyteller outro that lets the fear "
                 "follow the viewer home and ends in one question")
        ladder = ("escalations: the three escalations, each naming its beat and how it is worse than "
                  "the one before: closer (the threat is physically nearer), more_personal (it knows or "
                  "takes something of the hero's) and safety_lost (the place, person, rule or belief the "
                  "hero relied on that stops protecting them). Fill all three for every escalation.")
        retention = ("rehooks: at least three new questions the story raises to stop viewers leaving, "
                     "one by about 30 seconds, one by about 2 minutes and one near the midpoint, each "
                     "with its time in seconds. midpoint_reversal: the turn halfway through. "
                     "peak_moment: the single most frightening moment.")
    else:
        shape = ("a hook in the first sentence; one fast escalation; the hero's choice if there is "
                 "room; a sting in the final line that, where it fits, loops back into the first line "
                 "so a replay feels seamless; and a one-line Storyteller question")
        ladder = ("escalations: the one escalation, with how the threat comes closer, gets more "
                  "personal and which safety the hero loses.")
        retention = ("rehooks: one question that lands by about 15 seconds. midpoint_reversal: "
                     "empty unless there is one. peak_moment: the sting.")
    return "\n\n".join(p for p in [
        f"Series: {series.title}. {series.description}",
        "Recipe:\n" + _recipe(recipe, formats, endings),
        "Approved pitch:\n" + _pitch(pitch), _bible(bible), _legend(legend, dossier), _world(world),
        f"Outline this episode beat by beat for about {target_seconds} seconds of narration: {shape}. "
        "For each beat give its purpose, what happens, its target length in seconds, its tension "
        "from 1 to 10, its turn (what is different at the end of the beat; a beat where nothing "
        "changes is cut), its open_loop (the question it leaves the viewer holding; empty only for "
        "the final beat), the clues it plants and the earlier clues it pays off. Tension climbs "
        "overall but breathes: it dips at the breath moment and the false relief. Place every "
        "story-bible clue, the refrain and its return, and the dramatic irony. State the central "
        f"question and the emotional core. {retention} {ladder}"] if p)


def review_outline(kind: str, pitch: dict[str, Any], outline: dict[str, Any], bible: dict | None = None) -> str:
    return (f"Pitch:\n{_pitch(pitch)}\n\n{_bible(bible)}\nOutline:\n{json.dumps(outline, ensure_ascii=False)}\n\n"
            "Approve only if: the hook raises a concrete question at once; new questions open before "
            "old ones close, so there is never a natural moment to leave; tension climbs but "
            "breathes; each escalation is worse than the last because the threat comes closer, gets "
            "more personal and takes away a safety; the hero drives events through choices; the "
            "viewer is made to dread what the hero does not know; every payoff was planted earlier "
            "and every plant pays off; the threat obeys its rules; the ending is earned, surprising "
            "(a first-time viewer would not guess it) and answers the theme; the emotional core is "
            "felt, not stated; and no beat is filler. " + PLAN_SCORING)


def revise_outline(outline: dict[str, Any], problems: list[str]) -> str:
    return (f"Outline:\n{json.dumps(outline, ensure_ascii=False)}\n\nEditor's problems:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nRevise the outline to solve every problem while keeping what works.")


def write_draft(*, series: Any, kind: str, recipe: Any, formats: dict[str, str], endings: dict[str, str],
                pitch: dict[str, Any], outline: dict[str, Any], min_words: int, max_words: int,
                legend: sqlite3.Row | None, dossier: dict | None, world: Sequence[sqlite3.Row],
                bible: dict | None = None, angle: str = "", avoid: Sequence[str] = ()) -> str:
    structure = ("intro: The Storyteller, 30-60 words: a cold open of one or two sentences flashed "
                 "forward from the story, then exactly \"The lantern is lit.\", then at most one short "
                 "line. body: the story. outro: The Storyteller, 25-60 words, ending with exactly one "
                 "question to the viewer." if kind == "long" else
                 "intro: empty string. body: the story; the first sentence is the hook, and where it "
                 "fits, the last line flows back into the first so a replay loops seamlessly. outro: "
                 "one short Storyteller line ending with a question to the viewer.")
    return "\n\n".join(p for p in [
        f"Series: {series.title}. {series.description}",
        "Recipe:\n" + _recipe(recipe, formats, endings),
        "Approved pitch:\n" + _pitch(pitch),
        _bible(bible),
        "Approved outline (follow its beats, plants and payoffs):\n" + json.dumps(outline, ensure_ascii=False),
        _legend(legend, dossier), _world(world), _avoid(avoid),
        f"Write the full narration. {structure} {angle} intro + body + outro together: {min_words}-{max_words} "
        "words. comment_question repeats the outro's final question exactly. hook_line is the body's "
        "first sentence. description_credit: for a legend, one sentence crediting the tradition and "
        "region; otherwise \"This is an original work of fiction.\" List new shared-world entities "
        "you introduced and existing ones you referenced."] if p)


def write_section(*, series: Any, recipe: Any, formats: dict[str, str], endings: dict[str, str],
                  pitch: dict[str, Any], outline: dict[str, Any], bible: dict | None, beats: list[dict],
                  part: int, parts: int, so_far: str, words: int, legend: sqlite3.Row | None,
                  dossier: dict | None, world: Sequence[sqlite3.Row], avoid: Sequence[str] = ()) -> str:
    if part == 1:
        place = "This is the opening part: its first sentence is the hook."
    elif part == parts:
        place = "This is the final part: bring the story to its ending; the Storyteller outro comes later."
    else:
        place = "This is a middle part."
    previous = (f"The story so far (continue seamlessly from its last line; never repeat or recap it):\n{so_far}"
                if so_far else "")
    ending = ("End this part on an open question or a turn that makes leaving impossible."
              if part < parts else "End on the story's final line.")
    return "\n\n".join(p for p in [
        f"Series: {series.title}. {series.description}",
        "Recipe:\n" + _recipe(recipe, formats, endings),
        "Approved pitch:\n" + _pitch(pitch), _bible(bible),
        "Full outline, for orientation:\n" + json.dumps(outline, ensure_ascii=False),
        _legend(legend, dossier), _world(world), _avoid(avoid), previous,
        f"Write part {part} of {parts} of the story body (no Storyteller intro or outro), covering "
        "only these beats, in order, with their plants and payoffs:\n"
        + json.dumps(beats, ensure_ascii=False)
        + f"\n\n{place} About {words} words. Write the full prose of these beats, for the ear, in the "
        f"house style. {ending}"] if p)


def _avoid(notes: Sequence[str]) -> str:
    if not notes:
        return ""
    return ("An earlier draft of this story was turned down by the editors. Write a fresh draft that "
            "avoids its problems:\n" + "\n".join(f"- {n}" for n in notes))


# Different takes for the Shorts contest, so the drafts compete on approach, not only wording.
SHORT_ANGLES = (
    "Take: build almost everything on dread the viewer sees before the hero does.",
    "Take: lead with the strangest concrete image and let the sting recolor it.",
    "Take: make us love the hero in two lines, so the sting hurts as well as scares.",
)


def write_frame(*, series: Any, kind: str, pitch: dict[str, Any], bible: dict | None, body: str,
                legend: sqlite3.Row | None, world: Sequence[sqlite3.Row]) -> str:
    return "\n\n".join(p for p in [
        f"Series: {series.title}. {series.description}", "Approved pitch:\n" + _pitch(pitch),
        _bible(bible), _world(world), f"The finished story body:\n{body}",
        "Write The Storyteller's frame for this story. intro: 30-60 words in this order: a cold "
        "open of one or two sentences flashed forward from a gripping moment in the story, in the "
        "present tense, raising a question without spoiling the ending; then exactly \"The lantern "
        "is lit.\"; then at most one short line. outro: 25-60 "
        "words that let the fear follow the viewer home (use the story bible's aftertaste) and end "
        "with exactly one question to the viewer, ideally about the open detail or the hero's "
        "choice. comment_question repeats that final question exactly. thumbnail_idea: one simple "
        "drawable image built around the red accent. description_credit: "
        + (f"one sentence crediting the tradition of {legend['title']} ({legend['region']})."
           if legend else "\"This is an original work of fiction.\"")
        + " List new shared-world entities the body introduced and existing ones it referenced."] if p)


def compare_drafts(kind: str, pitch: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> str:
    return (f"Two drafts of the same {'episode' if kind == 'long' else 'Short'} ({pitch['title']}).\n\n"
            f"DRAFT A:\n{_script(a)}\n\nDRAFT B:\n{_script(b)}\n\nRead each as a listener hearing it "
            "once. Which keeps you more gripped from first line to last, lands its ending harder and "
            "sounds better aloud? Ignore length unless it hurts the story. Reason briefly, then choose.")


def review(kind: str, recipe: Any, formats: dict[str, str], endings: dict[str, str],
           script: dict[str, Any], dossier: dict | None, bible: dict | None = None) -> str:
    accuracy = (f"\nLegend dossier for accuracy:\n{json.dumps(dossier, ensure_ascii=False)}\n"
                if dossier else "")
    return (f"Recipe:\n{_recipe(recipe, formats, endings)}{accuracy}\n{_bible(bible)}\nScript:\n{_script(script)}\n\n"
            f"{REVIEW_GUIDE}\nScore 1-10 on each criterion using the guide: hook, tension, character, "
            "emotional_payoff, originality, prose_for_the_ear, drawability and follows_recipe (for a "
            "retold legend, faithfulness to the legend counts more than the suggested ingredients). Flag "
            "every rule violation. Then give up to six notes on the changes that would most raise the "
            "lowest scores: for each, quote the exact passage (copied from the script), say what is "
            "wrong with it, and give the fix (ideally the rewritten line).")


def fit_length(script: dict[str, Any], current: int, min_words: int, max_words: int) -> str:
    target = (min_words + max_words) // 2
    change = (f"Cut about {current - target} words" if current > target
              else f"Add about {target - current} words")
    return (f"Script:\n{json.dumps(script, ensure_ascii=False)}\n\nintro + body + outro is {current} words. "
            f"It must be {min_words}-{max_words} words; aim for {target}. {change} by tightening or "
            "enriching sentences. Keep every story event, clue, the ending and the outro's final "
            "question. Return every field.")


def line_edit(script: dict[str, Any], problems: list[str]) -> str:
    return (f"Script:\n{json.dumps(script, ensure_ascii=False)}\n\nA line editor found these problems:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nRewrite only the sentences involved, in the house style: concrete, plain, for the "
            "ear. Keep every event, clue, the ending, the outro's final question and about the same "
            "length. Return every field.")


def revise(kind: str, script: dict[str, Any], notes: list[str], min_words: int, max_words: int,
           bible: dict | None = None) -> str:
    current = len(" ".join(str(script.get(k, "")) for k in ("intro", "body", "outro")).split())
    return (f"{_bible(bible)}Current script ({current} words; it must stay {min_words}-{max_words}):\n"
            f"{json.dumps(script, ensure_ascii=False)}\n\nEditors' notes:\n"
            + "\n".join(f"- {n}" for n in notes)
            + f"\n\nRevise the script to address every note while keeping what already works. "
            f"intro + body + outro: {min_words}-{max_words} words. Return every field.")


# --- Audience test and polish -----------------------------------------------

def audience(kind: str, chunks: list[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return (f"The narration below is split into numbered parts of about 30 seconds each. Read them in "
            f"order as if hearing the video.\n\n{numbered}\n\nFor every part give your interest from 1 "
            "to 10, whether you would have left the video at that point, and a short honest note on "
            "why. Once you would have left, keep rating as if you had stayed. Then sum up your "
            "experience in one line.")


def rewrite_chunks(script: dict[str, Any], chunks: list[str], weak: dict[int, list[str]]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    issues = "\n".join(f"[{n}] " + " | ".join(notes) for n, notes in sorted(weak.items()))
    return (f"Full script for context:\n{_script(script)}\n\nThe story body split into numbered parts:\n"
            f"{numbered}\n\nTest viewers lost interest in these parts:\n{issues}\n\nRewrite only those "
            "parts so they grip, keeping their story events, the clues they carry and how they connect "
            "to the parts around them. Keep each part about the same length. Return only the rewritten "
            "parts with their numbers.")


def polish(kind: str, script: dict[str, Any], min_words: int, max_words: int, max_pauses: int = 10) -> str:
    return (f"Script:\n{_script(script)}\n\nDo a final read-aloud line edit for a single narrator voice: "
            "smooth any sentence that is hard to say or easy to mishear, tune rhythm with punctuation, "
            "and remove anything that only works on the page. Do not change events, clues, the "
            f"refrain, the ending or the final question. Keep {min_words}-{max_words} words. This is a "
            "light touch: change only the sentences that need it and copy every other sentence exactly. "
            "Return intro, body and outro in full.")


def annotate(script: dict[str, Any], max_pauses: int) -> str:
    return (f"Final script:\n{_script(script)}\n\nPrepare it for the narrator's voice engine. "
            "pronunciations: every name or foreign word a voice engine might mispronounce, with a "
            "simple English respelling (empty if none). pauses: horror narration lives in its silences; "
            f"mark up to {max_pauses} dramatic pauses, each as the exact phrase (3-8 words, copied "
            "word for word from the script) after which the narrator stops, and the pause length in "
            "milliseconds (400-2500): before reveals, after the worst lines and before the final question.")


def final_check(script: dict[str, Any], dossier: dict | None) -> str:
    if dossier:
        accuracy = (f"Check the legend against this dossier:\n{json.dumps(dossier, ensure_ascii=False)}\n\n")
        task = "list factual errors about the legend"
    else:
        accuracy = "This is an original story, not a retold legend.\n\n"
        task = "leave accuracy_issues empty (there is no legend to check)"
    return (f"{accuracy}Script:\n{_script(script)}\n\nFinal compliance check before publishing. Flag "
            f"any rule violation, {task}, and list anything culturally insensitive. Grade each issue: "
            "must_fix for a factual error about the legend's traditional core, or anything that would "
            "offend, stereotype or misrepresent a culture; minor for missing context, nuance or a "
            "suggestion. A retelling may add an original modern frame and its own characters; that "
            "alone is not an error. Folklore varies by region and era: a detail that matches any "
            "known version, or a small physical detail left out, is minor, and interpreting a legendary "
            "being's motives or inner life is not an error. Empty lists mean it is clear.")


def scene_plan(narration: str, scene_range: tuple[int, int], key_range: tuple[int, int],
               known_looks: Sequence[tuple[str, str]] = (), bible: dict | None = None) -> str:
    canon = ""
    if bible:
        canon += (f"This story's red accent: {bible.get('red_accent', '')}. Use red only for it. "
                  f"Its recurring sound: {bible.get('sound_motif', '')}; put it in the sound effects "
                  "wherever the narration implies it.\n")
    if known_looks:
        canon += ("Recurring characters already have fixed looks; reuse them word for word if they "
                 "appear:\n" + "\n".join(f"- {name}: {look}" for name, look in known_looks) + "\n")
    return (f"NARRATION:\n{narration}\nTASK: Split the narration into {scene_range[0]}-{scene_range[1]} "
            "visual scenes for the animator. Each scene's `text` is copied exactly from the narration; "
            "scenes are in order and together cover the whole narration with nothing skipped. "
            "Describe each picture in words (what is drawn and what moves), keeping it drawable: few "
            "figures, clear silhouettes, darkness and at most one red accent. List sound effects. "
            "Give each scene a delivery cue for the narrator's voice. "
            f"Mark {key_range[0]}-{key_range[1]} scenes as key_moment: the scares, reveals and "
            "emotional peaks that deserve a brand-new image.\n"
            "Also write the cast: every character who appears, with one fixed look an illustrator "
            "can repeat in every image (build, height, clothing, hair or headwear, one distinctive "
            "detail; any glasses or other accessory always worn, never sometimes), as a real person "
            "who could be cast in a film. Scenes refer to characters by these exact "
            f"names.\n{canon}")


def alternative_openings(script: dict[str, Any], opening: str, count: int) -> str:
    return (f"Script:\n{_script(script)}\n\nThe story body currently opens with:\n\"{opening}\"\n\n"
            f"Write {count} alternative versions of exactly that opening passage. Each must raise a "
            "concrete, unanswered question within the first two sentences, plant the same clues, lead "
            "seamlessly into the sentence that follows it, and stay about the same length. Make them "
            "genuinely different approaches: a strange detail, a line of dialogue, a rule, a sound, "
            "a time stamp.")


def compare_openings(a: str, b: str, continuation: str) -> str:
    return (f"Two ways to open the same horror story. Whatever opening is chosen, the story continues: "
            f"\"{continuation}\"\n\nOPENING A:\n{a}\n\nOPENING B:\n{b}\n\nA viewer will decide within "
            "ten seconds whether to keep watching. Which opening makes them unable to leave? Reason "
            "briefly, then choose.")


def compare_titles(a: str, b: str, thumbnail: str, synopsis: str) -> str:
    return (f"Two YouTube titles for the same horror story video. Thumbnail: {thumbnail}\nStory: "
            f"{synopsis}\n\nTITLE A: {a}\nTITLE B: {b}\n\nWhich title would more viewers click on "
            "a crowded home page, while still being honest about the story? Reason briefly, then choose.")


def packaging(series: Any, script: dict[str, Any], pitch: dict[str, Any], credit: str,
              bible: dict | None = None) -> str:
    red = (f"The story's red accent is {bible.get('red_accent', '')}; the thumbnail should show it at "
           "the story's key moment, without spoiling the twist.\n\n" if bible else "")
    return (f"Series: {series.title}.\nPitch:\n{_pitch(pitch)}\n\nFinal script:\n{_script(script)}\n\n{red}"
            "Package this video for YouTube. titles: five options, best first, at most 60 characters, "
            "curiosity without lying. thumbnail_brief: at most 35 words describing one simple, "
            "striking image in our style that reads at phone size: one or two figures at most, one "
            "red element, plenty of darkness; it should add a question the title doesn't, never "
            "repeat the title. thumbnail_text: at most three words, or empty if the image works "
            f"alone. description: two short paragraphs that tease without spoiling, ending with: "
            f"\"{credit}\". tags: ten to fifteen. pinned_comment: a question that invites theories or "
            "personal stories. end_screen_line: one line of at most 20 words The Storyteller says over "
            "the end screen, inviting the viewer to the next story without naming it.")


def teaser_short(script: dict[str, Any], min_words: int, max_words: int) -> str:
    return (f"Full episode script:\n{_script(script)}\n\nCut a 60-second teaser Short from this episode "
            f"that makes viewers watch the full story. body: {min_words - 20}-{max_words - 25} words in the "
            "episode's own voice: open on its strongest hook, climb through the rising dread, and stop "
            "on a cliffhanger just before the reveal. Never give away the twist or the ending. outro: "
            "one short Storyteller line saying the full story is on the channel, ending with a question "
            "to the viewer.")


# --- Plan contest, section rewrites, legend recognition, surprise and ending labs ----

def _plan(plan: dict[str, Any]) -> str:
    return (f"STORY BIBLE:\n{json.dumps(plan['bible'], ensure_ascii=False)}\n"
            f"OUTLINE:\n{json.dumps(plan['outline'], ensure_ascii=False)}")


def compare_plans(kind: str, pitch: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> str:
    return (f"Two writers planned the same {'episode' if kind == 'long' else 'Short'} from this pitch:\n"
            f"{_pitch(pitch)}\n\nPLAN A:\n{_plan(a)}\n\nPLAN B:\n{_plan(b)}\n\nWhich plan will make "
            "the better story for a listener: a hero we care about, escalations that get closer and more "
            "personal, dread through what the viewer knows, an ending a first-time viewer won't guess "
            "but will find inevitable, and a feeling that stays? Reason briefly, then choose.")


def rewrite_sections(script: dict[str, Any], sections: list[str], notes: dict[int, list[str]],
                     bible: dict | None = None) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(sections))
    asks = "\n".join(f"[{n}] " + " | ".join(items) for n, items in sorted(notes.items()))
    return (f"{_bible(bible)}Full script for context:\n{_script(script)}\n\nThe story body in numbered "
            f"sections:\n{numbered}\n\nThe editors' notes on these sections:\n{asks}\n\nRewrite only "
            "those sections to solve their notes. Every other section stays exactly as it is, so each "
            "rewritten section must still connect seamlessly to the sections around it, keep the clues "
            "it carries and the events later sections depend on, and stay about the same length. "
            "Return only the rewritten sections with their numbers.")


def recognition(legend: sqlite3.Row, dossier: dict | None, script: dict[str, Any]) -> str:
    notes = json.dumps(dossier, ensure_ascii=False) if dossier else (legend["summary"] or "")
    return (f"Legend: {legend['title']} ({legend['region']}).\nResearch notes: {notes}\n\n"
            f"Script:\n{_script(script)}\n\nThis episode claims to retell this legend. Would someone who "
            f"grew up with the legend recognize its being in this script as {legend['title']}: its nature, "
            "where it belongs, its signature powers and its role in the story? A modern frame, new human "
            "characters and an interpretation of the being's motives or feelings are fine. Answer "
            "recognizable false only if the being is essentially a different creature wearing the "
            "legend's name. contradicted: the core traits the script contradicts (empty if none).")


def predict_ending(kind: str, opening: str) -> str:
    return (f"Here is the first part of a horror {'story' if kind == 'long' else 'Short'}, as far as a "
            f"viewer has heard it:\n\n{opening}\n\nWhat is really going on, and how does it end? Give "
            "your single best guess in one or two sentences, as a seasoned horror fan would.")


def grade_guesses(script: dict[str, Any], guesses: list[str]) -> str:
    listed = "\n".join(f"[{i + 1}] {g}" for i, g in enumerate(guesses))
    return (f"Full script:\n{_script(script)}\n\nViewers who heard only the first part guessed:\n{listed}"
            "\n\nFor each numbered guess, did it get the story's central twist or ending essentially "
            "right (not just the mood or a minor detail)?")


def misdirection_notes(guesses: list[str]) -> list[str]:
    return ["Test viewers guessed the ending too early. Their guesses: " + " | ".join(guesses),
            "Keep the same ending and every fair clue, but make it harder to see coming: plant one "
            "plausible wrong explanation early, make the true clues quieter and more ordinary, and "
            "remove any line that hints at the answer too directly."]


def alternative_endings(script: dict[str, Any], ending: str, count: int) -> str:
    return (f"Script:\n{_script(script)}\n\nThe story body currently ends with:\n\"{ending}\"\n\n"
            f"Write {count} alternative versions of exactly that closing passage. Viewers remember the "
            "worst moment and the last line most. Each must land the same outcome and answer the "
            "story's emotional core without explaining it, pay off the same clues and the refrain if "
            "it returns there, leave an image that follows the viewer home, flow from the sentence "
            "before it, and stay about the same length. Make them genuinely different approaches.")


def compare_endings(a: str, b: str, before: str) -> str:
    return (f"The same horror story, up to its last passage:\n\"...{before}\"\n\nENDING A:\n{a}\n\n"
            "ENDING B:\n" + b + "\n\nWhich ending lands harder, feels both surprising and inevitable, "
            "and will stay with a viewer tonight? Reason briefly, then choose.")


def learn(top: Sequence[sqlite3.Row], bottom: Sequence[sqlite3.Row]) -> str:
    def rows(items: Sequence[sqlite3.Row]) -> str:
        return "\n".join(
            f"- {r['title']} | {r['series']} | setting: {r['setting']} | twist: {r['twist']} | "
            f"ending: {r['ending']} | views: {r['views']} | avg viewed: {r['avg_view_percentage']}% | "
            f"CTR: {r['ctr']}%" for r in items)
    return (f"Our best-performing videos:\n{rows(top)}\n\nOur worst-performing videos:\n{rows(bottom)}"
            "\n\nWhat separates them? Write up to ten concrete, actionable insights for our writers "
            "about hooks, settings, twists, endings and titles. Only claim what the data supports.")
