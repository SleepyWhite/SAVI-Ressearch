"""Chain format contract (PLAN §2.2) — the model must write the **intermediate state** and the **answer** on two separate lines.

Why it is needed
- The dataset has no independent relation annotation (STATUS §5: the class is back-derived
  from the answer). So if "was the model's state judgment right" can only be back-derived
  from the answer, that amounts to scoring the state with the readout rule — the
  decoupling would be nominal.
- Having the model explicitly write which reading it chose lets two things be measured
  separately:
    (1) did it judge the state correctly            RELATION line vs gold relation
    (2) did it derive correctly from its own state  Final Answer line vs the answer
        implied by its own RELATION line
  Item (2) needs no gold, so it is untouched by the 47.4% label noise.

Freezing discipline
- `CONTRACT_TEMPLATE` and the wording of the two readings are frozen verbatim; sha256 in
  `CONTRACT_SHA256`, pinned by `tests/test_contract_sha.py`. Any change requires a new sha
  and a STATUS changelog entry.
- The wording of the two readings is **taken verbatim from the L2-validated `prose` tier**
  (`probe_oracle_state.condition_text`'s 'prose'). The `entry` semi-formal notation is not
  used: that tier's following rate is only 0.487 vs 0.978 for prose; using it would mix
  the wording problem into the measurement of the intervention form.
- The answer line keeps the upstream `FORMATTING`'s `Final Answer [X]`, so the extraction
  function stays consistent project-wide.
"""
import hashlib
import re

# ---------------------------------------------------------------- the two readings (frozen verbatim)
# REQ: γ3 is an additional necessary condition (p∧r→q) → gold c
# ALT: γ3 is an independent alternative path (p∨r→q) → gold a (ponens) / b (tollens)
# The asymmetric framing of the two arms is required (carried over from L2): if ALT were
# also framed as necessity, then under tollens ¬p cannot be derived from ¬q, gold b becomes
# underivable, and half of the ALT data would be silently mis-measured.
READING_REQ = ('The fact that {p} does not by itself bring it about that {q}; '
               'that happens only if in addition {r}.')
READING_ALT = ('The fact that {p} by itself brings it about that {q}; '
               'and the fact that {r} by itself also brings it about that {q}.')

# ---------------------------------------------------------------- contract template (frozen verbatim)
# ⚠️ Readings use **digits** 1/2, answers use **letters** a/b/c — the two alphabets must
# not overlap. Measured in the smoke phase: when both lines were written `[X]` and both
# used letters, the model would write `Final Answer [B]` (the reading's letter), and the
# upstream `get_final_answer` reads that as the legal option `b`.
# No crash, just wrong data — exactly the class of silent failure CLAUDE.md calls out.
CONTRACT_TEMPLATE = (
    '{premises}\n\n'
    'There are exactly two possible readings of how the third statement relates '
    'to the first:\n'
    '(1) {reading_a}\n'
    '(2) {reading_b}\n\n'
    'First decide which reading is correct, then answer the question below under '
    'that reading.\n\n'
    'What necessarily had to follow assuming that the above premises were true?\n'
    '(a) {opt_a}\n'
    '(b) {opt_b}\n'
    '(c) {opt_c}\n\n'
    # ⚠️ The wording must be a "reminder sentence", not "copy these two lines". Measured in
    # smoke: with "end your reply with exactly these two lines: ... [X]", 27% of outputs
    # copied the placeholder [X] verbatim. The phrasing below aligns with the upstream
    # run_generative.FORMATTING, whose format-failure rate on Qwen2.5-7B is 0
    # (STATUS §6.1 noFA=0).
    # The relation line keeps **no placeholder**: written as 'RELATION: N' + "fill N with
    # 1 or 2", the model copies out 'RELATION: N2' (smoke: 47/48 did). Listing both legal
    # writings in full leaves nothing to copy.
    'Reason briefly. Then state your chosen reading by writing either '
    '"RELATION: 1" or "RELATION: 2". Finally, write your final answer as '
    '"Final Answer [X]." and fill [X] with either a, b, or c.'
)

CONTRACT_SHA256 = hashlib.sha256(
    (CONTRACT_TEMPLATE + READING_REQ + READING_ALT).encode('utf-8')).hexdigest()


def build(premises, p, r, q, opts, req_first):
    """Render one item.

    `req_first` decides whether REQ sits at (A) or (B) — the **position-swap control**.
    Models have positional preferences over A/B; without the swap, "chose correctly"
    cannot be separated from "always chooses A". Same role as L2's swap control, which
    once caught a false positive.

    Returns (prompt, slot_of_REQ), slot ∈ {'1','2'}.
    """
    req = READING_REQ.format(p=p, r=r, q=q)
    alt = READING_ALT.format(p=p, r=r, q=q)
    a, b = (req, alt) if req_first else (alt, req)
    prompt = CONTRACT_TEMPLATE.format(premises=premises, reading_a=a, reading_b=b,
                                      opt_a=opts[0], opt_b=opts[1], opt_c=opts[2])
    return prompt, ('1' if req_first else '2')


# Between the colon and the digit, tolerate a few decorative characters (`[`, `**`, even a
# copied-out placeholder `N`), but never across a line break and never past any digit —
# otherwise it would falsely grab a "Reading 2" in the body text.
_REL = re.compile(r'RELATION\s*:?[^\d\n]{0,4}([12])\b', re.IGNORECASE)
VALID_ANSWERS = ('a', 'b', 'c')


def parse_relation(text):
    """Take the **last** occurrence of the RELATION line (mid-reasoning the model may write a wrong one first, then correct it). Returns None if absent."""
    m = _REL.findall(text)
    return m[-1] if m else None


def implied_answer(relation_is_req, modus):
    """From the reading the model itself chose, mechanically derive the answer it **should** give. Does not look at gold."""
    if relation_is_req:
        return 'c'
    return 'a' if modus == 'ponens' else 'b'
