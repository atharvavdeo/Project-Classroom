/** Turn the pipeline's vocabulary into English a reviewer can read.
 *
 * The codes are deliberately machine-shaped -- `SAM3_PHONE_NAMED`,
 * `no_dominant_equipment_explanation` -- because they are a closed vocabulary
 * that has to survive being written to disk and diffed. None of that is a
 * reason to show them to a person. Every mapping here is presentation only;
 * the underlying code is what the record still carries.
 */

/** Plain-English question each fusion condition is really asking. */
const CONDITION_QUESTION: Record<string, string> = {
  proposal_survives_geometry: "Was the object at their hand, and small enough?",
  sam3_supports_or_cannot_exclude: "Did the second model back it up?",
  associated_with_this_person: "Was it this person's, not a neighbour's?",
  lasts_or_recurs: "Did it last, or keep happening?",
  // Phrased so Yes always means "passed". The condition is
  // `no_dominant_equipment_explanation`, so asking "is there an innocent
  // explanation?" inverted it -- a passing row read "Yes ... not ordinary desk
  // equipment", which contradicts itself.
  no_dominant_equipment_explanation: "Ruled out ordinary desk equipment?",
};

/** What a pass and a fail each mean, in words, so the reviewer never has to
 *  translate a boolean into a consequence. */
const CONDITION_MEANING: Record<string, { pass: string; fail: string }> = {
  proposal_survives_geometry: {
    pass: "The object was close to their hand and the right size.",
    fail: "Nothing was close enough to a hand, or it was too large to be held.",
  },
  sam3_supports_or_cannot_exclude: {
    pass: "A second model looked at the crop and agreed something was there.",
    fail: "The second model could not back this up.",
  },
  associated_with_this_person: {
    pass: "The object can be tied to this person specifically.",
    fail: "The object could belong to someone else nearby.",
  },
  lasts_or_recurs: {
    pass: "It went on long enough, or happened repeatedly.",
    fail: "Too brief to be worth a reviewer's time.",
  },
  no_dominant_equipment_explanation: {
    pass: "Most of what was seen was not ordinary desk equipment.",
    fail: "This is mostly keyboard, mouse or monitor — normal desk equipment.",
  },
};

export function conditionQuestion(name: string): string {
  return CONDITION_QUESTION[name] ?? humanizeCode(name);
}

export function conditionMeaning(name: string, passed: boolean): string {
  const m = CONDITION_MEANING[name];
  if (!m) return "";
  return passed ? m.pass : m.fail;
}

/** Reason codes, as short human labels. Anything unmapped falls back to a
 *  title-cased version of the code rather than being hidden -- a code the
 *  console does not recognise is still something the reviewer should see. */
const REASON_LABEL: Record<string, string> = {
  SAM3_PHONE_NAMED: "Second model called it a phone",
  SAM3_SUPPORTED: "Second model agreed",
  SAM3_NOT_CONFIRMED: "Second model could not confirm",
  SAM3_EQUIPMENT_CONTEXT: "Called ordinary desk equipment",
  SAM3_UNAVAILABLE: "Second model unavailable",
  SAM3_NOT_RUN: "Not sent to the second model",
  OBJECT_PROPOSAL_PAPER: "Possible paper",
  OBJECT_PROPOSAL_PHONE: "Possible phone",
  OBJECT_AT_WRIST: "Object at the hand",
  OBJECT_AWAY_FROM_WRIST: "Object away from the hand",
  OBJECT_TOO_LARGE: "Object too large to be held",
  OBJECT_NO_WRIST_RESOLVED: "No hand resolved for this object",
  OBJECT_STATIONARY_ON_DESK: "Object sat still on the desk",
  OBJECT_HANDLING_SUSTAINED: "Handled continuously",
  OBJECT_HANDLING_RECURRENT: "Handled repeatedly",
  DFINE_OBJECT_CONTEXT: "Workstation context only",
  SEAT_UNCALIBRATED: "Seat not calibrated",
  TALKING_NOT_MEASURABLE: "Talking cannot be measured",
  ORIENTATION_NOT_ASSESSABLE: "Head direction not assessable",
  ORIENTATION_AWAY_SUSTAINED: "Turned away for a while",
  ORIENTATION_DOWN_PROLONGED: "Head down for a while",
  ORIENTATION_TOWARD_NEIGHBOUR: "Turned toward a neighbour",
  ORIENTATION_CHANGE_FREQUENT: "Looked around repeatedly",
  TRACK_COVERAGE_LOW: "Seen for only part of the recording",
};

const FLAG_LABEL: Record<string, string> = {
  target_object_near_hands: "Object near their hands",
  persistent_object_beside_seat: "Something left beside the seat",
  sustained_attention_away: "Turned away for a while",
  frequent_orientation_change: "Looked around repeatedly",
  prolonged_downward_orientation: "Head down for a while",
  no_orientation_baseline: "Not enough views of the head",
  hand_below_desk_line: "Hand below the desk",
  absent_from_frame: "Left the frame",
  low_coverage: "Only briefly on camera",
};

/** SAM 3 verdicts as reviewer-facing phrases. `unsupported` is deliberately
 *  "could not confirm", never "no phone": the contract is explicit that not
 *  confirmed does not mean absent. */
const VERDICT_LABEL: Record<string, string> = {
  corroborated: "Second model agreed",
  phone_supported: "Second model called it a phone",
  reclassified_as_phone: "Second model called it a phone",
  phone_was_paper: "Second model called it paper",
  unsupported: "Could not confirm",
  unsupported_phone: "Could not confirm a phone",
  suppressed: "Called desk equipment",
};

/** What a reviewer's own answer is called back to them. Phrased as the thing
 *  they decided, not as a state name -- "dismissed by reviewer" describes the
 *  record; "not a violation" describes what they said. */
const DECISION_LABEL: Record<string, string> = {
  human_confirmed: "confirmed",
  human_dismissed: "marked not a violation",
  needs_better_view: "left open for a second look",
};

export function decisionLabel(decision: string): string {
  return DECISION_LABEL[decision] ?? humanizeCode(decision);
}

export function reasonLabel(code: string): string {
  return REASON_LABEL[code] ?? humanizeCode(code);
}

export function flagLabel(flag: string): string {
  return FLAG_LABEL[flag] ?? humanizeCode(flag);
}

export function verdictLabel(verdict: string | null | undefined): string {
  if (!verdict) return "Not adjudicated";
  const base = verdict.startsWith("suppressed")
    ? "suppressed"
    : verdict.startsWith("unsupported")
      ? verdict
      : verdict;
  return VERDICT_LABEL[base] ?? humanizeCode(verdict);
}

/** Last resort: SNAKE_CASE or kebab-case to Sentence case. */
export function humanizeCode(code: string): string {
  const words = code.replace(/[_-]+/g, " ").trim().toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Object class names as a reviewer would say them. */
const CLASS_LABEL: Record<string, string> = {
  phone: "phone",
  paper_like_object: "paper",
  book_or_notebook_like_object: "paper or notebook",
  secondary_paper_chit: "paper",
  keyboard: "keyboard",
  mouse: "mouse",
  monitor: "monitor",
  bottle: "bottle",
};

export function classLabel(cls: string | null | undefined): string {
  if (!cls) return "object";
  return CLASS_LABEL[cls] ?? humanizeCode(cls).toLowerCase();
}

/** One line summarising why this person is in front of the reviewer. */
export function headline(
  supported: boolean,
  cls: string | null | undefined,
  verdict: string | null | undefined,
): string {
  if (supported) {
    return `A second model confirmed a ${classLabel(cls)} at this person's hand.`;
  }
  if (verdict) {
    return `${verdictLabel(verdict)} — the detector proposed a ${classLabel(cls)}.`;
  }
  return `The detector proposed a ${classLabel(cls)} near this person's hand.`;
}
