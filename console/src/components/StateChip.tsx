import type { State } from "../types";
import { STATE_LABEL, STATE_VAR } from "../lib/format";
import "./StateChip.css";

/**
 * A state, as dot + fill + its literal name.
 *
 * Never colour alone. Reviewers screenshot these into reports, print them in
 * greyscale, and read them with colour-vision differences; the word has to
 * carry the meaning by itself.
 */
export function StateChip({
  state,
  size = "md",
}: {
  state: State;
  size?: "sm" | "md";
}) {
  return (
    <span
      className={`chip chip--${size}`}
      style={{
        color: `var(--s-${STATE_VAR[state]})`,
        background: `var(--s-${STATE_VAR[state]}-bg)`,
      }}
    >
      <i aria-hidden="true" />
      {STATE_LABEL[state]}
    </span>
  );
}
