import type { Condition } from "../types";
import "./ConditionLedger.css";

/**
 * All five fusion conditions, pass and fail alike, each with its arithmetic.
 *
 * There is deliberately no `showFailuresOnly` prop and no collapse. The output
 * contract requires the whole argument including the parts against, and a card
 * that cannot show its own working does not get shown. Hiding the failing rows
 * would turn an argument into an assertion.
 */
export function ConditionLedger({ conditions }: { conditions: Condition[] }) {
  if (!conditions.length) {
    return (
      <p className="ledger__empty">
        No conditions were evaluated for this track.
      </p>
    );
  }

  const passed = conditions.filter((c) => c.passed).length;

  return (
    <div className="ledger">
      <div className="ledger__head">
        <span>Conditions</span>
        <span className="mono">
          {passed} of {conditions.length} passed
        </span>
      </div>
      <ul>
        {conditions.map((c) => (
          <li key={c.name} className={c.passed ? "is-pass" : "is-fail"}>
            <span className="ledger__verdict">{c.passed ? "PASS" : "FAIL"}</span>
            <span className="ledger__body">
              <b>{c.name}</b>
              <span>{c.detail}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
