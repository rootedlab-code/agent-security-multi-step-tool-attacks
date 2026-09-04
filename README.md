# Provenance-Blindness in LLM Agents — attack and working note

The submitted attack algorithm and the working note for the Kaggle competition
**AI Agent Security — Multi-Step Tool Attacks**, hosted by OpenAI.

Final standing: **51st of 4252** teams on the private leaderboard, at 25.935.

## What is here

| file | what it is |
|---|---|
| `attack.py` | the attack algorithm as submitted |
| `working_note/working_note_v2.ipynb` | the working note, as an executable notebook |
| `working_note/working_note_v2_read.html` | the note rendered for reading — prose, figures and outputs, no code |
| `working_note/working_note_v2.html` | the same page with the code cells shown |

Start with `working_note_v2_read.html`. Open it in a browser; it needs no network.

## The notebook checks itself

The note is built so that its numbers cannot drift from the measurements behind them, and it
enforces that on itself rather than asking to be trusted:

- a **provenance gate** re-reads the notebook's own prose and resolves every numeric literal in it
  against an archived measurement, or against a passage that names its own external source;
- a **calibration gate** checks that every under-determined rate the note quotes carries an interval.

Both run when you execute the notebook. It carries the claim index and the values its figures read in
its second cell, so it runs standalone: verified in an empty directory, zero errors, both gates
passing, 843 of 843 literals resolving. Three things reduce without the full apparatus, and each says
so in its own output rather than passing silently.

## Revision of 2026-09-04

§10 (Defenses) is restated after the private leaderboard resolved the note's first forward prediction
against it: value provenance is **necessary and not sufficient**, and four defense classes replace the
earlier two-axis framing, each with what it held against, what it did not, and its cost, in a table a
defender can read alone. Every live section now leads with its current position and keeps what it used to say, verbatim and
dated, in a *Record of changes* at its end; §0–§5 stay as first posed, and the calibration gate
exempts only those. The same notebook runs as the current version of the note's Kaggle kernel.

## What is not here

The apparatus that produced the archived measurements — the experiment builders, the test suite and
the instruments — is not part of this release. §11 of the note states that plainly and claims no more
than what the notebook above can demonstrate about itself.

`attack.py` is published **exactly as submitted**, which is why its docstring still points at an
internal document that is not here. Fixing the pointer would have meant shipping something other than
the artefact that was scored, so the dangling reference is left in place and named here instead.

## License

MIT, see `LICENSE`. The vendored competition SDK the experiments ran against is the organizers' own,
published separately under MIT.

— `rootedlab-code` · rootedlab@proton.me
