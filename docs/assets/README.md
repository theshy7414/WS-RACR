# Figures and videos

Asset guide for the images and videos used by the README and the website (`docs/index.html`).
The file names below are the ones the embed snippets at the end refer to; GitHub Pages serves `docs/`,
so the files live in `docs/assets/`.

General rules

- Images: PNG (or JPG for photos), 1600 px wide or more, light background, no personal data. Keep each file under 1 MB (`pngquant` / `squoosh` help).
- Videos: MP4 (H.264), 1920x1080, 30 fps, no audio needed unless stated, English captions burned in.
  GitHub renders an `.mp4` in the README only if it is under 10 MB; otherwise upload to YouTube and link a
  thumbnail. Keep recordings out of git history if large (`*.mp4` is git-ignored on purpose; force-add
  only the small final files with `git add -f`).
- Use the same terminology as the README: Layer 1 / 2 / 3 are the three decision layers; W<sub>t</sub> =
  {S<sub>t</sub>, O<sub>t</sub>, H<sub>t</sub>, R<sub>t</sub>}; outcomes REUSE / ADAPT / CLARIFY / REJECT.

## Figures

| # | File | What it shows | Source / how to make | Used in |
|---|---|---|---|---|
| F1 | `architecture.png` | System overview: perception (D435i top-down, D455 front) -> world state W<sub>t</sub> -> reasoning and memory (Layer 1 -> Plan RAG recall -> Layer 2 -> Layer 3 -> LLM planner) -> execution (MoveIt2 / TM driver). Mark the asynchronous perception pipeline and the three memories (interaction log, plan RAG, temporal log). | Redraw the thesis system-architecture figure (Ch. 3 / Ch. 4) as a clean vector figure; export PNG. | README (top), website "Approach" |
| F2 | `decision-flow.png` | One command through the decision layers, with the four outcomes. Boxes: utterance -> Layer 1 (split, route EXECUTE / ANSWER / ASK) -> Plan RAG recall (candidates only) -> Layer 2 hard gate (H<sub>t</sub> stop, R<sub>t</sub> gripper, O<sub>t</sub> target visible) -> Layer 3 rules 1-4 -> REUSE / ADAPT / CLARIFY / REJECT. Show the thresholds 0.85 / 0.75 / 0.50. | Draw from `docs/architecture.md` section 4. | README, website |
| F3 | `world-state.png` | The four components of W<sub>t</sub>, their sources and refresh path (TSM-ResNet18, YOLOv8-seg, MediaPipe gesture, robot IO), with the stabilisation step before the state is snapshotted for the orchestrator. | Thesis Ch. 3 figure, simplified. | `docs/architecture.md` |
| F4 | `workcell.jpg` | Photo of the real work cell: TM5-900, gripper, workpieces on the table, **D435i top-down** and **D455 front** cameras, human position. Add small labels for the two cameras. | Take a photo; no faces. | README, website |
| F5 | `perception-overlay.png` | One screenshot of the live perception view: YOLOv8-seg masks with object labels on the top-down image, the gesture/intent label from the front camera, current step class. | Screenshot from `python -m ws_racr.app.main --perception-only`. | README |
| F6 | `ablation.png` | Bar chart of the EXP4 accuracy: 31.8 % -> 56.8 % (+Layer 1) -> 70.5 % (+Layer 2) -> 93.2 % (all three). Optional second panel: EXP3 accuracy 48.8 % -> 93.8 %. Caption: "Layer ablation on 44 stratified scenarios" (see `docs/evaluation.md`). | Plot the numbers printed by `python -m evaluation.run_all --out results`; do not retype them from memory. | README "Evaluation", website "Results" |
| F7 | `cases.png` (optional) | Three small panels, one each for the example scenarios in `data/scenarios/` (execute / answer / ask): the utterance, the world state at that moment, the system's decision and spoken reply. | Take from the real system or from the JSON files. | `docs/architecture.md` |

## Videos

Record the real system. Each clip must show the **spoken command**, the **decision shown by the
system** (REUSE / ADAPT / CLARIFY / REJECT, from the log or UI), and the **robot or the answer**.

| # | File | Length | Content (scene by scene) | Used in |
|---|---|---|---|---|
| V1 | `demo-overview.mp4` | 60-90 s | (1) 5 s: work cell and title. (2) Command that is **reused** from a stored plan, robot executes. (3) Command that is **adapted** (same action, different object). (4) Ambiguous command ("give me that") -> **CLARIFY** question from the robot. (5) A command whose target object has been removed from the table -> **REJECT** with spoken reason. (6) End card with the repository URL. Show W<sub>t</sub> / decision log side by side with the camera view. | README top, website hero |
| V2 | `clip-reject-missing-object.mp4` | 15-20 s | Remove the target part, give the command that would normally succeed; show Layer 2 rejecting it although the stored plan is highly similar. | README "Approach", `docs/architecture.md` |
| V3 | `clip-multi-intent.mp4` | 15-20 s | One utterance with an action and a question (e.g. pick up a part and ask where the next one goes). Show the split into EXECUTE + ANSWER and the answer being spoken while the robot moves. | README |
| V4 | `clip-pointing.mp4` | 15-20 s | "Hand me **that one**" while pointing at one part; show the grounded target name overlay. | README |
| V5 | `clip-stop.mp4` (optional) | 10 s | Human stop gesture during execution: robot stops, W<sub>t</sub> shows the stop in H<sub>t</sub>. | `docs/architecture.md` |

## Embed snippets

README (below the title block):

```markdown
<p align="center"><img src="docs/assets/architecture.png" width="820" alt="WS-RACR architecture"></p>

https://github.com/theshy7414/WS-RACR/assets/...   <!-- drag demo-overview.mp4 into the GitHub editor -->
```

Website `docs/index.html` (in the "Approach" section, after the layer table):

```html
<video controls preload="metadata" style="width:100%;border-radius:8px" poster="assets/workcell.jpg">
  <source src="assets/demo-overview.mp4" type="video/mp4" />
</video>
```

and add `<img src="assets/architecture.png" alt="WS-RACR architecture" style="width:100%">`
above the ASCII diagram.
