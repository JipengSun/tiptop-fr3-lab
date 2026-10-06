# Four-orange pick-and-place demo

This extension keeps TiPToP's existing perception, SAM2 segmentation, M2T2 grasp generation, cuTAMP planning, and
execution pipeline. It adds a deterministic language-grounding step between Gemini detection and SAM2 segmentation
for the specific tabletop scene with four oranges and one box.

## What changes

When the instruction contains a supported selector, the grounder:

1. requires exactly four detected oranges;
2. renames them `orange_1` through `orange_4` from left to right in the wrist-camera image;
3. resolves numbered, ordinal, left/right, or top/bottom references in English or Chinese;
4. replaces any VLM-generated orange goals with exactly one `on(selected_orange, box)` predicate;
5. draws an operator-facing overlay with all ranked oranges, the selected target, box, selector, and final goal; and
6. stops before planning if the orange count, bounding boxes, or destination is ambiguous.

Instructions without a supported selector continue through the unmodified TiPToP behavior.

## Supported instruction examples

```text
put orange 3 in the box
put the second orange from the right in the box
put the left orange in the box
把从左数第二个橙子放进盒子
把第4个橙子抓到盒子里
```

Numbering is always left-to-right in the current capture image. “Top” and “bottom” refer to image coordinates, not
the robot's world frame. Avoid “middle orange” because a set of four has no unique middle object.

## Grounding-only dry run

After installing the TiPToP environment:

```bash
pixi run tiptop-orange-check "把从左数第二个橙子放进盒子" \
  --output /tmp/tiptop-orange-check
```

This command uses synthetic detections and does not contact Gemini, start GPU perception services, or move a robot.
It writes `orange_selection_viz.png` plus `perception/orange_grounding.json`, so the same visual and symbolic checks
can be reviewed without hardware.

## Planner dry run on the bench

Start the normal M2T2 and FoundationStereo services, place four oranges and one open box in the calibrated workspace,
and first run without executing:

```bash
pixi run tiptop-run --cutamp-visualize --no-execute-plan
```

Enter a supported instruction and verify in Rerun that:

- all four oranges have separate masks;
- `orange_1` through `orange_4` appear left-to-right;
- `orange_demo/selection` highlights only the commanded orange in green and the destination in blue;
- the goal contains only the selected orange and the intended box;
- the grasp is on the orange rather than the box; and
- the entire trajectory clears the table, camera mount, box rim, and other oranges.

Each grounded run also saves `orange_selection_viz.png` and `perception/orange_grounding.json` beside the normal
TiPToP capture artifacts. Only after those checks should an operator enable execution. Follow the upstream TiPToP
setup guide, use a low `time_dilation_factor`, keep the workspace clear, and keep a hand on the E-stop.

