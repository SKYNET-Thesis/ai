# AGENTS.md

## Project overview

Low-cost VLA manipulation on SO-101 / SO-ARM101. Single-arm pick-and-place and light assembly, trained with LeRobot. Primary policy: **ACT**; comparison policy: **SmolVLA** fine-tuned on the same demonstrations. Output artifacts are (1) training code and configs, (2) an auditable final report with success-rate evidence.

Target task: single-item packing / sorting / light insertion, items ≤ 500 g, within 400 mm reach (conservative). Do not assume the 500 mm third-party spec.
