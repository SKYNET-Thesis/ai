# VISTA: A Vision-Language-Action System for Tabletop Manipulation with Robot Arms

![Status](https://img.shields.io/badge/status-in--development-blue)
![Project Code](https://img.shields.io/badge/project_code-FA2AI63-brightgreen)
![Group](https://img.shields.io/badge/group-GFA26AI04-orange)

> Replacing SmolVLA's OCR-focused vision backbone with modern, lightweight VLMs to boost robotic manipulation performance.

---

## Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Project Objectives](#-project-objectives)
- [Methodology & Technology Stack](#-methodology--technology-stack)
- [Evaluation Tasks](#-evaluation-tasks)
- [Project Structure](#-project-structure)
- [Project Team](#-project-team)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

This project investigates and proposes a solution to a key limitation in the **SmolVLA (Small Vision-Language-Action)** model — a compact and efficient policy for robotic manipulation. The core of our proposal is to replace SmolVLA's original vision-language model (VLM) backbone, `SmolVLM`, which is primarily pretrained on document-reading and OCR tasks, with more advanced and capable lightweight VLMs.

We hypothesize that this replacement will significantly improve SmolVLA's performance on manipulation tasks while preserving its status as a highly efficient and accessible model.

To validate this hypothesis, we will conduct a series of experiments using the low-cost **SO-ARM101** robotic arm and the **LeRobot** library. Our methodology involves benchmarking the modified SmolVLA models against the original SmolVLA and the ACT baseline on two tabletop manipulation tasks of varying difficulty:

- **Desk Stationery Decluttering**
- **Conveyor Belt Object Sorting**

---

## ❗ The Problem

Before 2023, robotic manipulation research was largely confined to large, expensive hardware, creating a significant barrier to entry. The introduction of systems like ALOHA and the ACT (Action Chunking with Transformers) paper revolutionized the field by demonstrating that fine-grained manipulation tasks could be achieved on low-cost hardware.

This breakthrough catalyzed the development of accessible tools like the **LeRobot** library and the **SO-ARM101** robotic arm. As hardware becomes more affordable, policies like ACT are being superseded by newer, more lightweight models such as **SmolVLA**.

However, despite being a compact and efficient policy, SmolVLA suffers from a significant limitation. As noted in Section 5.1 of its own paper, it relies on an off-the-shelf VLM, `SmolVLM`, which is pretrained mainly on **document reading and OCR tasks**. Consequently, `SmolVLM` is not proficient at localization, describing, or identifying objects within images — a critical capability for robotic manipulation.

This design trade-off was made to create a small, efficient, and community-driven VLA that drastically reduces training and inference costs. This project aims to address this specific problem by swapping SmolVLA's OCR-focused VLM backbone with more recent, visually adept backbones.

---

## 💡 Our Solution

We propose to enhance SmolVLA's performance by replacing its `SmolVLM` backbone with newer, lightweight VLMs that are better suited for visual tasks, while maintaining its core advantages as the most lightweight and compact VLA model.

After careful consideration, we have selected two suitable VLM models for testing and verifying the legitimacy of our solution:

| Candidate VLM            | Parameters | Rationale                                              |
| ------------------------ | ---------- | ------------------------------------------------------ |
| **LMF2.5-VL-0.45B**      | 0.45B      | Lightweight, strong visual processing capabilities     |
| **FastVLM-0.5B**         | 0.5B       | Compact, efficient, and adept at image understanding   |

Both models are potential candidates for replacement due to their lightweight characteristics and their inherent ability to process images effectively.

---

## 🎯 Project Objectives

1. **Identify Limitations** — Formalize the performance limitations of the original SmolVLA model, specifically those stemming from its OCR-focused VLM backbone.
2. **Propose and Implement a Solution** — Replace the `SmolVLM` backbone with modern, lightweight VLMs (LMF2.5-VL-0.45B and FastVLM-0.5B) to improve visual understanding.
3. **Benchmark Performance** — Systematically evaluate the modified SmolVLA models against the original SmolVLA and the ACT baseline on two distinct tabletop manipulation tasks.
4. **Analyze and Report** — Analyze the results to validate the hypothesis, quantify the performance improvements, and document the entire project lifecycle, including methodology, system design, and conclusions.

---

## 🛠️ Methodology & Technology Stack

### Hardware

- **Robotic Arm:** SO-ARM101 (low-cost, accessible robotic arm)
- **Setup:** Tabletop manipulation environment

### Software & Libraries

- **Core Framework:** [LeRobot](https://github.com/huggingface/lerobot) — open-source library for the entire robot learning stack
- **Baseline Models:**
  - ACT (Action Chunking with Transformers)
  - Original SmolVLA (with `SmolVLM` backbone)
- **Candidate VLM Backbones:**
  - `LMF2.5-VL-0.45B`
  - `FastVLM-0.5B`

---

## 🧪 Evaluation Tasks

The performance of the modified models will be benchmarked on two tasks of varying difficulty:

| # | Task                              | Description                                                              | Difficulty |
| - | --------------------------------- | ------------------------------------------------------------------------ | ---------- |
| 1 | **Desk Stationery Decluttering**  | Agent must identify and sort different stationery objects on a desk.     | Easy       |
| 2 | **Conveyor Belt Object Sorting**  | Agent must identify and sort objects from a moving conveyor belt.        | Hard       |


## 👥 Project Team

**Group Name:** GFA26AI04  
**Project Code:** FA2AI63

| Name                  | Student ID | Role        |
| --------------------- | ---------- | ----------- |
| Nguyễn Gia Phát       | SE190236   | Team Member |
| Nguyễn Minh Đức       | SE         | Team Member |
| Nguyễn Duy Sơn        | SE         | Team Member |
| Nguyễn Lê Anh Quân    | SE         | Team Member |

**Supervisor:** Mr. Giang-Nam Luu, Ph.D

## Project tree

D:.
├───configs
├───data
├───experiments
├───Phowhisper_Model
│   ├───data
│   │   └───audios
│   ├───models
│   ├───outputs
│   └───src
├───reports
├───results
├───scripts
├───src
└───thesis-ai
    └───.venv
        ├───Lib
        │   └───site-packages
        │       └───__pycache__
        └───Scripts