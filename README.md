# 📚 Enabling Jurimetrics: Deploying Open‑Source Large Language Models for Empirical Insights into Brazilian Courts

**Enabling jurimetrics in Brazil with open-source large-language models**

![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg) ![R ≥ 4.3](https://img.shields.io/badge/R-%E2%89%A54.3-success) ![Python ≥ 3.10](https://img.shields.io/badge/Python-%E2%89%A53.10-success) 

Welcome! This repository accompanies my 2025 master thesis at the [Hertie School](http://www.hertie-school.org) MSc Data Science for Public Policy.
It contains every script, prompt, and artefact needed to reproduce the experiments that test whether open-source LLMs can pull structured facts from Brazilian judicial decisions. Since the raw data was potentially sensitive, it is not included in this repository. 

---

## 1 · Project snapshot

* **Research question** CCan open-source large language models be deployed in the Brazilian legal context to ac-
curately extract information, recognize entities, and draw conclusions from complex legal
documents, compared to leading proprietary systems?
* **Dataset** 258 TJ-SP drug-trafficking judgments (2017 – 18) with 50 hand-labelled features.
* **Models** 12 models (both open-source and commercial). Fine-tuning was performed on GPT-4o-mini (baseline)  Phi-4, Llama 3.2.
* **Key result** Fine-tuned Phi-4 increases the accuracy to 0.92—matching GPT-4o-mini while cutting inference cost by 25 %.

---

## 2 · Repository layout

```
.
├── R/                    # R scripts - mainly for data collection, cleaning & sampling
├── data/                 # ⬜ NOT in Git – add raw & processed files locally
│   ├── raw/
│   └── processed/
├── doc/                  # ⬜ NOT in Git until grading - Thesis PDF, defence slides, figure assets
├── evaluate_results/     # Final CSVs + plots for the manuscript
├── open_ai/              # Batch-job helpers for GPT-4 / GPT-4o
├── open_source_models/   # Unsloth + HF scripts for Gemma, Phi-4, Llama …
├── prompt/               # Portuguese prompt + Pydantic JSON schema
├── .gitignore            # Keeps large / sensitive files out of Git
└── README.md             # You are here
```

## 3 · Dependencies

| Stack             | Key packages                                                            |
| ----------------- | ----------------------------------------------------------------------- |
| **R ≥ 4.3**       | tidyverse, arrow, gt, jsonlite, reticulate, webshot2                    |
| **Python ≥ 3.10** | torch, transformers, unsloth, pydantic, langchain, tiktoken             |
| **System**        | CUDA GPU (12 GB +) for fine-tune scripts; Chrome/PhantomJS for webshot2 |

---
## 5 · How to cite

```bibtex
@mastersthesis{dornelles2025,
  author       = {Dornelles, Rodrigo F.},
  title        = {Enabling Jurimetrics: Deploying Open-Source Large-Language Models for Empirical Insights into Brazilian Courts},
  school       = {Hertie School},
  year         = {2025}
}
```

## 5 · Licence

*Code*  —  MIT
*Labelled dataset*  —  CC BY-NC-SA 4.0
Raw court documents remain under TJ-SP public-domain terms; redistribute responsibly.

---

## 6 · Contact

Questions or ideas?
Open an issue or email me at rodornelles@gmail.com

