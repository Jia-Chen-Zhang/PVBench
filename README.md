# PVBench

## News & Progress
**[Current Status]** 🚀 Dataset Released! The PVBench dataset is now fully open-sourced.

**[Coming Soon]** ⏳ The evaluation codebase (including the implementation of the Tiered Penalty Metric and strict evaluation pipelines) is currently being cleaned up and will be released very soon. Stay tuned!

## Dataset Summary
Evaluating LLMs with LLMs (LLM-as-a-Judge) has become a standard practice, but existing evaluators suffer from two critical flaws:

**1.** Pseudo-verification: Instead of carefully verifying the provided reasoning steps, judge models often rely on an outcome-matching shortcut (re-solving the problem in the background and comparing the final answer), leading to significant bias, especially when self-evaluating.

**2.** Overconfidence: Traditional binary evaluation (Correct/Incorrect) forces models to guess blindly when uncertain, masking hallucinations and inflating evaluation scores.


## Dataset Statistics
The domain distribution of the finalized dataset is as follows:

| Domain Category | Description | Count |
| :--- | :--- | :---: |
| **`math-s`** | Standard Mathematical Reasoning | 104 |
| **`math-h`** | Hard/Advanced Mathematical Reasoning | 125 |
| **`knowledge`** | Knowledge & Common Sense | 565 |
| **`code`** | Code Generation & Debugging | 39 |
| **`all`** | **Total Validated Pairs** | **833** |
