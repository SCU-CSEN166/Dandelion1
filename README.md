# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** D1  
**Challenge:** Dandelion Africa  
**Team members:** Christopher Le, Ali Parpia, Davis Kim  
**Repository:** [Link](https://github.com/SCU-CSEN166/Dandelion1)  
**Last updated:** 10/30/2026

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy
[Explain where authorized data can be accessed. Do not commit personal information, private WhatsApp exports, API keys, credentials, or restricted grant/application records. Use synthetic or anonymized examples in the README.]

### Known limitations
[Keep a current list of what the system cannot reliably do.]

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A project plan (write in this README file) describing your problem formulation, proposed approach, each team member's responsibilities, proposed baseline models with Hugging Face links, proposed evaluation metrics, proposed data with links, and goals for each week.

### 1. Problem formulation and social impact
- **Problem:**  
Our goal is to convert exported WhatsApp messages and voice recordings into a structured report of activities and outcomes, as well as anonymize any personal information from the exports.
- **Intended users:**  
This system is built for Dandelion Africa with the purpose of making reports of field outcomes. Any other organization who also needs to create reports from internal communications could also use it. Most of this information is going to be used for reports or the creation of easy to read structured data of any activities that have been done.
- **Inputs and outputs:**  
The system will receive chat files or voice notes on WhatsApp and produce structured outputs later that are anonymized. A concrete example would be a worker fixing machinery and reporting the action or outcome on WhatsApp. The system should transform that chat or voice note into a structured report that any supervisor or member of the organization can easily view.
- **Success criteria:**  
The system would be useful and reliable if it writes consistent and useful reports, is able to accurately anonymize personal information, removes any unnecessary information, and correctly identifies important outcomes.
- **Constraints and risks:**  
Constraints we have is having good testable data, context window, dealing with AI hallucinations, human accents, typos, and informal language.

### 2. Proposed technical approach
- **Pipeline:**  
The main steps from input to output depends on if the message is in text or voice note. If it's a voice note, we need to transcribe it first. Then we can classify the the information into personal information, important information, and irrelevant information. Then we can split it off into 3 differnet tasks: anonymizing personal information, structuring important information into reports, and removing irrelevant information.
- **Baseline(s):**  
The simplest end-to-end method is fully blacking out any personal information, taking anything valuable information into an output, and removing irrelevant information. 
- **Proposed models/tools:**  
For text generation: llama 3.1 8B instruct https://huggingface.co/meta-llama/Llama-3.1-8B I selected this because it has a large enough parameter for all the text it will receive and it's an efficient model that can run on the WAVE HPC without taking up too many resources.
For audio to text: Whisper https://github.com/openai/whisper Whisper is one of the best audio to text transcribers and is accurate.
For classification: Bart large mnli https://huggingface.co/facebook/bart-large-mnli It's one of the most downloaded and used classification models on Huggingface. It's also backed by Meta which is a successful name in the AI industry. It has .4 billion parameters, which is more than enough for classifying.
- **Alternative approach:**  
If the baseline underperforms, then we will try and increase model sizes to see if the model's are the ones causing this performance as they're doing most of the heavy work or we can change our training data.

### 3. Proposed data
| Dataset/source | Link or access instructions | Purpose | Size / languages | Privacy or usage considerations |
|---|---|---|---|---|
| Custom Made |  | Training / Evaluation | EN | Fictional WhatsApp exports; not intended to resemble any real people or events  

We're going to make up our own text dataset, since we could not find a public WhatsApp dataset meant to resemble a professional organization's communications.
- **Data preparation:**  
To prepare our data, we will clean it by removing any content labeled irrelevant by a classifier. Then we will anonymize it by identifying any personal information and blacking it out. We will label and annotate data by marking what data is important, irrelevant, and personal. Then split the data into those 3 sections and blacking out all personal information, taking the important information and using it for the output, and removing all unimportant data.

- **Data limitations:**  
Dealing with fully irrelevant data, human errors, and informal language.

### 4. Proposed evaluation
| Metric / test | What it measures | How it will be calculated | Target / success criterion |
|---|---|---|---|
| Recall | How many outcomes it correctly reports | TP/(TP+FN) | 100% (It's bad if any major outcomes are not reported) |
| Recall | How much personal information was correctly identified and redacted | TP/(TP+FN) | 90% |

- **Evaluation examples:** [How will you build or select a development set?]
We'll build a development set by taking each input and building a list of all the important information we expect the system to find and report. 
- **Human verification:** [If relevant, who will check outputs and how?]
We will check output reports and the recall score. Depending on how good the reports are, or if they include too much text, then we'll tweak the model.
- **Failure cases to watch:** [At least two anticipated risks]
Including personal information or the report is missing any important details.

### 5. Team responsibilities
| Member | Primary responsibility | Specific Weekly contribution | Planned next task |
|---|---|---|---|
| Christopher Le | Evaluation |  | Create evaluation systems |
| Davis Kim | Models |  | Prepare models for each stage of processing |
| Ali Parpia | Data |  | Create synthetic data |

---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status
- **What runs end-to-end now?** [Describe exactly.]
- **How to run it:** [Commands, dependencies, configuration; refer to Setup below.]
- **Model(s) and data actually used:** [Names, versions, links, and any changes from the plan.]
- **Example 1:** Input: [non-sensitive example] → Output: [result]
- **Example 2:** Input: [non-sensitive example] → Output: [result]

### Attempts and outcomes
| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
|---|---|---|---|
| [Attempt] | [Status] | [Evidence] | [Learning] |

### Individual contributions during that week
| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
|---|---|---|---|---|
| [Name] | [Specific contribution] | [Status] | [Link] | [Next step] |

---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup
- **Development data:** [Source, size, selection, annotations, link or access instructions]
- **Metrics and rationale:** [Why these metrics reflect the intended use]
- **Comparison:** [Baseline versus current approach, if different]

| System / version | Metric 1 | Metric 2 | Notes |
|---|---:|---:|---|
| Baseline | [Value] | [Value] | [Notes] |
| Current | [Value] | [Value] | [Notes] |

### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| [Case] | [Expected] | [Observed] | [Status] | [Hypothesis] |

- **What improved?** [Evidence]
- **What did not improve?** [Evidence]
- **Next experiment:** [Specific change and how you will measure it]

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** [Commit hash]
- **Hidden-test metrics and leaderboard result:** [Results, when available]
- **Difference from development results:** [Compare; do not assume the hidden set is identical]

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
|---|---|---|---|---|
| [Case 1] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 2] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 3] | [Observation] | [Hypothesis] | [Impact] | [Fix] |

### Changes attempted after stress-testing
| Change | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
|---|---|---|---|
| [Change] | [Status] | [Result] | [Decision] |

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing
- **Team evaluated:** [Team ID]
- **One thing that worked well:** [Evidence]
- **One meaningful failure:** [Reproduction steps or example]
- **One actionable recommendation:** [Specific fix]
- **Feedback received from another team:** [Summary]

### Responsible AI assessment
- **Potential harms or affected users:** [Who could be affected and how?]
- **Privacy and data handling:** [How sensitive information is protected]
- **Reliability and unsupported claims:** [How outputs are verified or uncertainty is communicated]
- **Fairness, accessibility, and usability:** [Relevant concerns and tests]
- **Human oversight:** [Where a human should review or correct outputs]
- **Remaining limitations:** [What should not be trusted or deployed yet]

### Final improvements and outcomes
| Feedback / issue | Change made | Worked? (Yes / Partly / No) | Evidence | Remaining limitation |
|---|---|---|---|---|
| [Issue] | [Change] | [Status] | [Link or metric] | [Limitation] |

### Individual contributions during that week
| Member | Final implementation / evaluation / documentation contribution | Result (worked / partly / failed) | Evidence | Reflection / remaining work |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Reflection] |



### Final submission checklist
- [ ] Code and final system/prototype are in the repository
- [ ] Setup and run instructions are complete and tested
- [ ] Data sources, model versions, and dependencies are documented
- [ ] Evaluation results are reported and compared with the baseline
- [ ] Important failures and limitations are explained
- [ ] Hidden-test and peer-feedback findings are addressed
- [ ] Responsible AI considerations and human oversight are discussed
- [ ] Every team member's contributions are documented for every week

**Final commit / release:** [Commit hash or tag]

---

## Setup and Reproducibility (keep updated)

### Requirements
[Python version, libraries, compute requirements, model access]

### Installation
```bash
# Add installation commands here
```

### Run the system
```bash
# Add an example command here
```

### Run evaluation
```bash
# Add an example evaluation command here
```

### Repository structure
```text
project/
├── README.md
├── requirements.txt
├── src/
├── tests/
├── evaluation/
└── examples/
```


