# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** D1  
**Challenge:** Dandelion Africa  
**Team members:** Christopher Le, Ali Parpia, Davis Kim  
**Repository:** [Link](https://github.com/SCU-CSEN166/Dandelion1)  
**Last updated:** 10/30/2026

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy
All used datasets are publicly accessible and linked when mentioned. These datasets do not contain real personal information.

### Known limitations
- The system may lose information on large chats when using models with small input sizes due to repeated summarizations
-   This can also drastically increase processing time
- The system does not yet properly insert transcribed audio into text prior to generation
- The system cannot find and redact all information, sometimes leading to mismatches between names and censored aliases

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
| WhatsApp-style-Conversations-Ar-En-Zh-Ko-Logs-Pack | [Link](https://huggingface.co/datasets/ChristianCaballero/WhatsApp-style-Conversations-Ar-En-Zh-Ko-Logs-Pack) | Training / Evaluation | EN, AR, CH, KO | Fictional WhatsApp-style chats; not intended to resemble any real people or events
| Custom Made |  | Training / Evaluation | EN | Fictional WhatsApp exports; not intended to resemble any real people or events  

We're going to make up our own text dataset, since we could not find a public WhatsApp dataset meant to resemble a professional organization's communications.
- **Data preparation:**  
To prepare our data, we will clean it by removing any content labeled irrelevant by a classifier. Then we will anonymize it by identifying any personal information and blacking it out. We will label and annotate data by marking what data is important, irrelevant, and personal. Then split the data into those 3 sections and blacking out all personal information, taking the important information and using it for the output, and removing all unimportant data.

- **Data limitations:**  
Dealing with fully irrelevant data, human errors, and informal language.

### 4. Proposed evaluation
| Metric / test | What it measures | How it will be calculated | Target / success criterion |
|---|---|---|---|
| Recall | How many outcomes it correctly reports | TP/(TP+FN) | 90% (It's bad if any major outcomes are not reported) |
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
| Christopher Le | Evaluation | README contributions | Create evaluation systems |
| Davis Kim | Models | README contributions | Prepare models for each stage of processing |
| Ali Parpia | Data | README contributions | Create synthetic data |

---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status
- **What runs end-to-end now?** The program loads a given dataset and scans/redacts personal identifying information following known patterns. For the sake of performance, hyperscan is used instead of the built-in re module. It then passes the text through a sequence-to-sequence model to summarize it. If the input token count is greater than the maximum sequence length, it will split the tokens into blocks and summarize them. This process is repeated with the separate generated summaries until it fits within the maximum sequence length, meaning only a single output sequence is generated. This is the final summary.
- **How to run it:** Use the following command line arguments. -d is required, and no result will be seen if neither -o nor --input_out are specified. -il should be set based on the model used.  
-d or --dataset: Name of the dataset to process, either locally stored or from Huggingface  
-s or --summarizer: Name of the summarization model to use, either locally or from Huggingface  
-l or --length: Minimum length of the summary  
-il or --input_length: Maximum input length in tokens that the summarization model can handle  
-b or --use_blocks: Whether text longer than the max input length should be split and processed in blocks. If this flag is not used, the input will be truncated  
-o or --output: File to write the output summary to  
--input_out: File to write the used (redacted) input text to  
- **Model(s) and data actually used:**  
WhatsApp-style-Conversations-Ar-En-Zh-Ko-Logs-Pack [Link](https://huggingface.co/datasets/ChristianCaballero/WhatsApp-style-Conversations-Ar-En-Zh-Ko-Logs-Pack)  
facebook/bart-large-cnn  
google/flan-t5-large  
- **Example 1:** Input: ******: Pogba\n****: Pogboom\n******: what a s strike yoh!\n****: was off the seat the moment he chopped the ball back to his right foot\n******: me too dude\n****: hope his form lasts\n******: This season he's more mature\n****: Yeah, Jose has his trust in him\n******: everyone does\n****: yeah, he really deserved to score after his first 60 minutes\n******: reward\n****: yeah man\n******: cool then \n****: cool → Output: Richie and Clay saw a very good football game, with one football player chopping the ball back to his foot, which was particularly exciting. Jose has trust in that player.
- **Example 2:** Input: Person0: r/dogs\nPerson1: Could the break up/move have made my dog depressed?\nPOPerson15: am currently going through the heartbreak of a long term relationship that I ended and I think my dog is feeling just as terrible because of it. \n\nMy boyfriend and I adopted him 2 years ago but he's really been my dog because my boyfriend rarely paid attention to him. Even so, my dog was obsessed with him. Always wanted to play and if he was around, it's like I was invisible. Anyway, we lived in a two bedroom and gave my dog run of the second bedroom while we were at work. He hated the neighborhood because of all the noise so his walks were always short but he loved sitting in the sunlight in his bedroom and was always down to play. Basically, he was a happy dog. \n\nThen we moved to the west cost and my dog LOVED IT! Outside time was his favorite time. He would always stare longingly at the door and loved to go out to the balcony and people watch. Time of his life. \n\nThe breakup came around and my dog and I moved back to the east coast into my moms place where I am sharing a bedroom with my two siblings. This place drains the life out of me because there is no sunlight. But my dog is taking it pretty badly. It's been just over a week and he's slowly becoming more lethargic and withdrawn. His food hadn't changed at all and I play with him just as much if not more than before and he has two other people that smother him with petting all the time. \n\nHe keeps hiding under the bed and won't respond to commands. He sleeps all day and only wants to play for a few minutes. He hates going out now and the sight of his leash sends him running. He hasn't touched his food all day today. I think he hates this place more than I do. \n\nMy sister took him to her friend's place so he can play with two other dogs and she told me he was wagging his tail and eager to play and roam around. \n\nThis is making me feel worse. I want to make him feel better. What do I do? → Output: moved back to east coast from west coast after breakup and my dog is not himself anymore. What do I do to make him better?

### Attempts and outcomes
| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
|---|---|---|---|
| Use regex matching for identify PII instead of an AI model | Partly | PII with clear formats such as emails and phone numbers are successfully identified. However, names are not so easily identified, as they are difficult to distinguish from regular words. Additionally, redacting names gives the summarizer less context to work with. | Assuming the format of the exported data is known, we can identify and store known usernames, and then search messages for similar words. |
| Start with small sequence-to-sequence models for summarization rather than big decoder-only models. | Partly | Sequence-to-sequence models tend to copy sentences rather than summarizing them, even with generation parameters meant to prevent this. Additionally, small models with specific training data may hallucinate often. For example, bart-large-cnn often hallucinated that a CNN reporter was present due to being trained only on CNN journalism. | We could use larger sequence-to-sequence models, or switch to instruction-based decoder-only models. |
| Compress input text by repeatedly processing it in blocks to get around the input size limit | Yes | This successfully generated a final summary containing information about several conversations included in the dataset (provided the summarization model didn't hallucinate or copy sentences verbatim, but that's a separate issue). See summary_example.txt | Currently, splitting can happen in the middle of conversations. It may be better to split based on message dates, so that entire conversations are not separated into different summaries. |
| SAMSum evaluation: 50 test chats, ROUGE vs. human summaries (bart-large-cnn) | Partly | results/samsum_bart_plain.json: ROUGE-1 0.332, ROUGE-2 0.1275, ROUGE-L 0.26 | which gives us a baseline score. The model most of the time copies lines instead of actually summarizing. Only 50 examples. |
| Same evaluation with redaction applied first | Partly | results/samsum_bart_redacted.json: ROUGE-1 0.2693, ROUGE-2 0.1271, ROUGE-L 0.219 (run with an older redact.py, commit 8bd50b9) | Redaction lowered ROUGE-1 by about 0.06. Needs a rerun with the current redaction. |
| Regex redaction on 50 FarmerChat Kenya queries | No | 0 of 50 queries changed; a test line's Kenyan phone number and name were missed | Regex sometimes misses Kenyan phone formats, names and place names. Next step is to add a phone pattern, try MasakhaNER 2 ( data set ). |
| Outcome-recall evaluation (eval.py): ROUGE + NLI fact recall on 20 SAMSum test chats, google/flan-t5-large summarizer, facebook/bart-large-mnli as the NLI checker | Partly | dataset_outputs/knkarthick_samsum_test_google_flan-t5-large.json: ROUGE-1 R 0.296 / P 0.512, ROUGE-2 R 0.155 / P 0.256, ROUGE-L R 0.258 / P 0.433. NLI outcome recall 0.146 | NLI is low partly because redaction turns names into Person0/Person1 while the reference summaries keep real names. | Next: set max_new_tokens, make generation deterministic for eval, run on 200+ samples. |
| Outcome-recall evaluation (eval.py) on 20 Reddit TL;DR posts (CarperAI/openai_summarize_tldr test split), google/flan-t5-large summarizer, facebook/bart-large-mnli NLI | No | results/CarperAI_openai_summarize_tldr_test_google_flan-t5-large.json: ROUGE-1 R 0.083 / P 0.225, ROUGE-2 R 0.016 / P 0.053, ROUGE-L R 0.059 / P 0.164; NLI outcome recall 0.00 | Much worse than SAMSum (ROUGE-1 R 0.296). Summaries were often off-topic or hallucinated. Inspecting the inputs showed redact.py was corrupting them the dataset's SUBREDDIT:/TITLE:/POST: labels were treated as names. Fix the redact.py offset, don't treat field labels as names. |
| Ran eval.py on the full 200-sample SAMSum set | No | UnicodeDecodeError in redact.py scan_batch | Found an offset bug in redact.py: the offset is overwritten instead of added to after each replacement, so after a few matches it slices the wrong bytes (and can redact the wrong text). Fix: offset += delta. |

### Individual contributions during that week
| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
|---|---|---|---|---|
| Davis Kim | Redaction and summarization | Worked | Commits through the week of 11/2 | Improve text splitting logic, improve name redaction |
| Ali Parpia | Found real public datasets (FarmerChat Kenya, SAMSum). I sampled 50 FarmerChat queries for hand labeling (sample_farmerchat.py). Wrote and ran a ROUGE evaluation of our summarizer on SAMSum, with and without redaction (eval_summaries.py). | Partly | Commits 05d11b7 and 09ba3be; results/samsum_bart_plain.json and results/samsum_bart_redacted.json | Label the 50 queries (PII and key info) and rerun the redacted evaluation. |
| Christopher Le | Found usable datasets. Wrote eval.py, an evaluation script that measures outcome recall with ROUGE recall/precision and NLI fact checking. | Partly | eval.pu. dataset_outputs/knkarthick_samsum_test_google_flan-t5-large.json and CarperAI_openai_summarize_tldr_test_google_flan-t5-large| Full 200-sample run crashed on a redact.py offset bug. Next: fix it, add max_new_tokens to summarize.py, rerun on 200+ SAMSum chats and on NotDevinLe/reddit_tldr. |
---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup
- **Development data:** [WhatsApp-style-Conversations](https://huggingface.co/datasets/ChristianCaballero/WhatsApp-style-Conversations-Ar-En-Zh-Ko-Logs-Pack) [openai_summarize_tldr](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr) [synthetic-chat-logs-pii-spans](https://huggingface.co/datasets/aitrainingdatabroker/synthetic-chat-logs-pii-spans) [pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) [samsum](https://huggingface.co/datasets/knkarthick/samsum)
- **Metrics and rationale:**
ROUGE: The ultimate goal of the project is generate human-readable and effective summaries of text chats and, more specifically, the measurable field outcomes they discuss. So, given a dataset centered around summarizing with good, human-written summary examples, ROUGE scores are a great fit given its purpose for comparing AI-generated text with human text.
Recall: For redaction, we 
- **Comparison:** Original approach used a sequence-to-sequence model with a barebones prefix to generate summaries. Current approach uses a text generation model with a more deliberate chat system prompt with an additional preprocessing stage. Current approach retains the one-at-a-time block summarization so that VRAM is not exhausted

Important note: Due to the redaction stage's use of aliases and the potential for missed names, NLI evaluation can be artificially reduced due to name mismatches. Additionally, the evaluation scripts currently treat redaction as an optional step and have no knowledge of the unredacted info, limiting the information it has. Additionally, even though the current outputs are clear and helpful from a human perspective, the ROUGE scores collapse because of the summarizer following a report template, which does not match the expected results when calculating the ROUGE scores. As a result, we've determined that our previous evaluation metrics must be changed given how significantly we updated the project this week.
| System / version | Metric | Value | Notes |
|---|---:|---:|---|
| Baseline | Rouge1 recall : CarperAI/openai_summarize_tldr : google/flan-t5-large | 0.0827 | flan often erroneously copied sequences |
| Baseline | Rouge1 precision : CarperAI/openai_summarize_tldr : google/flan-t5-large | 0.225 | flan often erroneously copied sequences |
| Baseline | Rouge2 recall : CarperAI/openai_summarize_tldr : google/flan-t5-large | 0.0165 | flan often erroneously copied sequences |
| Baseline | Rouge2 precision : CarperAI/openai_summarize_tldr : google/flan-t5-large | 0.0528 | flan often erroneously copied sequences |
| Baseline | Rouge3 recall : CarperAI/openai_summarize_tldr : google/flan-t5-large | 0.0594 | flan often erroneously copied sequences |
| Baseline | Rouge3 precision : CarperAI/openai_summarize_tldr : google/flan-t5-large | 0.164 | flan often erroneously copied sequences |
| Baseline | samsum flan nli_outcome_recall | 0.1463 | Important chat outcomes were not summarized due to seq2seq models too frequently copying sequences, even with sampling with high temperature generation parameters |
| Baseline | Redaction recall targeted | 0.193 | Needs improvement for phone numbers and names |
| Baseline | samsum BART plain rouge1 : bart-large-cnn | 0.332 | BART often erroneously copied sequences |
| Baseline | samsum BART plain rouge2 : bart-large-cnn | 0.1275 | BART often erroneously copied sequences |
| Baseline | samsum BART plain rougeL : bart-large-cnn | 0.26 | BART often erroneously copied sequences |
| Baseline | samsum flan plain rouge1 flan-t5-large | recall=0.2734, precision=0.4322 | flan often erroneously copied sequences |
| Baseline | samsum flan plain rouge2 flan-t5-large | recall=0.1258, precision=0.1975 | flan often erroneously copied sequences |
| Baseline | samsum flan plain rougeL flan-t5-large | recall=0.2311, precision=0.3610 | flan often erroneously copied sequences |
| Baseline | openai_summarize_tldr flan rouge1 flan-t5-large | recall=0.1069 precision=0.2582 | flan often erroneously copied sequences |
| Baseline | openai_summarize_tldr flan rouge2 flan-t5-large | recall=0.0205 precision=0.05496 | flan often erroneously copied sequences |
| Baseline | openai_summarize_tldr flan rougeL flan-t5-large | recall=0.08074 precision=0.2008 | flan often erroneously copied sequences |
| Current | knkarthick/samsum Qwen/Qwen3-0.6B rouge1 for n=50 | F-score=0.1633 recall=0.6002 precision=0.0974 | Generated report is significantly longer than the human benchmark; See note above table |
| Current | knkarthick/samsum Qwen/Qwen3-0.6B rouge2 for n=50 | F-score=0.0433 recall=0.1667 precision=0.0257 | '' |
| Current | knkarthick/samsum Qwen/Qwen3-0.6B rougeL for n=50 | F-score=0.1139 recall=0.4356 precision=0.0675 | '' |

### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| Summarization | For a decoder-only model, follow the given template in the system prompt to create a categorized report | Template followed, report correctly generated | Success | Switched from encoder-decoder to decoder-only, used transformer library's chat template methods, changed generation parameters to disable sampling, add preprocessing stage to remove system messages, special characters, and embedded media strings |
| Resolving out of memory errors | VRAM stays within expected range (eg 4-6 GB for Qwen3 0.6B) | Allocated VRAM was as expected | Success | Generating summaries one at a time instead of passing all sequences at once, changing data format to bfloat16, regularly flushing CUDA |
| Redaction | Names and other PII redacted and replaced with aliases | Most PII redacted, names in chats not redacted, unusual phone number formats not supported | Failure | Need more queries for different PII formats, need to add support for finding words similar to known names |

- **What improved?** Chat summarization greatly improved. The generated report now contains multiple lines about field outcomes. Unlike last week, no copied sequences or blatant hallucinations were observed.
- **What did not improve?** Redaction and evaluations. The redaction module was bugfixed, but no new features were added. Our previous evaluation setup must now be changed, as the summarizer now follows a chat template that does not align with the expected human-written predictions in the rouge scoring.
- **Next experiment:** In the redaction stage, perform an additional pass searching for substrings similar to known names, and remove if found. As before, redaction will be evaluated with recall. Evaluations must also be made for audio transcriptions, likely with rouge scoring. It may be pertinent to classify and purge irrelevant text lines during the preprocessing stage, though small text generation models with proper chat templates have demonstrated the ability to discern which chat is relevant and which is not.

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| Davis Kim | Summarizer improvements/fixes: Testing decoder-only models, memory management, chat template + system prompt. Preprocessing stage for purging unnecessary text from the chat exports. Redaction bug fixes. | Worked | See commit history and [PR #1](https://github.com/SCU-CSEN166/Dandelion1/pull/1) | Improve redaction scanning, integrate audio transcription module, adjust evaluation setups |

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


