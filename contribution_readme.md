# Contribution [#]: [88]

**Contribution Number:** [1]  
**Student:** [Mafeyisopin Ayeni]  
**Issue:** [https://github.com/vecna-labs/open-range/issues/88]  
**Status:** Phase III Complete

---

## Why I Chose This Issue
I chose issue #88 "Tag generated cyber tasks with MITRE ATT&CK technique IDs" 
because it aligns with my Python experience and my goal to contribute to a real 
AI/cybersecurity project. The issue is labeled "help wanted", has no assignees, 
and has a clear definition of done listed directly in the issue.

I'm interested in this because:
1. I'm comfortable with Python and regex patterns, which are the core skills 
   needed to build the TechniqueClassifier
2. The codebase area is contained; the issue lists exactly which files to 
   create and modify, so I know where to focus
3. The maintainer has clearly thought through the architecture, including a 
   sample YAML and flowchart, which gives me a strong starting point
4. I want to learn how real security tooling maps commands to threat 
   intelligence frameworks like MITRE ATT&CK

From reading the issue thread, I understand the current problem is that when 
an AI agent runs commands like "nmap" or "sqlmap", they are stored as plain 
untagged strings; invisible to the security community. My contribution will 
add a Python classifier that automatically tags those commands with standardized 
MITRE ATT&CK technique IDs, making training results comparable to industry 
benchmarks.

Left a comment on the issue introducing myself and expressing intent to work on it.


---

## Understanding the Issue

### Problem Description

[In your own words, what's broken or missing?]

### Expected Behavior

[What should happen?]

### Current Behavior

[What actually happens?]

### Affected Components

[Which parts of the codebase are involved?]

---

## Reproduction Process

### Environment Setup
Cloned the fork on Windows using Git and VS Code. Installed the `uv` 
package manager via `pip install uv`. Ran `python -m uv sync` to 
install all 23 project dependencies successfully. The only challenge 
was that running `uv sync` directly failed with "command not found" 
on Windows, which was resolved by using `python -m uv sync` instead. 
Ran `python -m uv run pytest` to verify the environment and confirmed 
640 tests passing. The 73 failures are pre-existing Windows-specific 
issues related to `os.killpg` which is a Linux-only function and are 
not related to this issue.

### Steps to Reproduce
1. Clone the fork and set up the environment using `python -m uv sync`
2. Run `grep -r "technique_id" .` across the entire codebase
3. Run `grep -r "MITRE" .` across the entire codebase
4. Observed result: Both searches return zero results inside the 
   actual source code, confirming that no MITRE ATT&CK technique 
   tagging exists anywhere in the codebase. The only matches for 
   "MITRE" were inside `contribution_readme.md` which is not part 
   of the project source code.

### Reproduction Evidence
- **Commit showing reproduction:** 
  https://github.com/mafeyisopin629-spec/updated-open-range/tree/fix-issue-88
- **Screenshots/logs:** 
  `grep -r "technique_id" .` returned no results
  `grep -r "MITRE" .` returned no results in source code
- **My findings:** 
  The `ontology.py` file defines structured attributes for cyber 
  tasks but contains no `technique_id` field. The 
  `families/pentest.py` file defines task families but has no 
  MITRE ATT&CK mappings. The feature requested in issue 88 is 
  completely absent from the codebase and ready to be implemented.

---

## Solution Approach

### Analysis
The cyber task data structures in 
`packs/cyber_webapp/cyber_webapp/ontology.py` and the task 
definitions in `packs/cyber_webapp/cyber_webapp/families/pentest.py` 
have no `technique_id` field. There is no classifier anywhere in the 
codebase that maps commands to MITRE ATT&CK technique IDs. Running 
`grep -r "technique_id"` returns zero results across the entire 
codebase, confirming the feature is completely missing.

### Proposed Solution
Build a Python classifier that reads a YAML config file containing 
command-to-technique mappings and automatically tags generated cyber 
tasks with the correct MITRE ATT&CK technique ID. Update the ontology 
to include a `technique_id` attribute and wire the classifier into the 
task generation pipeline.

### Implementation Plan
Using UMPIRE framework (adapted):

**Understand:** 
When the AI agent generates cyber tasks (e.g. running "nmap" or 
"sqlmap"), those tasks are stored as plain strings with no security 
taxonomy attached. The MITRE ATT&CK framework provides standardized 
technique IDs (e.g. T1046 for network scanning) that would make 
these tasks comparable to industry benchmarks. The feature is 
completely absent from the codebase.

**Match:** 
The `ontology.py` file already defines structured attributes for 
cyber tasks using `AttrSpec` and `AttrType`. This is the same 
pattern we will follow to add a `technique_id` field. The 
`families/pentest.py` file is where individual task families are 
defined and is where MITRE mappings will be added.

**Plan:**
1. Create `packs/cyber_webapp/cyber_webapp/technique_classifier.py` 
   as a new Python classifier that maps command patterns to MITRE 
   ATT&CK technique IDs using a YAML config file
2. Create `packs/cyber_webapp/cyber_webapp/mitre_techniques.yaml` 
   containing command-to-technique-ID mappings 
   (e.g. nmap to T1046, sqlmap to T1190)
3. Update `ontology.py` to add a `technique_id` attribute to the 
   relevant node kinds
4. Update `families/pentest.py` to call the classifier and tag 
   generated tasks with the appropriate technique ID
5. Add unit tests in `packs/cyber_webapp/tests/` to verify correct 
   tagging

**Implement:** 
https://github.com/mafeyisopin629-spec/updated-open-range/tree/fix-issue-88

**Review:** 
Will self-review against `CONTRIBUTING.md` and ensure commit 
messages follow the project's `feat:` prefix convention before 
opening a PR.

**Evaluate:** 
Run `pytest packs/cyber_webapp/` to confirm new tests pass. 
Manually verify that a generated nmap task returns `technique_id: 
T1046`. Confirm all 640 existing passing tests still pass.

---

## Testing Strategy

### Unit Tests

- [x] Test case 1: nmap command correctly returns T1046 (Network Service Discovery)
- [x] Test case 2: sqlmap command correctly returns T1190 (Exploit Public-Facing Application)
- [x] Test case 3: ssh command correctly returns T1078 (Valid Accounts)
- [x] Test case 4: unknown command correctly returns None when no match found
- [x] Test case 5: hydra command correctly returns T1110 (Brute Force)

### Integration Tests

- [x] Integration scenario 1: classify_technique reads mitre_techniques.yaml successfully
- [x] Integration scenario 2: full test suite of 640 existing tests still pass with no regressions

### Manual Testing

Ran `python -m uv run pytest packs/cyber_webapp/tests/test_technique_classifier.py -v`
in the local Windows environment. All 5 tests passed in 0.48s. Also ran the full
project test suite confirming no regressions were introduced by the new files.

---

## Implementation Notes

### Week 3 Progress
**What I built:**
- Created `mitre_techniques.yaml` mapping 5 MITRE ATT&CK techniques
  to command keywords (T1046, T1190, T1059, T1078, T1110)
- Created `technique_classifier.py` with two functions:
  `load_techniques()` and `classify_technique()`
- Updated `ontology.py` to add `technique_id` attribute to
  vulnerability node kind
- Updated `families/pentest.py` to import and call the classifier
- Created `tests/test_technique_classifier.py` with 5 passing tests

**Challenges faced:**
- `uv` command not found on Windows, resolved using `python -m uv`
- hydra keyword conflict between T1078 and T1110, resolved by
  removing hydra from T1078 keywords in the YAML file

**Code Changes:**
- **Files modified:** `ontology.py`, `families/pentest.py`
- **Files created:** `technique_classifier.py`, `mitre_techniques.yaml`,
  `tests/test_technique_classifier.py`
- **Branch:** https://github.com/mafeyisopin629-spec/updated-open-range/tree/fix-issue-88

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
