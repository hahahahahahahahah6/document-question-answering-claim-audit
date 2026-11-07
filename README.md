# Documentation claim auditor

An offline, deterministic Python 3.9+ utility checking documentation against
static call evidence for answering questions about a document image or a scanned
page. Runtime uses only the standard library, with no network, telemetry, target
code execution, or file writes. Results go to stdout and diagnostics to stderr.

## Usage

```sh
python3 -m pip install .
document-question-answering-claim-audit --version
document-question-answering-claim-audit audit /path/to/repo --json
python3 -m document_question_answering_claim_audit audit --ignore docs/drafts/ /path/to/repo
python3 -m unittest discover
```

The repository defaults to the current directory. Each ignore option accepts one
relative path prefix and can be repeated. Ignored paths supply neither claims nor
evidence. The default failure threshold is high; choose medium to fail on test
fixtures too. Exit codes are 0 for no findings at the threshold, 2 for findings
at the threshold, and 1 for usage or IO errors. Version output is 0.1.0.

JSON output is a list of findings with rule, severity, path, phrase, verdict, and
message fields. Each finding has verdict FAIL; an empty list means no findings.
The verbose option reports scope and individual file decisions on stderr.

```text
documented-document-question-answering-claim-false HIGH
(documented positive claims confined to tests have MEDIUM severity)
documented-no-document-question-answering-claim-false HIGH
```

## Matching and evidence

Root README.md and Markdown files under docs are shipped prose. Files under a
tests directory can supply test-only claims. Backtick fenced code is excluded;
inline code counts as prose. Exact case-insensitive substring matching removes
denial spans first. Longest-first matching makes each occurrence match once
under its longest phrase. There is no stemming or hyphen normalization.
A positive claim without evidence is a finding; a denial with evidence is a
finding. No claims passes; a denial without sources also passes.

Every Python module is checked independently, including tests. All parts of a
route must occur in one module. Imports and aliases resolve structurally, and
local function or class definitions shadow route names throughout that module.
Git metadata, bytecode caches, conventional environment directories, directories
containing pyvenv.cfg, and symbolic links are skipped. Unreadable scoped text
causes an IO error. Invalid Python contributes no evidence.

The supported construction and call combinations are:

```text
transformers.pipeline("document-question-answering") + instance call
transformers.pipeline(task="document-question-answering") + instance call
DonutProcessor construction + VisionEncoderDecoderModel construction + model.generate()
LayoutLMForQuestionAnswering construction + instance call
LayoutLMv2ForQuestionAnswering construction + instance call
LayoutLMv3ForQuestionAnswering construction + instance call
```

Dedicated constructors also accept from_pretrained forms. Assigned and chained
instances count. Generic method calls on any base and imported bare calls count
when their names appear in the vocabulary module.

## Differentiation

```text
question-answering-claim-audit
```

Checks answering over plain text. This utility checks answering questions about
a document image or a scanned page.

```text
ocr-claim-audit
```

Checks reading text out of a document. This utility checks answering questions
about it.

```text
visual-question-answering-claim-audit
```

Checks answering about general images. This utility checks answering about a
document.

## Limitations

This is static pattern matching, not semantic judgement. Exact phrases miss
restatements. Detected calls cannot prove execution, correct inputs, or answer
quality. Extraction alone, general image tasks, table tasks, and plain text tasks
do not establish the checked capability. Construction alone is insufficient.
Dynamic imports, control flow, runtime mutation, and intermodule data flow are
outside the analysis. Simple assignment aliases are recognized.

The following pipeline task is deliberately excluded because it answers over
text already given:

```python
pipeline("question-answering")
```

A generic method name on an unrelated object counts as evidence because the
name is shared. This is an intentional naming heuristic:

```python
unrelated.document_qa(data)
```

These phrases can also describe a quality-assurance process, a FAQ page, or
asking a human team. They still match; the utility cannot determine intent:

```text
document qa review process
answer questions about documents in our help centre
ask questions about documents by emailing support
```

## Distribution

Source archives include the test suite and its discovery marker. Build with:

```sh
python3 -m build
```
