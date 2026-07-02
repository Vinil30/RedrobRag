# Context-Aware Candidate Ranking Pipeline

This project implements a multi-stage AI ranking system for matching candidates to a job description. The central idea is simple but important:

> Candidate ranking should not be treated as keyword matching, and it should not stop at a single vector search over the JD.

The system converts the job description into a structured hiring-intent model, retrieves candidates through multi-statement semantic search, reranks them with cross-encoder context scoring, applies negative-constraint penalty scoring, and finally produces a top-100 submission with profile-based reasoning.

The design is intentionally built around recruiter behavior:

- Recruiters look for positive fit.
- Recruiters also eliminate candidates based on mismatch, weak evidence, missing context, or explicit negative traits.
- A candidate who mentions the right words is not always the right candidate.
- A candidate who strongly matches multiple intent dimensions is usually more reliable than one who matches a single phrase.

This pipeline is designed to capture all of that.

---

## Highlights

Each point is explained in detail later in this document.

| Area | What This System Does |
| --- | --- |
| One structured LLM call | A single LLM call converts the JD into HyDE candidate profiles, hard constraints, and bucket weights. |
| HyDE instead of raw JD search | The JD is transformed into hypothetical candidate profiles, aligning the JD point of view with the candidate profile point of view. |
| Four-bucket intent model | The system models ideal candidates, strong candidates, minimum-fit candidates, and undesirable candidates. |
| Negative scoring | The system explicitly models what the JD does not want and uses that as a penalty signal. |
| Multi-statement retrieval | Each generated candidate-intent statement is searched separately, avoiding dependence on one generic query. |
| Repeated-match reward | Candidates are rewarded for matching multiple strong statements, not just one lucky vector hit. |
| Metadata guardrails | Structured constraints from the JD are translated into Qdrant metadata filters where safe. |
| Cross-encoder reranking | Shortlisted candidates are reranked with pairwise context scoring, not only embedding similarity. |
| Honeypot resistance | Candidates who mention keywords without meaningful context are less likely to survive deep reranking. |
| Weighted final score | Positive, weak-positive, and negative buckets are combined with JD-specific weights. |
| Explainable output | Final results include rank, normalized score, and human-readable reasoning from candidate profile signals. |
| Efficiency controls | Retrieval shortlist, Stage 2 cap, TinyBERT cross-encoder, batching, and top-100 output keep the system practical. |

---

## High-Level Architecture

```mermaid
flowchart TD
    A["Job Description"] --> B["JD Extraction"]
    B --> C["Single Structured LLM Call"]
    C --> D["HyDE Buckets: B1, B2, B3, B4"]
    C --> E["Metadata Constraints"]
    C --> F["Bucket Weights"]

    D --> G["Multi-Statement Vector Retrieval"]
    E --> G
    G --> H["Candidate Union + Retrieval Score"]
    H --> I["Top Retrieval Shortlist"]

    I --> J["Candidate Description Builder"]
    J --> K["Cross-Encoder Positive Scoring: B1 + B2"]
    K --> L["Stage 2 Shortlist: Top 500"]
    L --> M["Cross-Encoder Weak/Negative Scoring: B3 + B4"]
    M --> N["Final Weighted Score"]

    N --> O["top_500.csv"]
    N --> P["submission.csv: Top 100"]
```

---

## Why This Is Not Keyword Matching

Keyword matching fails because it assumes that words are enough.

Example:

```text
JD needs: production ranking systems, embeddings, retrieval, evaluation

Candidate A: "Built production embedding retrieval systems and evaluated ranking quality."
Candidate B: "Read about embeddings and ranking systems in tutorials."
```

A keyword system may treat both candidates as similar.

This pipeline does not.

It uses:

- HyDE candidate intent generation
- semantic retrieval over multiple candidate-profile statements
- cross-encoder pairwise understanding
- negative bucket penalty scoring
- profile-derived reasoning

The objective is not just to find candidates who sound relevant. The objective is to rank candidates who are contextually convincing.

---

## Why This Is Not Basic Vector Search

A common approach is:

```text
embed(JD) -> vector_search -> top candidates
```

That is simple, but it has several weaknesses:

- A JD is not written like a candidate profile.
- JDs contain responsibilities, requirements, benefits, and company context mixed together.
- One embedding cannot represent all recruiter intent dimensions equally.
- It does not explicitly model rejection criteria.
- It often rewards candidates who match one broad semantic area.

This system instead does:

```text
JD -> HyDE candidate profiles -> many semantic searches -> candidate union -> reranking
```

The retrieval query is no longer the JD itself. The retrieval queries are hypothetical candidate profiles that represent what the best candidates would look like.

This is a much better semantic alignment:

```text
Candidate-like query  <->  Candidate profile
```

instead of:

```text
Job description  <->  Candidate profile
```

---

## The Single LLM Call

The system uses one structured LLM call in `BucketGen.generate(jd_text)`.

That call produces:

```text
Bucket1
Bucket2
Bucket3
Bucket4
hard_rejects
bucket_weights
```

After this step, the ranking pipeline is deterministic model-driven retrieval and scoring:

```text
LLM call -> vector search -> cross-encoder scoring -> weighted ranking -> CSV output
```

The LLM is not called repeatedly for every candidate. It is used once to convert the job description into a structured ranking strategy.

This is important because it makes the pipeline:

- more scalable
- more reproducible
- less expensive
- less vulnerable to candidate-by-candidate generation noise

---

## HyDE: Bringing JD POV And Candidate POV Onto One Line

HyDE means Hypothetical Document Embeddings.

Instead of directly embedding the JD, the system asks:

> What would the ideal candidate profiles for this JD look like?

This creates candidate-like statements such as:

```text
An experienced AI engineer who has shipped production retrieval systems,
worked with embeddings and ranking evaluation, and owned model deployment.
```

These statements are then embedded and searched against the candidate database.

The advantage is that the generated query looks like the thing we are searching for: a candidate profile.

That reduces the mismatch between:

```text
JD language: "responsible for building ranking pipelines"
Candidate language: "built retrieval and ranking systems in production"
```

HyDE gives the retrieval stage a better semantic bridge.

---

## Four-Bucket Candidate Intent Model

The system does not produce one generic search query. It produces four separate semantic buckets.

### Bucket 1: Ideal Candidates

These are the candidates recruiters would immediately shortlist.

They strongly match:

- core skills
- role seniority
- relevant domain context
- production ownership
- deployment experience
- evaluation experience
- platform signals
- important JD-specific responsibilities

Bucket 1 receives the strongest positive weight.

### Bucket 2: Strong Candidates

These candidates are still highly relevant, but may miss some preferred signals.

They may have:

- most required skills
- good adjacent experience
- strong but incomplete domain match
- enough evidence to be taken seriously

Bucket 2 receives a positive weight, lower than Bucket 1.

### Bucket 3: Minimum-Fit Candidates

These candidates satisfy basic expectations but are weaker.

They may:

- meet minimum requirements
- have partial skill overlap
- lack depth
- lack clear ownership
- be acceptable backups but not top choices

Bucket 3 receives the smallest positive weight.

### Bucket 4: Negative / Avoid Candidates

This is one of the most important differentiators in the system.

Bucket 4 represents candidates the recruiter would likely avoid.

Examples:

- research-only profiles when production experience is required
- tutorial-level experience
- wrong domain
- no software engineering exposure
- architecture-only background without implementation
- prompt-engineering-only experience
- missing required deployment or ownership evidence

Bucket 4 receives a negative weight.

This means the final score is not only:

```text
How much does this candidate match what we want?
```

It is also:

```text
How much does this candidate match what we explicitly do not want?
```

Most simple ranking systems ignore this. This system makes it a first-class scoring signal.

---

## Metadata Constraints As Guardrails

The LLM also extracts structured constraints from the JD where they are safe to apply.

Examples:

```text
years_experience >= 5
country == India
preferred_work_mode in ["remote", "hybrid"]
notice_period_days <= 30
```

These constraints are converted into Qdrant metadata filters where possible.

The system intentionally avoids applying unsafe semantic filters too early. Fields such as title, skills, location, and industry can be semantically messy, so the pipeline can skip those as hard metadata filters and let semantic retrieval handle them.

This keeps the pipeline balanced:

- use metadata filters when the condition is structured and reliable
- avoid over-filtering when meaning requires semantic understanding
- allow vector search and cross-encoder scoring to handle nuanced fit

---

## Stage 1: Multi-Statement Semantic Retrieval

Stage 1 retrieves candidates using Bucket 1 and Bucket 2 statements.

For each statement:

```text
embedding = BAAI/bge-small-en-v1.5(statement)
results = Qdrant.vector_search(embedding, limit=1000)
```

The system does this for every statement in the positive buckets.

Instead of relying on one search query, this creates many semantic entry points into the candidate database.

Why this matters:

- different strong candidates may match different parts of the JD
- a single query may over-focus on one dimension
- multiple statements improve semantic coverage
- repeated matches become evidence of stronger fit

Each retrieved candidate is added to a candidate union.

For each candidate, the system stores the similarity scores that caused the candidate to appear:

```text
candidate["bucket1_scores"]
candidate["bucket2_scores"]
```

This gives the pipeline not just a candidate list, but evidence of how often and how strongly each candidate matched the generated hiring intent.

---

## Retrieval Scoring: Rewarding Breadth, Not One Lucky Hit

After retrieval, candidates are scored before cross-encoder reranking.

The retrieval score uses the candidate's strongest match and their top repeated matches:

```text
retrieval_score = 0.6 * max_score + 0.4 * average_top_3_scores
```

This is important.

A candidate should not rank highly only because of one accidental semantic match.

The score rewards:

- the strongest matching signal
- consistency across multiple statements
- candidates who match several dimensions of the JD

This prevents the pipeline from over-trusting one high vector similarity score.

The current pipeline keeps the top retrieval candidates before cross-encoder scoring:

```text
retrieval candidate pool -> top 1000 -> cross-encoder scoring
```

This makes the expensive deep scoring stage practical.

---

## Candidate Description Construction

Before cross-encoder scoring, each candidate is converted into a single text description.

The description is built from:

- profile headline
- profile summary
- career history
- company names
- job titles
- industries
- experience descriptions
- skills

Conceptually:

```text
candidate_description =
headline
+ summary
+ career history
+ role descriptions
+ skills
```

This gives the cross-encoder a richer candidate representation than raw metadata alone.

The model can then inspect not only whether a skill exists, but how it appears in context.

---

## Stage 2: Cross-Encoder Positive Fit Scoring

Vector search is fast, but it is still similarity-based.

The cross-encoder is used for deeper contextual analysis.

Current model:

```text
cross-encoder/ms-marco-TinyBERT-L-2-v2
```

The cross-encoder receives pairs:

```text
(HyDE statement, candidate_description)
```

For Stage 2 positive scoring, the system scores every shortlisted candidate against Bucket 1 and Bucket 2 statements.

For each bucket:

```text
bucket_score = average(cross_encoder(statement, candidate_description))
```

Then the Stage 2 positive score is computed as:

```text
stage1_score =
bucket1_score * Bucket1_weight
+
bucket2_score * Bucket2_weight
```

This produces a context-aware positive-fit ranking.

The Stage 2 shortlist is capped at:

```text
top 500 candidates
```

This creates the intermediate file:

```text
top_500.csv
```

---

## Why Cross-Encoder Helps

A bi-encoder or vector search compares separate embeddings.

A cross-encoder jointly reads:

```text
statement + candidate_description
```

That allows it to detect relationships that embedding similarity can miss.

For example:

```text
Statement:
Candidate has production experience building retrieval and ranking systems.

Candidate:
Worked on ML tutorials and explored ranking algorithms in side projects.
```

Vector search may see overlap:

```text
ML, ranking, retrieval
```

The cross-encoder can better understand:

```text
This is not the same as production ownership.
```

This helps catch:

- keyword stuffing
- shallow experience
- honeypot candidates
- candidates who mention the right terms without enough evidence
- candidates with adjacent but insufficient experience

---

## Stage 3: Weak-Fit And Negative Penalty Scoring

After the top 500 candidates are selected, the system scores Bucket 3 and Bucket 4.

This is where the pipeline becomes penalty-aware.

Bucket 3 captures minimum-fit candidates:

```text
acceptable, but weak
```

Bucket 4 captures undesirable candidates:

```text
negative traits, rejection signals, or "not needed" profiles
```

Both are scored with the cross-encoder because negative decisions require context.

This matters because rejecting or penalizing a candidate based on shallow similarity can be dangerous. A candidate should be penalized only when their profile context actually aligns with the negative pattern.

Example:

```text
Bucket 4:
Research-only candidate without production deployment experience.

Candidate:
Published academic ML papers but has no deployed product systems.
```

The model can connect the negative profile to the candidate evidence and produce a penalty signal.

---

## Final Weighted Score

The final score combines all four buckets:

```text
final_score =
bucket1_score * Bucket1_weight
+
bucket2_score * Bucket2_weight
+
bucket3_score * Bucket3_weight
+
bucket4_score * Bucket4_weight
```

The weights come from the structured LLM output and are inferred from the JD's strictness.

Typical behavior:

```text
Bucket 1: highest positive weight
Bucket 2: lower positive weight
Bucket 3: smallest positive weight
Bucket 4: negative weight
```

This produces a ranking that balances:

- ideal fit
- strong but imperfect fit
- minimum acceptable fit
- negative/rejection signals

The final score is then normalized for CSV output.

---

## Output Files

The pipeline produces two CSV outputs.

### `top_500.csv`

This file contains the full Stage 2 ranked candidate set after cross-encoder reranking and final scoring.

Purpose:

- analysis
- debugging
- review of near-final candidates
- evidence that the pipeline does not jump directly to 100
- inspection of ranking distribution

### `submission.csv`

This file contains only the final top 100 candidates.

Columns:

```text
candidate_id
rank
score
reasoning
```

This is the final submission output.

---

## Reasoning Generation

The reasoning field is generated from candidate profile data.

It includes signals such as:

- current title
- years of experience
- industry
- company
- priority skills
- open-to-work status
- notice period
- preferred work mode
- recruiter response rate
- GitHub activity signal
- country

This makes the output more explainable.

The final CSV is not just:

```text
candidate_id, score
```

It also gives a compact reason why the candidate appears in the ranking.

Example reasoning shape:

```text
Machine Learning Engineer with 6.0 yrs in HR Tech at ExampleCorp;
skills: Python, embeddings, vector search, PyTorch;
open to work, 30-day notice, hybrid preference, India.
```

---

## Inference And Runtime Optimization

The pipeline includes several practical optimization choices.

### 1. One LLM call only

The LLM is used once to generate the structured ranking plan.

It is not called per candidate.

### 2. Retrieval before cross-encoder scoring

Cross-encoder scoring is expensive, so the system does not apply it to the full candidate database.

Instead:

```text
large candidate database
-> vector retrieval
-> top retrieval candidates
-> cross-encoder reranking
```

### 3. Candidate cap before CE

The candidate pool is pruned before cross-encoder scoring:

```text
top 1000 retrieval candidates
```

This keeps deep scoring focused.

### 4. Stage 2 cap

After positive cross-encoder scoring:

```text
top 500 candidates
```

move into final scoring and output analysis.

### 5. Final top 100 output

Only the final top 100 candidates are written to `submission.csv`.

### 6. TinyBERT cross-encoder

The system uses:

```text
cross-encoder/ms-marco-TinyBERT-L-2-v2
```

This gives the benefits of cross-encoder interaction while being faster than heavier cross-encoder models.

### 7. Batched cross-encoder inference

Candidates are scored in batches:

```text
candidates_per_batch = 6
batch_size = 64
```

This reduces repeated prediction overhead and improves inference utilization.

---

## End-To-End Process

```text
1. Extract job description text.

2. Use one structured LLM call to generate:
   - Bucket 1 ideal candidate profiles
   - Bucket 2 strong candidate profiles
   - Bucket 3 minimum-fit candidate profiles
   - Bucket 4 negative/avoid candidate profiles
   - metadata constraints
   - bucket weights

3. Build Qdrant metadata filters from structured constraints where safe.

4. For every Bucket 1 and Bucket 2 statement:
   - encode the statement
   - search Qdrant
   - collect candidate matches
   - store per-candidate retrieval scores

5. Merge candidates across all statement searches.

6. Compute retrieval score:
   retrieval_score = 0.6 * max_score + 0.4 * average_top_3_scores

7. Keep top retrieval candidates for cross-encoder reranking.

8. Build candidate descriptions from profile, summary, career history, and skills.

9. Score Bucket 1 and Bucket 2 with cross-encoder.

10. Rank candidates by positive-fit score.

11. Keep Stage 2 top 500 candidates.

12. Score Bucket 3 and Bucket 4 with cross-encoder.

13. Compute final weighted score across all buckets.

14. Write top_500.csv.

15. Write final top 100 to submission.csv.
```

---

## Scoring Summary

### Retrieval Score

```text
retrieval_score =
0.6 * max(candidate_vector_scores)
+
0.4 * average(top_3_candidate_vector_scores)
```

### Bucket Score

```text
bucket_score =
average(cross_encoder(statement, candidate_description))
```

### Positive Stage Score

```text
stage1_score =
bucket1_score * Bucket1_weight
+
bucket2_score * Bucket2_weight
```

### Final Score

```text
final_score =
bucket1_score * Bucket1_weight
+
bucket2_score * Bucket2_weight
+
bucket3_score * Bucket3_weight
+
bucket4_score * Bucket4_weight
```

Because Bucket 4 has a negative weight, matching negative profiles lowers the final score.

---

## Why This Should Rank Better

This system combines the strengths of several approaches instead of relying on one.

| Method | Strength | Weakness | How This Pipeline Uses It |
| --- | --- | --- | --- |
| Metadata filtering | Fast and precise for structured constraints | Unsafe for semantic meaning | Used only as a guardrail |
| Vector search | Fast broad semantic retrieval | Can reward shallow similarity | Used for scalable candidate discovery |
| HyDE | Aligns search query with candidate language | Needs careful generation | Used to create candidate-profile statements |
| Cross-encoder | Deep context understanding | Slower | Used after shortlist, not on full database |
| Negative scoring | Captures rejection criteria | Often ignored | Used as a penalty bucket |
| Reasoning | Improves explainability | Must be grounded | Built from candidate profile fields |

The pipeline is intentionally layered:

```text
fast broad recall first
deep contextual precision second
negative penalty scoring third
explainable final output last
```

---

## Evaluation-Focused Strengths

This README is meant to make the design clear for both human and AI evaluation. The most important qualities are:

### 1. Recruiter-intent modeling

The system does not treat the JD as a bag of words. It infers the type of candidate the recruiter likely wants.

### 2. Candidate-profile alignment

HyDE transforms the JD into candidate-like profiles, making semantic retrieval more natural and accurate.

### 3. Multi-dimensional retrieval

The system searches across many generated statements, not one generic embedding.

### 4. Repeated evidence scoring

Candidates who match multiple strong signals are rewarded more than candidates who match one phrase.

### 5. Contextual reranking

The cross-encoder reads the statement and candidate description together, allowing deeper fit assessment.

### 6. Penalty-aware ranking

The system explicitly models negative candidate profiles and subtracts score when a candidate matches them.

### 7. Practical scalability

The expensive model is used only after candidate filtering and shortlist creation.

### 8. Explainable output

Final candidates include reasoning grounded in their profile fields.

---

## Core Philosophy

The ranking system is built on this principle:

```text
The best candidate is not merely the most similar candidate.
The best candidate is the one who repeatedly matches the JD's positive intent,
does not match the JD's negative intent,
and has enough profile context to support the ranking.
```

This is why the pipeline uses:

- HyDE to model the ideal candidate
- vector search to retrieve at scale
- cross-encoder scoring to understand context
- Bucket 4 to penalize negative fit
- weighted scoring to reflect recruiter strictness
- final reasoning to make the ranking interpretable

The result is a context-aware, penalty-aware, recruiter-aligned candidate ranking system.
