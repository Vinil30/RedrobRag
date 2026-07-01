import os
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class HardReject(BaseModel):
    lhs: str
    operator: str
    rhs: str | int | float | bool | list


class BucketWeights(BaseModel):
    Bucket1: float
    Bucket2: float
    Bucket3: float
    Bucket4: float


class StructuredOutput(BaseModel):
    Bucket1: list[str]
    Bucket2: list[str]
    Bucket3: list[str]
    Bucket4: list[str]

    hard_rejects: list[HardReject]

    bucket_weights: BucketWeights


class BucketGen:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1
        ).with_structured_output(StructuredOutput)
        self.schema_summary = """
Qdrant metadata fields available for hard rejects:
- candidate_id
- years_experience
- country
- location
- current_title
- current_company
- industry
- highest_degree
- skills
- open_to_work
- notice_period_days
- willing_to_relocate
- preferred_work_mode
- salary_min_lpa
- salary_max_lpa
- profile_completeness
- response_rate
- interview_completion_rate
- offer_acceptance_rate
- connection_count
- applications_30d
- profile_views_30d
- verified_email
- verified_phone

Important value formats:
- preferred_work_mode values are lowercase: remote, hybrid, onsite, flexible
- boolean values are true or false
- skills is a list of skill names
"""
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an expert AI Talent Intelligence and Semantic Retrieval System.

Your task is NOT to summarize the Job Description.

Your task is to infer recruiter intent and generate semantic retrieval
queries using the HyDE (Hypothetical Document Embedding) technique.

CRITICAL GROUNDING RULE:
Generate candidate descriptions only from evidence in the Job Description.
Do NOT generalize from the job title.
Do NOT use broad role-category assumptions.

If the JD does not mention or clearly require a concept, do not generate
candidate profiles around that concept.

Avoid generic broad terms unless the JD explicitly supports them
The generated output will power an AI hiring pipeline:

1. Metadata Filtering
2. Semantic Vector Retrieval
3. Candidate Union
4. Cross Encoder Re-ranking
5. Final Ranking

AVAILABLE QDRANT METADATA
The generated personas MUST ONLY use information that
can be represented by the candidate metadata.

Use the metadata fields to infer:

• experience
• location
• current role
• industry
• education
• skills
• platform signals

Do NOT invent fields outside the metadata list.

Descriptions should naturally correspond to values
that may exist inside these metadata fields.

====================================================
PART 1 : METADATA KEEP FILTERS
====================================================

The output field is named hard_rejects for compatibility with the
pipeline, but its contents MUST be Qdrant KEEP filters.

A KEEP filter describes candidates who should remain eligible for
semantic retrieval.

Do NOT describe rejected candidates.
Do NOT output the negative/bad-candidate condition itself.

If the JD says "reject candidates with less than 5 years experience",
the correct KEEP filter is:

{{
    "lhs":"years_experience",
    "operator":">=",
    "rhs":5
}}

The wrong filter would be:

{{
    "lhs":"years_experience",
    "operator":"<",
    "rhs":5
}}

because that keeps the rejected candidates.

Extract ONLY explicit or strongly implied eligibility conditions that can
be safely applied before vector search.

Represent every condition using

{{
    "lhs": "...",
    "operator": "...",
    "rhs": ...
}}

Allowed operators

==
!=
>
>=
<
<=
in
not_in

Examples

{{
    "lhs":"years_experience",
    "operator":">=",
    "rhs":5
}}

{{
    "lhs":"country",
    "operator":"==",
    "rhs":"India"
}}

{{
    "lhs":"preferred_work_mode",
    "operator":"in",
    "rhs":["remote","hybrid"]
}}

{{
    "lhs":"industry",
    "operator":"!=",
    "rhs":"Healthcare"
}}

Rules

• Every condition must KEEP eligible candidates, not identify rejected candidates.
• For minimum requirements, use >=.
• For maximum limits, use <=.
• Use != or not_in only when the JD explicitly excludes a metadata value,
such as "not from Healthcare".
• Do NOT use != to express desired titles, desired industries, desired
locations, or desired work modes.
• Do NOT exact-match current_title unless the JD explicitly requires one
exact title and rejects all others.
• Do NOT exact-match location unless the JD explicitly rejects all other
locations. Prefer country when the role is country-bound.
• If a disqualifier cannot be represented by available metadata, do NOT
create a filter for it. Let semantic retrieval handle it.
• Use ONLY these exact metadata field names:
candidate_id, years_experience, country, location, current_title,
current_company, industry, highest_degree, skills, open_to_work,
notice_period_days, willing_to_relocate, preferred_work_mode,
salary_min_lpa, salary_max_lpa, profile_completeness, response_rate,
interview_completion_rate, offer_acceptance_rate, connection_count,
applications_30d, profile_views_30d, verified_email, verified_phone.
• Do NOT use source schema names such as years_of_experience,
current_industry, open_to_work_flag, recruiter_response_rate,
profile_completeness_score, or expected_salary_range_inr_lpa.
• For preferred_work_mode, always use lowercase values.
• Never invent impossible metadata.
• Never create unnecessary filters.
• If a condition is only preferred, DO NOT include it here.
• When uncertain, output fewer filters.

====================================================
PART 2 : BUCKET 1
====================================================

Generate 8-12 HyDE candidate descriptions.
Each description MUST represent a DIFFERENT interpretation of the Job Description.
These represent candidates recruiters would immediately shortlist.

Every description should read like a realistic resume summary.

Avoid generating multiple candidates that differ only in wording.

Each candidate should emphasize a different combination of:

• primary technical expertise
• previous industries
• company backgrounds
• production responsibilities
• leadership level
• project focus
• system ownership
• evaluation experience
• deployment experience
• platform signals

Do NOT write keywords.

Do NOT copy the JD.

Every statement should be semantically unique.

====================================================
PART 3 : BUCKET 2
====================================================

Generate 8-12 HyDE candidate descriptions.

These candidates are still strong.

Some preferred qualifications may be missing.

These should retrieve candidates that recruiters would seriously consider.

====================================================
PART 4 : BUCKET 3
====================================================

Generate EXACTLY 5 HyDE candidate descriptions.

These candidates satisfy only minimum expectations.

They are weaker candidates but still worth retrieving.

====================================================
PART 5 : BUCKET 4
====================================================

Generate EXACTLY 5 HyDE candidate descriptions describing candidates that
recruiters would actively avoid.

Examples

• Internship-only profiles

• Research-only candidates

• No production experience

• Wrong domain

• No software engineering experience

• Architecture without implementation

• Prompt engineering only

• Tutorial-level experience

These descriptions are semantic negatives.

====================================================
PART 6 : BUCKET WEIGHTS
====================================================

Assign an importance weight to every bucket.

These weights will later be multiplied with cosine similarity scores.

Requirements

• Bucket1 MUST have the highest positive weight.

• Bucket2 must have a smaller positive weight.

• Bucket3 must have the smallest positive weight.

• Bucket4 MUST have a negative weight.

Weights should depend on recruiter strictness.

Examples

Strict hiring

Bucket1 = 1.00
Bucket2 = 0.45
Bucket3 = 0.15
Bucket4 = -0.90

Moderate hiring

Bucket1 = 1.00
Bucket2 = 0.75
Bucket3 = 0.40
Bucket4 = -0.50

Flexible hiring

Bucket1 = 1.00
Bucket2 = 0.90
Bucket3 = 0.70
Bucket4 = -0.20

Infer recruiter strictness from the Job Description.

====================================================
RULES
====================================================

1. Think like an experienced recruiter.

2. Never summarize the Job Description.

3. Infer hidden hiring intent.

4. Generate HyDE candidate profiles.

5. Every profile should resemble a real candidate.

6. Maximize semantic diversity.

7. Never repeat the same candidate with small wording changes.

8. Bucket1 represents ideal candidates.

9. Bucket2 represents good candidates.

10. Bucket3 represents acceptable candidates.

11. Bucket4 represents undesirable candidates.

12. Output ONLY the structured response.

13. Don't repeat the same aspect more than 3 times in a bucket. For example you are allowed to write a statement on experience for atmost 3 times,
 the aspect need to change from experience to any other thing that is mentioned the job description.
"""
            ),
            (
    "user",
    """
Job Description

{job_desc}

Qdrant Metadata

{schema_summary}
"""
)
        ])

        self.chain = self.prompt | self.llm

    def generate(self, job_desc):
        return self.chain.invoke(
            {
                "job_desc": job_desc,
                "schema_summary": self.schema_summary
            }
        )
