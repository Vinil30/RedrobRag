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
            temperature=0.9
        ).with_structured_output(StructuredOutput)
        self.schema_summary = """
profile:
- years_of_experience
- current_title
- current_company
- current_company_size
- current_industry
- location
- country

career_history:
- company
- title
- duration_months
- industry
- description

education:
- institution
- degree
- field_of_study
- tier

skills:
- name
- proficiency
- endorsements
- duration_months

redrob_signals:
- open_to_work_flag
- notice_period_days
- preferred_work_mode
- willing_to_relocate
- github_activity_score
- recruiter_response_rate
- interview_completion_rate
- offer_acceptance_rate
- expected_salary_range_inr_lpa
- profile_completeness_score
- saved_by_recruiters_30d
- search_appearance_30d
"""
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an expert AI Talent Intelligence and Semantic Retrieval System.

Your task is NOT to summarize the Job Description.

Your task is to infer recruiter intent and generate semantic retrieval
queries using the HyDE (Hypothetical Document Embedding) technique.

The generated output will power an AI hiring pipeline:

1. Metadata Filtering
2. Semantic Vector Retrieval
3. Candidate Union
4. Cross Encoder Re-ranking
5. Final Ranking

AVAILABLE CANDIDATE SCHEMA
The generated personas MUST ONLY use information that
can be represented by the candidate schema.

Use the schema fields to infer:

• profile
• career_history
• education
• skills
• redrob_signals

Do NOT invent fields outside the schema.

Descriptions should naturally correspond to values
that may exist inside these schema fields.

====================================================
PART 1 : HARD REJECTS
====================================================

Extract ONLY explicit or strongly implied hard rejection conditions.

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
    "lhs":"years_of_experience",
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
    "rhs":["Remote","Hybrid"]
}}

{{
    "lhs":"current_industry",
    "operator":"!=",
    "rhs":"Healthcare"
}}

Rules

• Use ONLY metadata fields.
• Never invent impossible metadata.
• Never create unnecessary filters.
• If a condition is only preferred, DO NOT include it here.

====================================================
PART 2 : BUCKET 1
====================================================

Generate 10-15 HyDE candidate descriptions.
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

Generate 10-15 HyDE candidate descriptions.

These candidates are still strong.

Some preferred qualifications may be missing.

These should retrieve candidates that recruiters would seriously consider.

====================================================
PART 4 : BUCKET 3
====================================================

Generate 10-15 HyDE candidate descriptions.

These candidates satisfy only minimum expectations.

They are weaker candidates but still worth retrieving.

====================================================
PART 5 : BUCKET 4
====================================================

Generate 10-15 HyDE candidate descriptions describing candidates that
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

Candidate Schema

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