import csv


class FinalScorer:

    def __init__(self):
        self.priority_skills = {
            "python",
            "machine learning",
            "deep learning",
            "llms",
            "rag",
            "embeddings",
            "vector search",
            "semantic search",
            "information retrieval",
            "learning to rank",
            "recommendation systems",
            "qdrant",
            "milvus",
            "pinecone",
            "weaviate",
            "faiss",
            "opensearch",
            "elasticsearch",
            "sentence transformers",
            "fine-tuning llms",
            "lora",
            "qlora",
            "peft",
            "mlops",
            "mlflow",
            "kubeflow",
            "pytorch",
            "tensorflow",
            "nlp",
        }

    def _format_number(self, value, suffix=""):
        if value is None:
            return None

        if isinstance(value, float):
            return f"{value:.1f}{suffix}"

        return f"{value}{suffix}"

    def _select_skills(self, skills, limit=4):
        skill_names = [
            skill.get("name")
            for skill in skills
            if skill.get("name")
        ]

        priority = [
            name
            for name in skill_names
            if name.lower() in self.priority_skills
        ]

        selected = priority[:limit]

        for name in skill_names:
            if len(selected) >= limit:
                break
            if name not in selected:
                selected.append(name)

        return selected

    def _build_reasoning(self, candidate):
        raw_candidate = candidate.get("raw_candidate", {})
        profile = raw_candidate.get("profile", {})
        signals = raw_candidate.get("redrob_signals", {})
        skills = raw_candidate.get("skills", [])

        title = profile.get("current_title") or "Candidate"
        years = self._format_number(
            profile.get("years_of_experience"),
            " yrs"
        )
        industry = profile.get("current_industry")
        company = profile.get("current_company")
        country = profile.get("country")

        parts = []

        lead = title
        if years:
            lead += f" with {years}"
        if industry:
            lead += f" in {industry}"
        if company:
            lead += f" at {company}"
        parts.append(lead)

        selected_skills = self._select_skills(skills)
        if selected_skills:
            parts.append(
                "skills: " + ", ".join(selected_skills)
            )

        signal_parts = []

        if signals.get("open_to_work_flag"):
            signal_parts.append("open to work")

        notice_period = signals.get("notice_period_days")
        if notice_period is not None:
            signal_parts.append(f"{notice_period}-day notice")

        work_mode = signals.get("preferred_work_mode")
        if work_mode:
            signal_parts.append(f"{work_mode} preference")

        response_rate = signals.get("recruiter_response_rate")
        if response_rate is not None and response_rate >= 0.7:
            signal_parts.append(
                f"{response_rate:.0%} recruiter response rate"
            )

        github_score = signals.get("github_activity_score")
        if github_score is not None and github_score >= 50:
            signal_parts.append(
                f"{github_score:.0f} GitHub activity score"
            )

        if country:
            signal_parts.append(country)

        if signal_parts:
            parts.append(", ".join(signal_parts))

        return "; ".join(parts) + "."

    def _normalize_score(self, score, min_score, max_score):
        if max_score == min_score:
            return 1.0

        return (score - min_score) / (max_score - min_score)

    def score(
        self,
        candidate_pool,
        bucket_weights,
        output_file="submission.csv"
    ):

        ranked_candidates = []

        for candidate_id, candidate in candidate_pool.items():

            final_score = (
                candidate["bucket1_score"] * bucket_weights.Bucket1
                +
                candidate["bucket2_score"] * bucket_weights.Bucket2
                +
                candidate["bucket3_score"] * bucket_weights.Bucket3
                +
                candidate["bucket4_score"] * bucket_weights.Bucket4
            )

            candidate["final_score"] = float(final_score)

            ranked_candidates.append(
                (candidate_id, candidate)
            )

        ranked_candidates.sort(
            key=lambda x: x[1]["final_score"],
            reverse=True
        )

        final_scores = [
            candidate["final_score"]
            for _, candidate in ranked_candidates
        ]
        min_score = min(final_scores)
        max_score = max(final_scores)

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "candidate_id",
                    "rank",
                    "score",
                    "reasoning"
                ]
            )

            for rank, (candidate_id, candidate) in enumerate(
                ranked_candidates,
                start=1
            ):

                candidate["rank"] = rank

                reasoning = self._build_reasoning(candidate)
                normalized_score = self._normalize_score(
                    candidate["final_score"],
                    min_score,
                    max_score
                )

                writer.writerow(
                    [
                        candidate_id,
                        rank,
                        round(normalized_score, 4),
                        reasoning
                    ]
                )

        return ranked_candidates
