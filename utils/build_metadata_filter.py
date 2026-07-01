from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
)


class MetadataFilter:

    def __init__(self):
        self.field_aliases = {
            "candidate_id": "candidate_id",
            "years_of_experience": "years_experience",
            "years_experience": "years_experience",
            "country": "country",
            "location": "location",
            "current_title": "current_title",
            "current_company": "current_company",
            "current_industry": "industry",
            "industry": "industry",
            "degree": "highest_degree",
            "highest_degree": "highest_degree",
            "skills": "skills",
            "skill": "skills",
            "open_to_work_flag": "open_to_work",
            "open_to_work": "open_to_work",
            "notice_period_days": "notice_period_days",
            "willing_to_relocate": "willing_to_relocate",
            "preferred_work_mode": "preferred_work_mode",
            "expected_salary_min_inr_lpa": "salary_min_lpa",
            "expected_salary_max_inr_lpa": "salary_max_lpa",
            "salary_min_lpa": "salary_min_lpa",
            "salary_max_lpa": "salary_max_lpa",
            "profile_completeness_score": "profile_completeness",
            "profile_completeness": "profile_completeness",
            "recruiter_response_rate": "response_rate",
            "response_rate": "response_rate",
            "interview_completion_rate": "interview_completion_rate",
            "offer_acceptance_rate": "offer_acceptance_rate",
            "connection_count": "connection_count",
            "applications_submitted_30d": "applications_30d",
            "applications_30d": "applications_30d",
            "profile_views_received_30d": "profile_views_30d",
            "profile_views_30d": "profile_views_30d",
            "verified_email": "verified_email",
            "verified_phone": "verified_phone",
        }

        self.lowercase_fields = {
            "preferred_work_mode",
        }

        self.semantic_fields = {
            "current_title",
            "industry",
            "location",
            "skills",
        }

    def _as_list(self, value):
        if isinstance(value, list):
            return value
        return [value]

    def _normalize_field(self, field):
        return self.field_aliases.get(field)

    def _normalize_value(self, field, value):
        if field not in self.lowercase_fields:
            return value

        if isinstance(value, list):
            return [
                item.lower() if isinstance(item, str) else item
                for item in value
            ]

        if isinstance(value, str):
            return value.lower()

        return value

    def filter_builder(self, hard_rejects):

        must = []
        must_not = []

        for hr in hard_rejects:
            field = self._normalize_field(hr.lhs)

            if field is None:
                print(
                    f"Skipping unsupported metadata filter field: {hr.lhs}"
                )
                continue

            if field in self.semantic_fields:
                print(
                    f"Skipping semantic metadata filter field: {hr.lhs}"
                )
                continue

            rhs = self._normalize_value(field, hr.rhs)

            if hr.operator == "==":
                must_not.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=rhs)
                    )
                )

            elif hr.operator == "!=":
                must.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=rhs)
                    )
                )

            elif hr.operator == "in":
                must_not.append(
                    FieldCondition(
                        key=field,
                        match=MatchAny(any=self._as_list(rhs))
                    )
                )

            elif hr.operator == "not_in":
                must.append(
                    FieldCondition(
                        key=field,
                        match=MatchAny(any=self._as_list(rhs))
                    )
                )

            elif hr.operator == ">=":
                must.append(
                    FieldCondition(
                        key=field,
                        range=Range(lt=rhs)
                    )
                )

            elif hr.operator == ">":
                must.append(
                    FieldCondition(
                        key=field,
                        range=Range(lte=rhs)
                    )
                )

            elif hr.operator == "<=":
                must.append(
                    FieldCondition(
                        key=field,
                        range=Range(gt=rhs)
                    )
                )

            elif hr.operator == "<":
                must.append(
                    FieldCondition(
                        key=field,
                        range=Range(gte=rhs)
                    )
                )

        return Filter(
            must=must,
            must_not=must_not
        )
