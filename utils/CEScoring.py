from sentence_transformers import CrossEncoder


class CEScoring:

    def __init__(self):

        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def score(
        self,
        response,
        candidate_pool
    ):

        buckets = [
            ("bucket1", response.Bucket1),
            ("bucket2", response.Bucket2),
            ("bucket3", response.Bucket3),
            ("bucket4", response.Bucket4),
        ]

        total_candidates = len(candidate_pool)

        for index, candidate in enumerate(candidate_pool.values(), start=1):

            description = candidate["description"]

            for bucket_name, statements in buckets:

                pairs = [
                    (statement, description)
                    for statement in statements
                ]

                scores = self.cross_encoder.predict(pairs)

                statement_scores = {}

                for statement, score in zip(statements, scores):

                    statement_scores[statement] = float(score)

                candidate[f"{bucket_name}_scores"] = statement_scores

                candidate[f"{bucket_name}_score"] = float(sum(scores) / len(scores))

            if index % 10 == 0 or index == total_candidates:
                print(
                    f"  Cross-encoder scored {index}/{total_candidates} candidates"
                )
        return candidate_pool
