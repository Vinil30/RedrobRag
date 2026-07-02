from sentence_transformers import CrossEncoder


class CEScoring:

    def __init__(self):

        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        )
        self.stage2_limit = 500
        self.candidates_per_batch = 6

    def _score_candidate_batch(
        self,
        candidates,
        buckets
    ):

        pairs = []
        candidate_ranges = []

        for candidate in candidates:
            description = candidate["description"]
            bucket_ranges = []

            for bucket_name, statements in buckets:
                start_index = len(pairs)

                pairs.extend(
                    (statement, description)
                    for statement in statements
                )

                end_index = len(pairs)
                bucket_ranges.append(
                    (bucket_name, statements, start_index, end_index)
                )

            candidate_ranges.append(
                (candidate, bucket_ranges)
            )

        scores = self.cross_encoder.predict(
            pairs,
            batch_size=64
        )

        for candidate, bucket_ranges in candidate_ranges:

            for bucket_name, statements, start_index, end_index in bucket_ranges:

                bucket_scores = scores[start_index:end_index]

                statement_scores = {}

                for statement, score in zip(statements, bucket_scores):

                    statement_scores[statement] = float(score)

                candidate[f"{bucket_name}_scores"] = statement_scores

                candidate[f"{bucket_name}_score"] = float(
                    sum(bucket_scores) / len(bucket_scores)
                )

    def _chunks(self, values):
        for start in range(0, len(values), self.candidates_per_batch):
            yield values[start:start + self.candidates_per_batch]

    def score(
        self,
        response,
        candidate_pool
    ):

        stage1_buckets = [
            ("bucket1", response.Bucket1),
            ("bucket2", response.Bucket2),
        ]

        stage2_buckets = [
            ("bucket3", response.Bucket3),
            ("bucket4", response.Bucket4),
        ]

        total_candidates = len(candidate_pool)

        print(
            f"  Stage 1 CE: scoring bucket1/bucket2 for {total_candidates} candidates"
        )

        candidates = list(candidate_pool.values())

        scored_count = 0

        for candidate_batch in self._chunks(candidates):

            self._score_candidate_batch(
                candidate_batch,
                stage1_buckets
            )

            for candidate in candidate_batch:
                candidate["_stage1_score"] = (
                    candidate["bucket1_score"] * response.bucket_weights.Bucket1
                    +
                    candidate["bucket2_score"] * response.bucket_weights.Bucket2
                )

            scored_count += len(candidate_batch)

            if scored_count % 10 == 0 or scored_count == total_candidates:
                print(
                    f"  Stage 1 CE scored {scored_count}/{total_candidates} candidates"
                )

        ranked_candidates = sorted(
            candidate_pool.items(),
            key=lambda item: item[1]["_stage1_score"],
            reverse=True
        )

        stage2_candidates = dict(
            ranked_candidates[:self.stage2_limit]
        )

        print(
            f"  Stage 2 CE: scoring bucket3/bucket4 for "
            f"{len(stage2_candidates)} candidates"
        )

        total_stage2_candidates = len(stage2_candidates)

        stage2_candidate_values = list(stage2_candidates.values())

        scored_count = 0

        for candidate_batch in self._chunks(stage2_candidate_values):

            self._score_candidate_batch(
                candidate_batch,
                stage2_buckets
            )

            scored_count += len(candidate_batch)

            if scored_count % 10 == 0 or scored_count == total_stage2_candidates:
                print(
                    f"  Stage 2 CE scored {scored_count}/{total_stage2_candidates} candidates"
                )

        return stage2_candidates
