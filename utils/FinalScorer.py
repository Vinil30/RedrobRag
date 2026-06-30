import csv


class FinalScorer:

    def __init__(self):
        pass

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

                positives = sorted(
                    candidate["bucket1_scores"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:2]

                secondary = sorted(
                    candidate["bucket2_scores"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:1]

                reasoning_parts = []

                reasoning_parts.extend(
                    statement
                    for statement, _ in positives
                )

                reasoning_parts.extend(
                    statement
                    for statement, _ in secondary
                )

                reasoning = "; ".join(reasoning_parts)

                writer.writerow(
                    [
                        candidate_id,
                        rank,
                        round(candidate["final_score"], 4),
                        reasoning
                    ]
                )

        return ranked_candidates