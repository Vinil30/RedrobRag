from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pathlib import Path


class BucketSearch:

    def __init__(self):

        self.embedding_model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        self.local_db_path = Path(__file__).resolve().parents[1] / "RAG" / "qdrant_d_b"
        self.client = QdrantClient(
            path=str(self.local_db_path)
        )

    def _query_points(
        self,
        collection_name,
        embedding,
        metadata_filter,
        limit
    ):
        def run_query(client, name, query_filter):
            return client.query_points(
                collection_name=name,
                query=embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            ).points

        def retry_without_filter_if_empty(client, name, points):
            if points or metadata_filter is None:
                return points

            print(
                "Metadata filter returned 0 candidates for one query. "
                "Retrying without metadata filter."
            )
            return run_query(client, name, None)

        try:
            points = run_query(
                self.client,
                collection_name,
                metadata_filter
            )
            return retry_without_filter_if_empty(
                self.client,
                collection_name,
                points
            )

        except Exception:
            local_collection_name = collection_name.lower()
            if local_collection_name == collection_name:
                raise

            points = run_query(
                self.client,
                local_collection_name,
                metadata_filter
            )
            return retry_without_filter_if_empty(
                self.client,
                local_collection_name,
                points
            )

    def search_bucket(
        self,
        bucket,
        metadata_filter,
        collection_name,
        bucket_name,
        candidate_pool,
        limit=1000
    ):

        for statement in bucket:

            embedding = self.embedding_model.encode(
                statement,
                normalize_embeddings=True
            ).tolist()
 
            results = self._query_points(
                collection_name=collection_name,
                embedding=embedding,
                metadata_filter=metadata_filter,
                limit=limit
            )

            for result in results:

                candidate_id = result.id
                if result.payload and result.payload.get("candidate_id"):
                    candidate_id = result.payload["candidate_id"]

                score = result.score

                if candidate_id not in candidate_pool:

                    candidate_pool[candidate_id] = {
                        "bucket1_scores": [],
                        "bucket2_scores": []
                    }

                candidate_pool[candidate_id][
                    f"{bucket_name}_scores"
                ].append(score)

        return candidate_pool
