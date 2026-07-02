from .build_metadata_filter import MetadataFilter
from .BucketSearch import BucketSearch

def func():
    pass

class CandidateRetriever:

    def __init__(self):      

        self.metadata_filter = MetadataFilter()
        self.bucket_search = BucketSearch()

    def retrieve(self, response, collection_name):

        candidate_pool = {}

        filter_object = self.metadata_filter.filter_builder(
            response.hard_rejects
        )

        buckets = [
            ("bucket1", response.Bucket1),
            ("bucket2", response.Bucket2)
        ]

        for bucket_name, statements in buckets:

            candidate_pool = self.bucket_search.search_bucket(
                bucket=statements,
                metadata_filter=filter_object,
                collection_name=collection_name,
                bucket_name=bucket_name,
                candidate_pool=candidate_pool
            )

        return candidate_pool
