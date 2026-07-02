# main.py
import os
import sys
from pathlib import Path

from utils.BucketGen import BucketGen
from utils.jd_extractor import JDExtractor
from utils.CandidateRetriever import CandidateRetriever
from utils.DescriptionLoader import DescriptionLoader
from utils.CEScoring import CEScoring
from utils.FinalScorer import FinalScorer


class PipelineOrchestrator:
    def __init__(self):
        self.jd_extractor = JDExtractor()
        self.bucket_gen = BucketGen()
        self.candidate_retriever = CandidateRetriever()
        self.description_loader = DescriptionLoader()
        self.ce_scoring = CEScoring()
        self.final_scorer = FinalScorer()

    def _retrieval_score(self, candidate):
        scores = []

        for key in ("bucket1_scores", "bucket2_scores"):
            values = candidate.get(key, [])
            if isinstance(values, dict):
                values = values.values()
            scores.extend(values)

        if not scores:
            return 0.0

        scores = sorted(scores, reverse=True)
        max_score = scores[0]
        top_scores = scores[:3]
        average_top_3 = sum(top_scores) / len(top_scores)

        return (0.6 * max_score) + (0.4 * average_top_3)

    def _keep_top_candidates(self, candidate_pool, limit=2000):
        scored_candidates = sorted(
            candidate_pool.items(),
            key=lambda item: self._retrieval_score(item[1]),
            reverse=True
        )

        return dict(scored_candidates[:limit])
    
    def run_pipeline(
        self,
        jd_file_path: str,
        candidates_file_path: str,
        collection_name: str = "Candidates",
        output_file: str = "submission.csv"
    ):
        """
        Run the complete candidate retrieval and ranking pipeline.
        
        Args:
            jd_file_path: Path to the job description file (.txt, .docx, or .pdf)
            candidates_file_path: Path to the JSONL file containing candidate data
            collection_name: Name of the Qdrant collection to search
            output_file: Path to the output CSV file
        
        Returns:
            List of ranked candidates with scores
        """
        print("=" * 60)
        print("STARTING CANDIDATE RETRIEVAL PIPELINE")
        print("=" * 60)
        
        # Step 1: Extract job description from file
        print("\n[1/6] Extracting job description...")
        try:
            jd_text = self.jd_extractor.extract(jd_file_path)
            print(f"✓ Job description extracted ({len(jd_text)} characters)")
        except Exception as e:
            print(f"✗ Failed to extract JD: {e}")
            raise
        
        # Step 2: Generate bucket queries using LLM
        print("\n[2/6] Generating HyDE bucket queries...")
        try:
            response = self.bucket_gen.generate(jd_text)
            print(f"✓ Generated queries:")
            print(f"  - Bucket1: {len(response.Bucket1)} statements")
            print(f"  - Bucket2: {len(response.Bucket2)} statements")
            print(f"  - Bucket3: {len(response.Bucket3)} statements")
            print(f"  - Bucket4: {len(response.Bucket4)} statements")
            print(f"  - Hard rejects: {len(response.hard_rejects)} conditions")
            print(f"  - Bucket weights: B1={response.bucket_weights.Bucket1}, "
                  f"B2={response.bucket_weights.Bucket2}, "
                  f"B3={response.bucket_weights.Bucket3}, "
                  f"B4={response.bucket_weights.Bucket4}")
        except Exception as e:
            print(f"✗ Failed to generate bucket queries: {e}")
            raise
        
        # Step 3: Retrieve candidates from Qdrant
        print("\n[3/6] Retrieving candidates from vector database...")
        try:
            candidate_pool = self.candidate_retriever.retrieve(
                response,
                collection_name=collection_name
            )
            print(f"✓ Retrieved {len(candidate_pool)} unique candidates from Qdrant")
        except Exception as e:
            print(f"✗ Failed to retrieve candidates: {e}")
            raise
        
        if not candidate_pool:
            print("⚠ No candidates retrieved. Pipeline stopping.")
            return []
        
        # Step 4: Load candidate descriptions
        print("\n[4/6] Loading candidate descriptions...")
        try:
            candidate_pool = self.description_loader.load(
                candidates_file_path,
                candidate_pool
            )
            # Count how many descriptions were loaded
            loaded_count = sum(
                1 for c in candidate_pool.values() 
                if c.get("description")
            )
            print(f"✓ Loaded descriptions for {loaded_count} candidates")
            
            # Check for candidates without descriptions
            missing_desc = len(candidate_pool) - loaded_count
            if missing_desc > 0:
                print(f"⚠ {missing_desc} candidates found in Qdrant but not in the JSONL file")
        except Exception as e:
            print(f"✗ Failed to load candidate descriptions: {e}")
            raise

        candidate_pool = {
            candidate_id: candidate
            for candidate_id, candidate in candidate_pool.items()
            if candidate.get("description")
        }

        before_prune = len(candidate_pool)
        candidate_pool = self._keep_top_candidates(
            candidate_pool,
            limit=1000
        )

        if before_prune > len(candidate_pool):
            print(
                f"✓ Kept top {len(candidate_pool)} candidates for cross-encoder "
                f"scoring out of {before_prune}"
            )
        
        # Step 5: Score candidates using cross-encoder
        print("\n[5/6] Scoring candidates with cross-encoder...")
        try:
            candidate_pool = self.ce_scoring.score(response, candidate_pool)
            print(f"✓ Scored {len(candidate_pool)} candidates")
        except Exception as e:
            print(f"✗ Failed to score candidates: {e}")
            raise
        
        # Step 6: Calculate final scores and generate ranking
        print("\n[6/6] Calculating final scores and ranking...")
        try:
            ranked_candidates = self.final_scorer.score(
                candidate_pool,
                response.bucket_weights,
                "top_500.csv"
            )

            top_100_candidate_pool = dict(ranked_candidates[:100])

            ranked_candidates = self.final_scorer.score(
                top_100_candidate_pool,
                response.bucket_weights,
                output_file
            )
            print(f"✓ Ranking complete. {len(ranked_candidates)} candidates ranked")
            print(f"✓ Stage 2 results saved to: top_500.csv")
            print(f"✓ Results saved to: {output_file}")
        except Exception as e:
            print(f"✗ Failed to calculate final scores: {e}")
            raise
        
        # Summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        if ranked_candidates:
            print("\nTop 5 Candidates:")
            for i, (candidate_id, candidate) in enumerate(ranked_candidates[:5], 1):
                print(f"  {i}. {candidate_id} - Score: {candidate['final_score']:.4f}")
        
        return ranked_candidates


def main():
    """Command-line interface for the pipeline."""
    # Default paths - modify these as needed
    JD_FILE =  r"C:\Users\VINIL\Documents\RedrobHackathon\data\data2\job_description.docx" # or .txt, .docx
    CANDIDATES_FILE = r"C:\Users\VINIL\Documents\RedrobHackathon\data\data2\candidates.jsonl"
    COLLECTION_NAME = "Candidates"  # Your Qdrant collection name
    OUTPUT_FILE = "submission.csv"
    
    # Check if files exist
    if not os.path.exists(JD_FILE):
        print(f"Error: Job description file '{JD_FILE}' not found.")
        print("Please specify the correct path or create the file.")
        sys.exit(1)
    
    if not os.path.exists(CANDIDATES_FILE):
        print(f"Error: Candidates file '{CANDIDATES_FILE}' not found.")
        print("Please specify the correct path or create the file.")
        sys.exit(1)
    
    # Initialize and run the pipeline
    orchestrator = PipelineOrchestrator()
    
    try:
        ranked_candidates = orchestrator.run_pipeline(
            jd_file_path=JD_FILE,
            candidates_file_path=CANDIDATES_FILE,
            collection_name=COLLECTION_NAME,
            output_file=OUTPUT_FILE
        )
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  - Total candidates processed: {len(ranked_candidates)}")
        print(f"  - Results saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
