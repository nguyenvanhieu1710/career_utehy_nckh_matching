import re
from typing import Dict, Any, List

import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient


class JobRecommenderMilvus:
    def __init__(
        self,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "jobs_multi_vector",
        model_name: str = "BAAI/bge-m3",
    ):
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.model_name = model_name

        self.model = SentenceTransformer(model_name)
        self.client = MilvusClient(uri=milvus_uri)

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def experience_match_score(self, candidate_years, exp_min, under_decay=0.15, min_score=0.2):
        if pd.isna(exp_min):
            exp_min = 0

        candidate_years = float(candidate_years)
        exp_min = float(exp_min)

        if candidate_years >= exp_min:
            return 1.0

        distance = exp_min - candidate_years
        score = 1.0 - under_decay * distance
        return max(min_score, score)

    def normalize_location(self, loc):
        if loc is None:
            return None

        loc = str(loc).strip().lower()

        mapping = {
            "hồ chí minh": "tp.hcm",
            "ho chi minh": "tp.hcm",
            "tphcm": "tp.hcm",
            "tp hcm": "tp.hcm",
            "tp.hcm": "tp.hcm",
            "sài gòn": "tp.hcm",
            "sai gon": "tp.hcm",
            "hà nội": "hà nội",
            "ha noi": "hà nội",
            "đà nẵng": "đà nẵng",
            "da nang": "đà nẵng",
            "hải dương": "hải dương",
            "hai duong": "hải dương",
            "hưng yên": "hưng yên",
            "hung yen": "hưng yên",
        }

        return mapping.get(loc, loc)

    def get_location_score(self, user_location, job_location):
        location_distance_matrix = {
            "hưng yên": {
                "hưng yên": 1.00, "hải dương": 0.88, "hà nội": 0.92, "tp.hcm": 0.20, "đà nẵng": 0.45,
            },
            "hải dương": {
                "hưng yên": 0.88, "hải dương": 1.00, "hà nội": 0.90, "tp.hcm": 0.20, "đà nẵng": 0.45,
            },
            "hà nội": {
                "hưng yên": 0.92, "hải dương": 0.90, "hà nội": 1.00, "tp.hcm": 0.18, "đà nẵng": 0.50,
            },
            "tp.hcm": {
                "hưng yên": 0.20, "hải dương": 0.20, "hà nội": 0.18, "tp.hcm": 1.00, "đà nẵng": 0.55,
            },
            "đà nẵng": {
                "hưng yên": 0.45, "hải dương": 0.45, "hà nội": 0.50, "tp.hcm": 0.55, "đà nẵng": 1.00,
            }
        }

        user_location = self.normalize_location(user_location)
        job_location = self.normalize_location(job_location)

        if not user_location or not job_location:
            return 0.5

        return location_distance_matrix.get(user_location, {}).get(job_location, 0.5)

    def _search_one_field(self, vector_field: str, query_vector: List[float], top_k: int = 50):
        return self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field=vector_field,
            limit=top_k,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=[
                "job_id",
                "job_title",
                "company_name",
                "location_city",
                "location_district",
                "experience_required",
                "exp_min",
                "exp_max",
            ],
        )[0]

    def _merge_candidates(self, title_hits, tech_hits, mota_hits) -> Dict[int, Dict[str, Any]]:
        candidates: Dict[int, Dict[str, Any]] = {}

        def upsert_hits(hits, score_field: str):
            for hit in hits:
                entity = hit["entity"]
                job_id = int(entity["job_id"])
                score = float(hit["distance"])

                if job_id not in candidates:
                    candidates[job_id] = {
                        "job_id": job_id,
                        "job_title": entity.get("job_title", ""),
                        "company_name": entity.get("company_name", ""),
                        "location_city": entity.get("location_city", ""),
                        "location_district": entity.get("location_district", ""),
                        "experience_required": entity.get("experience_required", ""),
                        "exp_min": float(entity.get("exp_min", 0.0)),
                        "exp_max": float(entity.get("exp_max", 999.0)),
                        "sim_title": 0.0,
                        "sim_tech": 0.0,
                        "sim_mota": 0.0,
                    }

                candidates[job_id][score_field] = max(candidates[job_id][score_field], score)

        upsert_hits(title_hits, "sim_title")
        upsert_hits(tech_hits, "sim_tech")
        upsert_hits(mota_hits, "sim_mota")

        return candidates

    def recommend_jobs(
        self,
        cv_title,
        cv_tech,
        cv_mota,
        years_experience,
        candidate_city=None,
        top_n=10,
        search_top_k=50,
        title_weight=0.10,
        tech_weight=0.40,
        mota_weight=0.25,
        loc_weight=0.15,
        exp_weight=0.10,
        under_decay=0.15,
        min_exp_score=0.2,
    ):
        if abs(title_weight + tech_weight + mota_weight + loc_weight + exp_weight - 1.0) > 1e-9:
            raise ValueError("Tổng các weight phải bằng 1.0")

        cv_title = self.clean_text(cv_title)
        cv_tech = self.clean_text(cv_tech)
        cv_mota = self.clean_text(cv_mota)

        cv_title_embedding = self.model.encode([cv_title], normalize_embeddings=True)[0].tolist()
        cv_tech_embedding = self.model.encode([cv_tech], normalize_embeddings=True)[0].tolist()
        cv_mota_embedding = self.model.encode([cv_mota], normalize_embeddings=True)[0].tolist()

        title_hits = self._search_one_field("title_vec", cv_title_embedding, top_k=search_top_k)
        tech_hits = self._search_one_field("tech_vec", cv_tech_embedding, top_k=search_top_k)
        mota_hits = self._search_one_field("mota_vec", cv_mota_embedding, top_k=search_top_k)

        candidates = self._merge_candidates(title_hits, tech_hits, mota_hits)

        if not candidates:
            return pd.DataFrame(columns=[
                "job_id", "job_title", "company_name",
                "location_city", "location_district",
                "experience_required", "exp_min", "exp_max",
                "sim_title", "sim_tech", "sim_mota",
                "loc_score", "exp_score", "final_score"
            ])

        temp = pd.DataFrame(candidates.values())

        temp["exp_score"] = temp["exp_min"].apply(
            lambda exp_min: self.experience_match_score(
                candidate_years=years_experience,
                exp_min=exp_min,
                under_decay=under_decay,
                min_score=min_exp_score
            )
        )
        
        

        temp["loc_score"] = temp["location_city"].apply(
            lambda job_loc: self.get_location_score(candidate_city, job_loc)
        )
        temp["weighted_title_score"] = title_weight * temp["sim_title"]
        temp["weighted_tech_score"] = tech_weight * temp["sim_tech"]
        temp["weighted_mota_score"] = mota_weight * temp["sim_mota"]
        temp["weighted_loc_score"] = loc_weight * temp["loc_score"]
        temp["weighted_exp_score"] = exp_weight * temp["exp_score"]



        temp["final_score"] = (
            title_weight * temp["sim_title"] +
            tech_weight * temp["sim_tech"] +
            mota_weight * temp["sim_mota"] +
            loc_weight * temp["loc_score"] +
            exp_weight * temp["exp_score"]
        )

        ranked = temp.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)

        cols = [
            "job_id", "job_title", "company_name",
            "location_city", "location_district",
            "experience_required", "exp_min", "exp_max",
            "sim_title", "sim_tech", "sim_mota",
            "loc_score", "exp_score", "weighted_title_score", "weighted_tech_score", "weighted_mota_score", "weighted_loc_score", "weighted_exp_score", "final_score"
        ]
        return ranked[cols]


if __name__ == "__main__":
    recommender = JobRecommenderMilvus(
        milvus_uri="http://localhost:19530",
        collection_name="jobs_multi_vector",
        model_name="BAAI/bge-m3",
    )

    result = recommender.recommend_jobs(
        cv_title="tôi đã làm data engineer 2 năm kinh nghiệm",
        cv_tech="python spark sql airflow kafka etl data lake",
        cv_mota="""
        xây dựng pipeline dữ liệu batch/streaming, tối ưu etl,
        debug job, xử lý log lỗi, thiết kế schema và partition
        """,
        years_experience=2,
        candidate_city="Hà Nội",
        top_n=10,
        search_top_k=97,
    )

    print(result)