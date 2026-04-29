from __future__ import annotations

import os
import re
from io import BytesIO
from typing import List, Dict, Any

import boto3
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType


class JobIndexerToMilvus:
    def __init__(
        self,
        milvus_uri: str,
        collection_name: str,
        model_name: str = "BAAI/bge-m3",
    ):
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.model_name = model_name

        self.model = SentenceTransformer(model_name)
        self.milvus = MilvusClient(uri=milvus_uri)

    @staticmethod
    def clean_text(text: Any) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    @staticmethod
    def parse_exp_range(exp_text: Any) -> tuple[float, float]:
        if pd.isna(exp_text):
            return 0.0, 999.0

        text = str(exp_text).strip().lower()

        m = re.search(r"(\d+)\s*-\s*(\d+)", text)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return min(a, b), max(a, b)

        m = re.search(r"(\d+)\s*\+", text)
        if m:
            a = float(m.group(1))
            return a, 999.0

        m = re.search(r"(\d+)", text)
        if m:
            a = float(m.group(1))
            return a, a

        return 0.0, 999.0

    def build_mota(self, df: pd.DataFrame) -> pd.Series:
        return (
            df["job_summary"].fillna("").astype(str) + " "
            + df["main_responsibilities"].fillna("").astype(str) + " "
            + df["requirements"].fillna("").astype(str)
        ).str.strip()

    def normalize_jobs_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required_cols = [
            "job_id",
            "job_title",
            "tech_stack",
            "job_summary",
            "main_responsibilities",
            "requirements",
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột bắt buộc: {col}")

        optional_defaults = {
            "company_name": "",
            "location_city": "",
            "location_district": "",
            "experience_required": "",
        }
        for col, default_value in optional_defaults.items():
            if col not in df.columns:
                df[col] = default_value

        df["job_title_clean"] = df["job_title"].apply(self.clean_text)
        df["tech_stack_clean"] = df["tech_stack"].apply(self.clean_text)

        df["mota"] = self.build_mota(df)
        df["mota_clean"] = df["mota"].apply(self.clean_text)

        exp_ranges = df["experience_required"].apply(self.parse_exp_range)
        df["exp_min"] = exp_ranges.apply(lambda x: float(x[0]))
        df["exp_max"] = exp_ranges.apply(lambda x: float(x[1]))

        return df

    @staticmethod
    def load_jobs_file_from_minio(
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        object_name: str,
    ) -> pd.DataFrame:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        obj = s3.get_object(Bucket=bucket, Key=object_name)
        data = obj["Body"].read()

        return pd.read_json(BytesIO(data))

    def ensure_collection(self, vector_dim: int) -> None:
        collections = self.milvus.list_collections()
        if self.collection_name in collections:
            return

        schema = self.milvus.create_schema(auto_id=False, enable_dynamic_field=False)

        schema.add_field(field_name="job_id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="job_title", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="company_name", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="location_city", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="location_district", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="experience_required", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="exp_min", datatype=DataType.FLOAT)
        schema.add_field(field_name="exp_max", datatype=DataType.FLOAT)

        schema.add_field(field_name="title_vec", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field(field_name="tech_vec", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field(field_name="mota_vec", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

        index_params = self.milvus.prepare_index_params()
        index_params.add_index(field_name="title_vec", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="tech_vec", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="mota_vec", index_type="AUTOINDEX", metric_type="COSINE")

        self.milvus.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def embed_columns(self, df: pd.DataFrame) -> tuple[List[List[float]], List[List[float]], List[List[float]]]:
        title_vecs = self.model.encode(
            df["job_title_clean"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        tech_vecs = self.model.encode(
            df["tech_stack_clean"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        mota_vecs = self.model.encode(
            df["mota_clean"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return title_vecs.tolist(), tech_vecs.tolist(), mota_vecs.tolist()

    def build_records(
        self,
        df: pd.DataFrame,
        title_vecs: List[List[float]],
        tech_vecs: List[List[float]],
        mota_vecs: List[List[float]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        for idx, row in enumerate(df.itertuples(index=False)):
            records.append({
                "job_id": int(row.job_id),
                "job_title": str(getattr(row, "job_title", "") or ""),
                "company_name": str(getattr(row, "company_name", "") or ""),
                "location_city": str(getattr(row, "location_city", "") or ""),
                "location_district": str(getattr(row, "location_district", "") or ""),
                "experience_required": str(getattr(row, "experience_required", "") or ""),
                "exp_min": float(getattr(row, "exp_min", 0.0)),
                "exp_max": float(getattr(row, "exp_max", 999.0)),
                "title_vec": title_vecs[idx],
                "tech_vec": tech_vecs[idx],
                "mota_vec": mota_vecs[idx],
            })

        return records

    def delete_existing_jobs(self, job_ids: List[int]) -> None:
        if not job_ids:
            return

        batch_size = 500
        for i in range(0, len(job_ids), batch_size):
            batch = job_ids[i:i + batch_size]
            expr = f"job_id in [{','.join(map(str, batch))}]"
            self.milvus.delete(
                collection_name=self.collection_name,
                filter=expr,
            )

    def index_jobs_df(self, df_jobs: pd.DataFrame) -> Dict[str, Any]:
        df_jobs = self.normalize_jobs_df(df_jobs)

        title_vecs, tech_vecs, mota_vecs = self.embed_columns(df_jobs)
        vector_dim = len(title_vecs[0]) if title_vecs else 0
        if vector_dim <= 0:
            raise ValueError("Không tạo được embedding")

        self.ensure_collection(vector_dim=vector_dim)

        job_ids = df_jobs["job_id"].astype(int).tolist()
        self.delete_existing_jobs(job_ids)

        records = self.build_records(df_jobs, title_vecs, tech_vecs, mota_vecs)

        self.milvus.insert(
            collection_name=self.collection_name,
            data=records,
        )

        return {
            "collection_name": self.collection_name,
            "num_jobs": len(records),
            "vector_dim": vector_dim,
        }

    def index_jobs_from_minio(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        object_name: str,
    ) -> Dict[str, Any]:
        df_jobs = self.load_jobs_file_from_minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            object_name=object_name,
        )
        return self.index_jobs_df(df_jobs)


if __name__ == "__main__":
    MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
    COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "jobs_multi_vector")

    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://100.110.31.4:19000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio@123")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "crawl-results")
    MINIO_OBJECT_NAME = os.getenv("MINIO_OBJECT_NAME", "admin_thanh_topcv_stage3_full_job_details_2026-04-10/851486bdhs124.json")

    indexer = JobIndexerToMilvus(
        milvus_uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
    )

    result = indexer.index_jobs_from_minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket=MINIO_BUCKET,
        object_name=MINIO_OBJECT_NAME,
    )
    print(result)