"""
FAISS Vector Store for Maintenance Knowledge Base
Provides semantic similarity search, filtering, and real-time incremental indexing
for the technician feedback loop.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MaintenanceVectorStore:
    def __init__(self, data_path: Optional[str] = None, model_name: str = "all-MiniLM-L6-v2"):
        if data_path is None:
            # default to ../data/maintenance_records.json relative to this file
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.data_path = base_dir / "data" / "maintenance_records.json"
        else:
            self.data_path = Path(data_path)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # MiniLM dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Cosine similarity when normalized
        self.records: List[Dict[str, Any]] = []
        
        self.load_and_index()

    def _format_case_text(self, record: Dict[str, Any]) -> str:
        """Create rich composite textual representation for embedding."""
        return (
            f"Equipment: {record.get('equipment_type', '')} ({record.get('equipment_id', '')}). "
            f"Location: {record.get('location', '')}. "
            f"Reported Issue: {record.get('reported_issue', '')}. "
            f"Observed Symptoms: {record.get('symptoms', '')}. "
            f"Root Cause: {record.get('root_cause', '')}. "
            f"Fix Action: {record.get('fix_action', '')}."
        )

    def load_and_index(self):
        """Loads records from JSON and populates FAISS index."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        if not self.records:
            return

        texts = [self._format_case_text(r) for r in self.records]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        logger.info(f"Indexed {len(self.records)} records in FAISS.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        equipment_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over historical records.
        Applies optional equipment filtering while maintaining top-k count.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)

        # Retrieve a broader candidate pool if filtering is active
        fetch_k = min(self.index.ntotal, top_k * 4 if equipment_filter and equipment_filter != "All" else top_k)
        distances, indices = self.index.search(query_vector.astype(np.float32), fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.records):
                continue
            rec = self.records[idx].copy()
            if equipment_filter and equipment_filter != "All":
                if rec.get("equipment_type", "").lower() != equipment_filter.lower():
                    continue

            # Convert cosine distance to percentage similarity score (0-100%)
            similarity_pct = round(float(dist) * 100, 1)
            rec["similarity_score"] = similarity_pct
            results.append(rec)
            if len(results) >= top_k:
                break

        return results

    def add_record(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Incrementally adds a newly resolved/confirmed case to the knowledge base
        and immediately updates the active FAISS index in real time.
        """
        try:
            # Assign record ID if missing
            if "id" not in record or not record["id"]:
                record["id"] = f"MNT-{len(self.records) + 1:04d}"

            # Format and embed single record
            text = self._format_case_text(record)
            vector = self.model.encode([text], convert_to_numpy=True)
            faiss.normalize_L2(vector)

            # Add to FAISS index
            self.index.add(vector.astype(np.float32))
            self.records.append(record)

            # Persist to disk
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)

            return True, f"Successfully indexed record {record['id']} into knowledge base ({self.index.ntotal} total)."
        except Exception as e:
            return False, f"Failed to add record: {str(e)}"

    def get_all_records(self) -> List[Dict[str, Any]]:
        return self.records

    def get_equipment_types(self) -> List[str]:
        types = set(r.get("equipment_type", "") for r in self.records if r.get("equipment_type"))
        return sorted(list(types))

    def get_locations(self) -> List[str]:
        locs = set(r.get("location", "") for r in self.records if r.get("location"))
        return sorted(list(locs))
