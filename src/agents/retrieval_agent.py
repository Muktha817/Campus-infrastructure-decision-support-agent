"""
Retrieval Agent: Formulates search queries, scans knowledge base,
and retrieves the most relevant historical maintenance cases with similarity scores.
"""

from typing import Dict, Any, List, Optional
from ..retrieval.vector_store import MaintenanceVectorStore

class RetrievalAgent:
    def __init__(self, vector_store: Optional[MaintenanceVectorStore] = None):
        self.vector_store = vector_store or MaintenanceVectorStore()

    def retrieve(
        self,
        reported_issue: str,
        symptoms: str = "",
        equipment_type: Optional[str] = None,
        location: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Constructs rich query and retrieves top matching historical cases.
        """
        query_parts = []
        if equipment_type and equipment_type != "All":
            query_parts.append(f"Equipment: {equipment_type}")
        if location:
            query_parts.append(f"Location: {location}")
        query_parts.append(f"Issue: {reported_issue}")
        if symptoms:
            query_parts.append(f"Symptoms: {symptoms}")

        query = ". ".join(query_parts)
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            equipment_filter=equipment_type if equipment_type != "All" else None
        )
        return results
