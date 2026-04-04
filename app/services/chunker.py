"""
Document-Specific Splitting Strategy
=====================================
Since our document is a structured Excel file with insurance policies,
each ROW is a self-contained semantic unit (one policy per customer).

Strategy:
- One chunk = one policy record
- Each chunk contains all fields formatted as human-readable text
- Metadata is stored separately for filtering and display
- No arbitrary token-based splitting — the document structure defines the split boundaries

This avoids cross-contamination between policies and ensures every
retrieved chunk contains complete, coherent information.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolicyChunk:
    """Represents a single policy document chunk."""
    chunk_id: str
    policy_id: str
    customer_id: str
    customer_name: str
    policy_type: str
    start_date: str
    end_date: str
    premium_amount: float
    sum_insured: float
    coverage_details: str
    exclusions: str
    status: str
    # The full text that gets embedded
    text: str = field(default="")
    # Metadata dict for ChromaDB
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.text = self._build_text()
        self.metadata = self._build_metadata()

    def _build_text(self) -> str:
        """
        Build a rich, natural-language text representation of the policy.
        This is what gets embedded — semantically descriptive so queries
        like 'motor policies that are active' retrieve relevant chunks.
        """
        return (
            f"Policy ID: {self.policy_id}\n"
            f"Customer: {self.customer_name} (ID: {self.customer_id})\n"
            f"Policy Type: {self.policy_type}\n"
            f"Status: {self.status}\n"
            f"Coverage Period: {self.start_date} to {self.end_date}\n"
            f"Premium Amount: ₦{self.premium_amount:,.2f}\n"
            f"Sum Insured: ₦{self.sum_insured:,.2f}\n"
            f"Coverage Details: {self.coverage_details}\n"
            f"Exclusions: {self.exclusions}"
        )

    def _build_metadata(self) -> dict:
        """Scalar metadata stored in ChromaDB for filtering."""
        return {
            "policy_id": self.policy_id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "policy_type": self.policy_type,
            "status": self.status,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "premium_amount": float(self.premium_amount),
            "sum_insured": float(self.sum_insured),
        }


class DocumentSpecificSplitter:
    """
    Document-specific splitter for tabular insurance policy data.

    Instead of generic token/character splitting, this splitter uses
    the inherent structure of the Excel document — each row is one policy —
    as the natural chunk boundary.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_and_split(self) -> list[PolicyChunk]:
        """Load Excel file and split into one chunk per policy row."""
        logger.info(f"Loading document from: {self.data_path}")
        df = self._load_dataframe()
        chunks = self._split_into_chunks(df)
        logger.info(f"Document-specific splitting produced {len(chunks)} chunks (one per policy row)")
        return chunks

    def _load_dataframe(self) -> pd.DataFrame:
        df = pd.read_excel(self.data_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = df.dropna(subset=["policy_id"])
        logger.info(f"Loaded {len(df)} rows from Excel ({df['policy_type'].nunique()} policy types)")
        return df

    def _split_into_chunks(self, df: pd.DataFrame) -> list[PolicyChunk]:
        chunks = []
        for idx, row in df.iterrows():
            chunk = PolicyChunk(
                chunk_id=f"chunk_{idx}_{row['policy_id']}",
                policy_id=str(row["policy_id"]),
                customer_id=str(row["customer_id"]),
                customer_name=str(row["customer_name"]),
                policy_type=str(row["policy_type"]),
                start_date=str(row["start_date"]),
                end_date=str(row["end_date"]),
                premium_amount=float(row["premium_amount"]),
                sum_insured=float(row["sum_insured"]),
                coverage_details=str(row["coverage_details"]),
                exclusions=str(row["exclusions"]),
                status=str(row["status"]),
            )
            chunks.append(chunk)
        return chunks