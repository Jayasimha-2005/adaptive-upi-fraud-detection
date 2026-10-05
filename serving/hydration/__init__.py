"""
serving/hydration package.
Provides Online Feature Hydration Adapter, Entity Profile Store, and Schemas.
"""
from serving.hydration.adapter import OnlineFeatureHydrationAdapter
from serving.hydration.entity_store import EntityProfileStore, EntityProfile, TransactionHistoryEntry
from serving.hydration.provenance import (
    classify_feature,
    get_provenance_manifest,
    TIER_DIRECT_EVENT,
    TIER_DERIVED_EVENT,
    TIER_ENTITY_PROFILE,
    TIER_HISTORICAL_AGG,
    TIER_UNRECONSTRUCTABLE,
)
from serving.hydration.schemas import (
    HydrationResult,
    StreamingTransactionPayload,
    VelocityMetrics,
)

__all__ = [
    "OnlineFeatureHydrationAdapter",
    "EntityProfileStore",
    "EntityProfile",
    "TransactionHistoryEntry",
    "HydrationResult",
    "StreamingTransactionPayload",
    "VelocityMetrics",
    "classify_feature",
    "get_provenance_manifest",
    "TIER_DIRECT_EVENT",
    "TIER_DERIVED_EVENT",
    "TIER_ENTITY_PROFILE",
    "TIER_HISTORICAL_AGG",
    "TIER_UNRECONSTRUCTABLE",
]
