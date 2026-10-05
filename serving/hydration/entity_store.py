"""
serving/hydration/entity_store.py
Point-in-Time Cardholder & Device Profile Store.

Enforces strict causal isolation: history queries for transaction at timestamp T
retrieve ONLY prior events where timestamp < T.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TransactionHistoryEntry:
    """Historical transaction entry."""
    transaction_id: str | int
    timestamp: float
    amount: float


@dataclass
class EntityProfile:
    """Cardholder and device profile attributes."""
    card_id: str
    card1: float
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    id_attributes: Dict[str, Any] = field(default_factory=dict)


class EntityProfileStore:
    """
    In-memory Point-in-Time Entity Profile and History Store.
    Provides thread-safe profile lookups and causal historical state.
    """

    def __init__(self):
        self._profiles: Dict[str, EntityProfile] = {}
        self._history: Dict[str, List[TransactionHistoryEntry]] = {}

    def register_profile(self, profile: EntityProfile) -> None:
        """Register or update a cardholder entity profile."""
        self._profiles[str(profile.card_id)] = profile
        if str(profile.card_id) not in self._history:
            self._history[str(profile.card_id)] = []

    def get_profile(self, card_id: str) -> Optional[EntityProfile]:
        """Lookup entity profile by card_id."""
        return self._profiles.get(str(card_id))

    def get_causal_history(
        self,
        card_id: str,
        current_dt: float,
    ) -> List[TransactionHistoryEntry]:
        """
        Retrieve transaction history strictly prior to current_dt.
        
        INVARIANT: Every returned transaction satisfies timestamp < current_dt.
        Zero future lookahead is mathematically guaranteed.
        """
        key = str(card_id)
        if key not in self._history:
            return []

        causal_records = []
        for entry in self._history[key]:
            if entry.timestamp < current_dt:
                causal_records.append(entry)
            else:
                # Any record with timestamp >= current_dt is in the future relative to current_dt
                continue

        # Sort chronologically
        causal_records.sort(key=lambda x: x.timestamp)
        return causal_records

    def record_transaction(
        self,
        card_id: str,
        timestamp: float,
        amount: float,
        transaction_id: str | int,
    ) -> None:
        """
        Record a newly completed transaction into the entity history.
        Must be called strictly AFTER scoring to preserve causality during inference.
        """
        key = str(card_id)
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(
            TransactionHistoryEntry(
                transaction_id=transaction_id,
                timestamp=timestamp,
                amount=amount,
            )
        )

    def seed_mock_profiles(self) -> None:
        """Seed representative profiles matching canonical IEEE-CIS categories."""
        p1 = EntityProfile(
            card_id="CARD-13926",
            card1=13926.0,
            card2=float("nan"),
            card3=150.0,
            card4="discover",
            card5=142.0,
            card6="credit",
            addr1=315.0,
            addr2=87.0,
            dist1=19.0,
            P_emaildomain=None,
            R_emaildomain=None,
            DeviceType="desktop",
            DeviceInfo="Windows",
            id_attributes={"id_01": 0.0, "id_02": 70787.0},
        )
        p2 = EntityProfile(
            card_id="CARD-3544193",
            card1=2616.0,
            card2=327.0,
            card3=150.0,
            card4="discover",
            card5=102.0,
            card6="credit",
            addr1=330.0,
            addr2=87.0,
            dist1=12.0,
            P_emaildomain="gmail.com",
            R_emaildomain="gmail.com",
            DeviceType="mobile",
            DeviceInfo="iOS Device",
            id_attributes={"id_01": -5.0, "id_02": 100000.0},
        )
        self.register_profile(p1)
        self.register_profile(p2)
