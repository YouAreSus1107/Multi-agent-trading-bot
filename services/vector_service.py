"""
War-Room Bot — Vector Database Service
Uses Pinecone to store and recall war-event market patterns.
"""

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    OPENAI_API_KEY, VECTOR_EMBEDDING_MODEL, VECTOR_DIMENSION, VECTOR_TOP_K,
)
from utils.logger import get_logger
from datetime import datetime, timezone
import hashlib
import json

logger = get_logger("vector_service")


class VectorService:
    """
    Stores and recalls war-event patterns using Pinecone.
    Each event is embedded with OpenAI and stored alongside market reaction data,
    allowing the bot to 'remember' how markets reacted to similar past events.
    """

    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.index = None
        self._ensure_index()

    def _ensure_index(self):
        """Create the Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if PINECONE_INDEX_NAME not in existing_indexes:
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=VECTOR_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info(f"Created Pinecone index: {PINECONE_INDEX_NAME}")

            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            logger.error(f"Pinecone init failed: {e}")

    def _embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a text string."""
        try:
            response = self.openai.embeddings.create(
                model=VECTOR_EMBEDDING_MODEL,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return [0.0] * VECTOR_DIMENSION

    def store_event(
        self,
        event_text: str,
        market_reaction: dict,
        sector: str = "general",
        event_date: str = None,
    ):
        """
        Store a war event and its market reaction for future recall.

        Args:
            event_text: Description of the geopolitical event.
            market_reaction: Dict with ticker movements, e.g.:
                {"XLE": +3.2, "GLD": +1.1, "SPY": -1.5, "regime": "risk-off"}
            sector: Primary sector affected.
            event_date: ISO date string (defaults to now).
        """
        if not self.index:
            logger.error("Pinecone index not available")
            return

        try:
            embedding = self._embed_text(event_text)

            # Generate a deterministic ID from the event text
            event_id = hashlib.md5(event_text.encode()).hexdigest()

            metadata = {
                "event_text": event_text[:500],  # Pinecone metadata limit
                "sector": sector,
                "market_reaction": json.dumps(market_reaction),
                "date": event_date or datetime.now(timezone.utc).isoformat(),
            }

            self.index.upsert(vectors=[(event_id, embedding, metadata)])

            logger.info(
                f"Stored event: {event_text[:80]}...",
                extra={
                    "agent": "vector_service",
                    "action": "store",
                    "data": {"sector": sector},
                },
            )
        except Exception as e:
            logger.error(f"Event store failed: {e}")

    def query_similar_events(
        self, event_text: str, top_k: int = VECTOR_TOP_K
    ) -> list[dict]:
        """
        Find similar historical events and their market reactions.

        Args:
            event_text: Current event description to match against.
            top_k: Number of similar events to return.

        Returns:
            List of dicts:
            [
                {
                    "event_text": str,
                    "sector": str,
                    "market_reaction": dict,
                    "date": str,
                    "similarity_score": float,
                }
            ]
        """
        if not self.index:
            logger.error("Pinecone index not available")
            return []

        try:
            embedding = self._embed_text(event_text)

            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
            )

            similar_events = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                similar_events.append({
                    "event_text": metadata.get("event_text", ""),
                    "sector": metadata.get("sector", ""),
                    "market_reaction": json.loads(
                        metadata.get("market_reaction", "{}")
                    ),
                    "date": metadata.get("date", ""),
                    "similarity_score": round(match.get("score", 0), 4),
                })

            logger.info(
                f"Found {len(similar_events)} similar events",
                extra={
                    "agent": "vector_service",
                    "action": "query",
                    "data": {"count": len(similar_events)},
                },
            )
            return similar_events

        except Exception as e:
            logger.error(f"Event query failed: {e}")
            return []

    def seed_historical_events(self):
        """
        Pre-seed the vector DB with known historical war-event patterns.
        This gives the bot 'memory' of past market reactions.
        """
        historical_events = [
            {
                "text": "Iranian drone and missile attack on Saudi Aramco oil facilities in Abqaiq, September 2019",
                "reaction": {"XLE": 3.5, "USO": 14.7, "GLD": 1.0, "SPY": -0.5},
                "sector": "energy",
                "date": "2019-09-16",
            },
            {
                "text": "US assassination of Iranian General Qasem Soleimani via drone strike in Baghdad, January 2020",
                "reaction": {"XLE": -1.2, "GLD": 1.6, "LMT": 3.6, "SPY": -0.7},
                "sector": "defense",
                "date": "2020-01-03",
            },
            {
                "text": "Houthi attacks on Red Sea commercial shipping forcing rerouting around Cape of Good Hope, 2024",
                "reaction": {"XLE": 2.0, "USO": 4.1, "GLD": 0.8, "SPY": -0.3},
                "sector": "energy",
                "date": "2024-01-12",
            },
            {
                "text": "Iran nuclear enrichment levels reach 60%, close to weapons-grade, IAEA report 2024",
                "reaction": {"GLD": 2.3, "LMT": 1.8, "RTX": 1.5, "SPY": -1.0},
                "sector": "defense",
                "date": "2024-03-05",
            },
            {
                "text": "US deploys additional carrier strike group to Persian Gulf amid Iranian threats, 2024",
                "reaction": {"LMT": 2.8, "RTX": 2.1, "NOC": 1.9, "GLD": 1.2},
                "sector": "defense",
                "date": "2024-04-15",
            },
            {
                "text": "Iran-backed proxy launches large-scale cyber attack on US critical infrastructure",
                "reaction": {"PANW": 5.2, "CRWD": 4.8, "ZS": 3.5, "SPY": -0.8},
                "sector": "cybersecurity",
                "date": "2024-06-01",
            },
            {
                "text": "Oil tanker attacked in Strait of Hormuz, oil prices spike amid supply fears",
                "reaction": {"XLE": 4.0, "USO": 8.5, "GLD": 1.5, "SPY": -1.2},
                "sector": "energy",
                "date": "2024-07-20",
            },
            {
                "text": "US sanctions on Iranian oil exports tightened, reducing global oil supply by estimated 1M barrels/day",
                "reaction": {"XLE": 2.5, "USO": 5.0, "CVX": 3.1, "XOM": 2.8},
                "sector": "energy",
                "date": "2025-01-10",
            },
        ]

        for event in historical_events:
            self.store_event(
                event_text=event["text"],
                market_reaction=event["reaction"],
                sector=event["sector"],
                event_date=event["date"],
            )

        logger.info(f"Seeded {len(historical_events)} historical events")
