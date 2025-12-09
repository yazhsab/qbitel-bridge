#!/usr/bin/env python3
"""
CRONOS AI - ChromaDB to Qdrant Migration Script

Migrates all vector data from ChromaDB to Qdrant for production deployment.

Usage:
    python scripts/migrations/migrate_chromadb_to_qdrant.py --check
    python scripts/migrations/migrate_chromadb_to_qdrant.py --migrate
    python scripts/migrations/migrate_chromadb_to_qdrant.py --verify
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

@dataclass
class MigrationStats:
    """Statistics for migration."""
    collection_name: str
    total_documents: int
    migrated_documents: int
    failed_documents: int
    start_time: datetime
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    @property
    def success_rate(self) -> float:
        if self.total_documents == 0:
            return 100.0
        return (self.migrated_documents / self.total_documents) * 100


class ChromaDBToQdrantMigrator:
    """Migrates data from ChromaDB to Qdrant."""

    # Collection mapping
    COLLECTION_MAPPING = {
        "protocol_knowledge": "protocols",
        "threat_patterns": "threats",
        "compliance_rules": "compliance",
        "documentation": "documentation",
        "protocol_fields": "protocols",
        "anomaly_patterns": "threats",
        "security_events": "threats",
        "user_queries": "documentation",
        "playbooks": "documentation",
    }

    def __init__(
        self,
        chroma_path: str = "./chroma_data",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        batch_size: int = 100,
    ):
        self.chroma_path = Path(chroma_path)
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.batch_size = batch_size

        self.stats: Dict[str, MigrationStats] = {}
        self._init_clients()

    def _init_clients(self):
        """Initialize database clients."""
        # Initialize ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings

            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.chroma_path),
                anonymized_telemetry=False,
            ))
            logger.info(f"ChromaDB client initialized: {self.chroma_path}")
        except ImportError:
            logger.error("ChromaDB not installed. Run: pip install chromadb")
            self.chroma_client = None
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}")
            self.chroma_client = None

        # Initialize Qdrant
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
            self.qdrant_models = models
            logger.info(f"Qdrant client initialized: {self.qdrant_url}")
        except ImportError:
            logger.error("Qdrant client not installed. Run: pip install qdrant-client")
            self.qdrant_client = None
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}")
            self.qdrant_client = None

    def check_sources(self) -> Dict[str, Any]:
        """Check ChromaDB collections and their sizes."""
        print(f"\n{Colors.BOLD}Checking ChromaDB collections...{Colors.RESET}\n")

        if not self.chroma_client:
            return {"error": "ChromaDB client not available"}

        results = {
            "collections": {},
            "total_documents": 0,
            "total_size_estimate_mb": 0,
        }

        try:
            collections = self.chroma_client.list_collections()

            for collection in collections:
                count = collection.count()
                results["collections"][collection.name] = {
                    "document_count": count,
                    "target_collection": self.COLLECTION_MAPPING.get(
                        collection.name,
                        "documentation"
                    ),
                }
                results["total_documents"] += count

                print(f"  {collection.name}: {count} documents")

            # Estimate size (rough: 1KB per document average)
            results["total_size_estimate_mb"] = results["total_documents"] * 1 / 1024

            print(f"\n  Total: {results['total_documents']} documents")
            print(f"  Estimated size: {results['total_size_estimate_mb']:.2f} MB")

        except Exception as e:
            logger.error(f"Failed to check ChromaDB: {e}")
            results["error"] = str(e)

        return results

    def check_target(self) -> Dict[str, Any]:
        """Check Qdrant status and existing collections."""
        print(f"\n{Colors.BOLD}Checking Qdrant target...{Colors.RESET}\n")

        if not self.qdrant_client:
            return {"error": "Qdrant client not available"}

        results = {
            "connected": False,
            "collections": {},
        }

        try:
            # Check connection
            collections_response = self.qdrant_client.get_collections()
            results["connected"] = True

            for collection in collections_response.collections:
                info = self.qdrant_client.get_collection(collection.name)
                results["collections"][collection.name] = {
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status,
                }
                print(f"  {collection.name}: {info.vectors_count} vectors")

            print(f"\n  {Colors.GREEN}Qdrant connection: OK{Colors.RESET}")

        except Exception as e:
            logger.error(f"Failed to check Qdrant: {e}")
            results["error"] = str(e)
            print(f"\n  {Colors.RED}Qdrant connection: FAILED{Colors.RESET}")

        return results

    def create_qdrant_collections(self, embedding_dim: int = 384):
        """Create required Qdrant collections."""
        print(f"\n{Colors.BOLD}Creating Qdrant collections...{Colors.RESET}\n")

        target_collections = set(self.COLLECTION_MAPPING.values())

        for collection_name in target_collections:
            try:
                # Check if collection exists
                try:
                    self.qdrant_client.get_collection(collection_name)
                    print(f"  {collection_name}: already exists")
                    continue
                except Exception:
                    pass

                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=self.qdrant_models.VectorParams(
                        size=embedding_dim,
                        distance=self.qdrant_models.Distance.COSINE,
                    ),
                )
                print(f"  {Colors.GREEN}{collection_name}: created{Colors.RESET}")

            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {e}")
                print(f"  {Colors.RED}{collection_name}: failed - {e}{Colors.RESET}")

    async def migrate_collection(
        self,
        source_collection_name: str,
        target_collection_name: str,
    ) -> MigrationStats:
        """Migrate a single collection."""
        stats = MigrationStats(
            collection_name=source_collection_name,
            total_documents=0,
            migrated_documents=0,
            failed_documents=0,
            start_time=datetime.now(),
        )

        try:
            # Get source collection
            source = self.chroma_client.get_collection(source_collection_name)
            stats.total_documents = source.count()

            if stats.total_documents == 0:
                stats.end_time = datetime.now()
                return stats

            # Fetch all documents
            results = source.get(
                include=["documents", "embeddings", "metadatas"]
            )

            # Migrate in batches
            batch_points = []

            for i, (doc_id, doc, embedding, metadata) in enumerate(zip(
                results["ids"],
                results["documents"] or [""] * len(results["ids"]),
                results["embeddings"] or [[]] * len(results["ids"]),
                results["metadatas"] or [{}] * len(results["ids"]),
            )):
                try:
                    # Create Qdrant point
                    point = self.qdrant_models.PointStruct(
                        id=str(uuid4()),  # New UUID for Qdrant
                        vector=embedding if embedding else [0.0] * 384,
                        payload={
                            "content": doc,
                            "original_id": doc_id,
                            "source_collection": source_collection_name,
                            "migrated_at": datetime.now().isoformat(),
                            **(metadata or {}),
                        },
                    )
                    batch_points.append(point)

                    # Upload batch
                    if len(batch_points) >= self.batch_size:
                        self.qdrant_client.upsert(
                            collection_name=target_collection_name,
                            points=batch_points,
                        )
                        stats.migrated_documents += len(batch_points)
                        batch_points = []

                        # Progress
                        progress = (stats.migrated_documents / stats.total_documents) * 100
                        print(f"\r  {source_collection_name}: {progress:.1f}%", end="")

                except Exception as e:
                    logger.warning(f"Failed to migrate document {doc_id}: {e}")
                    stats.failed_documents += 1

            # Upload remaining batch
            if batch_points:
                self.qdrant_client.upsert(
                    collection_name=target_collection_name,
                    points=batch_points,
                )
                stats.migrated_documents += len(batch_points)

            print(f"\r  {source_collection_name}: 100.0% - {stats.migrated_documents} migrated")

        except Exception as e:
            logger.error(f"Failed to migrate collection {source_collection_name}: {e}")

        stats.end_time = datetime.now()
        return stats

    async def migrate_all(self) -> Dict[str, MigrationStats]:
        """Migrate all collections."""
        print(f"\n{Colors.BOLD}Starting migration...{Colors.RESET}\n")

        if not self.chroma_client or not self.qdrant_client:
            logger.error("Database clients not available")
            return {}

        # Create target collections
        self.create_qdrant_collections()

        # Get source collections
        collections = self.chroma_client.list_collections()

        for collection in collections:
            target = self.COLLECTION_MAPPING.get(collection.name, "documentation")

            print(f"\nMigrating: {collection.name} -> {target}")

            stats = await self.migrate_collection(collection.name, target)
            self.stats[collection.name] = stats

        return self.stats

    def verify_migration(self) -> Dict[str, Any]:
        """Verify migration was successful."""
        print(f"\n{Colors.BOLD}Verifying migration...{Colors.RESET}\n")

        results = {
            "verified": True,
            "collections": {},
        }

        for source_name, stats in self.stats.items():
            target_name = self.COLLECTION_MAPPING.get(source_name, "documentation")

            try:
                # Check Qdrant collection
                info = self.qdrant_client.get_collection(target_name)

                results["collections"][source_name] = {
                    "source_count": stats.total_documents,
                    "migrated_count": stats.migrated_documents,
                    "target_count": info.vectors_count,
                    "success_rate": stats.success_rate,
                    "duration_seconds": stats.duration_seconds,
                }

                if stats.success_rate < 99.0:
                    results["verified"] = False
                    print(f"  {Colors.RED}{source_name}: {stats.success_rate:.1f}% success{Colors.RESET}")
                else:
                    print(f"  {Colors.GREEN}{source_name}: {stats.success_rate:.1f}% success{Colors.RESET}")

            except Exception as e:
                results["verified"] = False
                results["collections"][source_name] = {"error": str(e)}
                print(f"  {Colors.RED}{source_name}: verification failed - {e}{Colors.RESET}")

        return results

    def print_summary(self):
        """Print migration summary."""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}MIGRATION SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

        total_docs = 0
        total_migrated = 0
        total_failed = 0
        total_time = 0

        for name, stats in self.stats.items():
            total_docs += stats.total_documents
            total_migrated += stats.migrated_documents
            total_failed += stats.failed_documents
            total_time += stats.duration_seconds

            status = f"{Colors.GREEN}OK{Colors.RESET}" if stats.success_rate >= 99 else f"{Colors.RED}ISSUES{Colors.RESET}"
            print(f"  {name}:")
            print(f"    Documents: {stats.migrated_documents}/{stats.total_documents}")
            print(f"    Duration: {stats.duration_seconds:.2f}s")
            print(f"    Status: {status}")
            print()

        print(f"  {Colors.BOLD}TOTALS:{Colors.RESET}")
        print(f"    Documents migrated: {total_migrated}/{total_docs}")
        print(f"    Failed: {total_failed}")
        print(f"    Total time: {total_time:.2f}s")

        if total_failed == 0:
            print(f"\n  {Colors.GREEN}{Colors.BOLD}Migration completed successfully!{Colors.RESET}")
        else:
            print(f"\n  {Colors.YELLOW}{Colors.BOLD}Migration completed with {total_failed} failures{Colors.RESET}")

        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="CRONOS AI ChromaDB to Qdrant Migration"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check source and target status"
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Run the migration"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration results"
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default="./chroma_data",
        help="ChromaDB data directory"
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for migration"
    )

    args = parser.parse_args()

    migrator = ChromaDBToQdrantMigrator(
        chroma_path=args.chroma_path,
        qdrant_url=args.qdrant_url,
        batch_size=args.batch_size,
    )

    print(f"\n{Colors.BOLD}CRONOS AI Vector Database Migration Tool{Colors.RESET}")
    print(f"ChromaDB -> Qdrant\n")

    if args.check or not any([args.migrate, args.verify]):
        migrator.check_sources()
        migrator.check_target()

    if args.migrate:
        await migrator.migrate_all()
        migrator.print_summary()

    if args.verify:
        results = migrator.verify_migration()
        if results["verified"]:
            print(f"\n{Colors.GREEN}All migrations verified successfully!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}Some migrations failed verification{Colors.RESET}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
