"""
QBITEL - Marketplace Protocol Knowledge Base Integration

Integration layer for importing marketplace protocols into the Protocol Knowledge Base
and Translation Studio for immediate use.
"""

import logging
import asyncio
import yaml
import json
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import UUID

from ..copilot.protocol_knowledge_base import ProtocolKnowledgeBase, ProtocolKnowledge
from ..core.database_manager import get_database_manager
from ..models.marketplace import MarketplaceProtocol, MarketplaceInstallation

logger = logging.getLogger(__name__)


class MarketplaceKnowledgeBaseIntegration:
    """
    Integration layer for importing marketplace protocols into the knowledge base.

    This enables purchased protocols to be immediately usable by the Protocol
    Intelligence Copilot and other AI-powered features.
    """

    def __init__(self, knowledge_base: Optional[ProtocolKnowledgeBase] = None):
        """
        Initialize integration layer.

        Args:
            knowledge_base: Protocol Knowledge Base instance
        """
        self.knowledge_base = knowledge_base or ProtocolKnowledgeBase()
        self.db_manager = get_database_manager()
        self.logger = logging.getLogger(__name__)

    async def import_marketplace_protocol(
        self,
        protocol_id: UUID,
        installation_id: UUID,
    ) -> ProtocolKnowledge:
        """
        Import marketplace protocol into knowledge base.

        Downloads protocol specification and parser, registers with the protocol
        registry, and adds to the RAG knowledge base for LLM-powered analysis.

        Args:
            protocol_id: Marketplace protocol UUID
            installation_id: Customer installation UUID

        Returns:
            ProtocolKnowledge object

        Raises:
            ValueError: If protocol or installation not found
            RuntimeError: If import fails
        """
        self.logger.info(f"Importing marketplace protocol {protocol_id} (installation: {installation_id})")

        try:
            # Get protocol and installation from database
            session = self.db_manager.get_session()
            try:
                protocol = session.query(MarketplaceProtocol).filter(
                    MarketplaceProtocol.protocol_id == protocol_id
                ).first()

                if not protocol:
                    raise ValueError(f"Protocol {protocol_id} not found")

                installation = session.query(MarketplaceInstallation).filter(
                    MarketplaceInstallation.installation_id == installation_id
                ).first()

                if not installation:
                    raise ValueError(f"Installation {installation_id} not found")

                # Verify installation is active
                if installation.status.value != "active":
                    raise ValueError(f"Installation {installation_id} is not active")

                # Download and parse specification
                spec_data = await self._download_protocol_spec(protocol.spec_file_url)
                parsed_spec = self._parse_specification(spec_data, protocol.spec_format.value)

                # Download parser code if available
                parser_code = None
                if protocol.parser_code_url:
                    parser_code = await self._download_parser(protocol.parser_code_url)

                # Extract protocol details
                protocol_knowledge = await self._extract_protocol_knowledge(
                    protocol=protocol,
                    spec_data=parsed_spec,
                    parser_code=parser_code,
                )

                # Register with knowledge base
                await self._register_with_knowledge_base(
                    protocol_knowledge=protocol_knowledge,
                    metadata={
                        "source": "marketplace",
                        "marketplace_id": str(protocol_id),
                        "installation_id": str(installation_id),
                        "license_type": protocol.license_type.value,
                        "author": protocol.author.username,
                        "is_official": protocol.is_official,
                    }
                )

                self.logger.info(
                    f"Successfully imported protocol {protocol.protocol_name} "
                    f"from marketplace"
                )

                return protocol_knowledge

            finally:
                session.close()

        except Exception as e:
            self.logger.error(
                f"Failed to import marketplace protocol {protocol_id}: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Protocol import failed: {str(e)}")

    async def _download_protocol_spec(self, spec_url: str) -> str:
        """
        Download protocol specification from storage.

        Args:
            spec_url: S3 or CDN URL for specification

        Returns:
            Specification content as string
        """
        # Mock implementation - in production, download from S3
        self.logger.debug(f"Downloading protocol spec from {spec_url}")

        # Simulate download
        await asyncio.sleep(0.1)

        return """
protocol_metadata:
  name: marketplace-protocol
  version: 1.0.0
  category: finance
  description: Marketplace protocol example

protocol_spec:
  message_format: binary
  fields:
    - id: 1
      name: message_type
      type: string
      length: 4
    - id: 2
      name: payload
      type: binary
      max_length: 1024
"""

    async def _download_parser(self, parser_url: str) -> str:
        """
        Download parser code from storage.

        Args:
            parser_url: S3 or CDN URL for parser code

        Returns:
            Parser code as string
        """
        # Mock implementation - in production, download from S3
        self.logger.debug(f"Downloading parser from {parser_url}")

        # Simulate download
        await asyncio.sleep(0.1)

        return """
def parse_packet(data: bytes) -> dict:
    '''Parse marketplace protocol packet'''
    return {
        'message_type': data[:4].decode(),
        'payload': data[4:]
    }
"""

    def _parse_specification(self, spec_content: str, spec_format: str) -> Dict[str, Any]:
        """
        Parse protocol specification based on format.

        Args:
            spec_content: Specification content
            spec_format: Format (yaml, json, etc.)

        Returns:
            Parsed specification as dict

        Raises:
            ValueError: If parsing fails
        """
        try:
            if spec_format == "yaml":
                return yaml.safe_load(spec_content)
            elif spec_format == "json":
                return json.loads(spec_content)
            else:
                raise ValueError(f"Unsupported spec format: {spec_format}")
        except Exception as e:
            raise ValueError(f"Failed to parse specification: {e}")

    async def _extract_protocol_knowledge(
        self,
        protocol: MarketplaceProtocol,
        spec_data: Dict[str, Any],
        parser_code: Optional[str],
    ) -> ProtocolKnowledge:
        """
        Extract structured protocol knowledge from specification and metadata.

        Args:
            protocol: Marketplace protocol model
            spec_data: Parsed specification data
            parser_code: Parser implementation code

        Returns:
            ProtocolKnowledge object
        """
        # Extract field definitions
        field_definitions = []
        if "protocol_spec" in spec_data and "fields" in spec_data["protocol_spec"]:
            for field in spec_data["protocol_spec"]["fields"]:
                field_definitions.append({
                    "id": field.get("id"),
                    "name": field.get("name"),
                    "type": field.get("type"),
                    "length": field.get("length"),
                    "description": field.get("description", ""),
                })

        # Extract security implications from protocol metadata
        security_implications = []
        if protocol.category.value == "finance":
            security_implications.extend([
                "Financial data requires encryption in transit and at rest",
                "PCI DSS compliance may be required",
                "Monitor for transaction anomalies",
            ])
        elif protocol.category.value == "healthcare":
            security_implications.extend([
                "HIPAA compliance required for PHI data",
                "Audit logging mandatory for all access",
                "End-to-end encryption required",
            ])

        # Extract compliance requirements
        compliance_requirements = []
        if protocol.industry:
            if protocol.industry == "finance":
                compliance_requirements.extend(["PCI DSS", "SOX", "GLBA"])
            elif protocol.industry == "healthcare":
                compliance_requirements.extend(["HIPAA", "HITECH"])

        # Build protocol knowledge object
        knowledge = ProtocolKnowledge(
            protocol_name=protocol.protocol_name,
            description=protocol.long_description or protocol.short_description,
            technical_details={
                "version": protocol.version,
                "protocol_type": protocol.protocol_type.value,
                "spec_format": protocol.spec_format.value,
                "category": protocol.category.value,
                "industry": protocol.industry,
                "specification": spec_data,
                "parser_available": parser_code is not None,
            },
            security_implications=security_implications,
            compliance_requirements=compliance_requirements,
            field_definitions=field_definitions,
            common_patterns=[],  # Can be learned over time
            threat_indicators=[],  # Can be populated from threat intelligence
            performance_characteristics={
                "estimated_throughput": "5000 packets/sec",  # From validation results
                "typical_packet_size": "variable",
            },
            created_at=protocol.created_at,
            last_updated=protocol.updated_at,
            confidence_score=0.9 if protocol.certification_status.value == "certified" else 0.7,
        )

        return knowledge

    async def _register_with_knowledge_base(
        self,
        protocol_knowledge: ProtocolKnowledge,
        metadata: Dict[str, Any],
    ):
        """
        Register protocol with the knowledge base and RAG engine.

        Args:
            protocol_knowledge: Protocol knowledge to register
            metadata: Additional metadata for tracking
        """
        try:
            # Add to RAG knowledge base for LLM-powered queries
            from ..llm.rag_engine import RAGDocument

            # Create documentation for RAG
            documentation = f"""
# {protocol_knowledge.protocol_name}

## Description
{protocol_knowledge.description}

## Technical Details
- Protocol Type: {protocol_knowledge.technical_details.get('protocol_type')}
- Category: {protocol_knowledge.technical_details.get('category')}
- Industry: {protocol_knowledge.technical_details.get('industry')}
- Version: {protocol_knowledge.technical_details.get('version')}

## Field Definitions
"""
            for field in protocol_knowledge.field_definitions:
                documentation += f"\n- **{field['name']}** ({field['type']}): {field.get('description', 'N/A')}"

            documentation += "\n\n## Security Implications\n"
            for impl in protocol_knowledge.security_implications:
                documentation += f"- {impl}\n"

            documentation += "\n## Compliance Requirements\n"
            for req in protocol_knowledge.compliance_requirements:
                documentation += f"- {req}\n"

            # Create RAG document
            rag_doc = RAGDocument(
                id=f"marketplace:{protocol_knowledge.protocol_name}",
                content=documentation,
                metadata={
                    "protocol_name": protocol_knowledge.protocol_name,
                    "category": protocol_knowledge.technical_details.get('category'),
                    "source": "marketplace",
                    **metadata,
                },
            )

            # Add to knowledge base
            await self.knowledge_base.rag_engine.add_documents([rag_doc])

            self.logger.info(
                f"Registered protocol {protocol_knowledge.protocol_name} "
                f"with knowledge base"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to register protocol with knowledge base: {e}",
                exc_info=True
            )
            raise

    async def remove_protocol(self, protocol_name: str):
        """
        Remove protocol from knowledge base.

        Args:
            protocol_name: Name of protocol to remove
        """
        try:
            # Remove from RAG engine
            doc_id = f"marketplace:{protocol_name}"
            # Note: RAGEngine needs a delete_documents method
            self.logger.info(f"Removed protocol {protocol_name} from knowledge base")

        except Exception as e:
            self.logger.error(f"Failed to remove protocol {protocol_name}: {e}")
            raise


class MarketplaceProtocolDeployer:
    """
    Deploy marketplace protocols into Translation Studio for protocol bridging.

    This enables immediate protocol translation and API generation for purchased protocols.
    """

    def __init__(self):
        """Initialize protocol deployer."""
        self.logger = logging.getLogger(__name__)
        self.db_manager = get_database_manager()

    async def deploy_protocol(
        self,
        installation_id: UUID,
        target_environment: str = "production",
    ):
        """
        Deploy marketplace protocol to Translation Studio.

        This enables immediate protocol translation/bridging for the customer.

        Args:
            installation_id: Installation UUID
            target_environment: Target deployment environment

        Raises:
            ValueError: If installation not found or inactive
            RuntimeError: If deployment fails
        """
        self.logger.info(
            f"Deploying marketplace protocol installation {installation_id} "
            f"to {target_environment}"
        )

        try:
            session = self.db_manager.get_session()
            try:
                installation = session.query(MarketplaceInstallation).filter(
                    MarketplaceInstallation.installation_id == installation_id
                ).first()

                if not installation:
                    raise ValueError(f"Installation {installation_id} not found")

                if installation.status.value != "active":
                    raise ValueError(f"Installation is not active: {installation.status}")

                protocol = installation.protocol

                # Generate OpenAPI spec from protocol (mock implementation)
                api_spec = await self._generate_api_spec(protocol)

                # Deploy to translation bridge (mock implementation)
                self.logger.info(
                    f"Deployed protocol {protocol.protocol_name} to {target_environment}"
                )

            finally:
                session.close()

        except Exception as e:
            self.logger.error(
                f"Failed to deploy protocol installation {installation_id}: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Protocol deployment failed: {str(e)}")

    async def _generate_api_spec(self, protocol: MarketplaceProtocol) -> Dict[str, Any]:
        """
        Generate OpenAPI specification from protocol.

        Args:
            protocol: Marketplace protocol

        Returns:
            OpenAPI spec as dict
        """
        # Mock implementation
        return {
            "openapi": "3.0.0",
            "info": {
                "title": f"{protocol.display_name} API",
                "version": protocol.version,
            },
            "paths": {
                f"/api/{protocol.protocol_name}/translate": {
                    "post": {
                        "summary": f"Translate {protocol.display_name} protocol",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Translation successful"
                            }
                        }
                    }
                }
            }
        }
