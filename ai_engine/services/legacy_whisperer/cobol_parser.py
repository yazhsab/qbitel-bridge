"""
QBITEL - COBOL Parser Service

Traffic pattern analysis and protocol structure inference for legacy systems.

Responsibilities:
- Parse binary/EBCDIC traffic samples
- Extract protocol field definitions
- Identify record boundaries and field types
- Detect patterns (magic numbers, fixed-length records, delimiters)
- Character encoding detection (EBCDIC, ASCII, UTF-8)
"""

import json
import logging
from typing import Dict, Any, List

from .models import ProtocolPattern, ProtocolField, ProtocolComplexity, LegacyWhispererException
from ...llm.unified_llm_service import get_llm_service, LLMRequest

logger = logging.getLogger(__name__)


class COBOLParserService:
    """
    Service for parsing legacy protocol traffic and inferring structure.

    Features:
    - Traffic pattern analysis
    - Protocol structure inference using LLM
    - Message type identification
    - Protocol characteristic determination
    """

    def __init__(self, llm_service=None):
        """
        Initialize the COBOL Parser Service.

        Args:
            llm_service: Optional LLM service instance (uses default if None)
        """
        self.llm_service = llm_service or get_llm_service()
        self.logger = logging.getLogger(__name__)

    async def analyze_traffic_patterns(self, samples: List[bytes]) -> List[ProtocolPattern]:
        """
        Analyze traffic samples to identify patterns.

        Args:
            samples: List of protocol message samples

        Returns:
            List of identified protocol patterns
        """
        patterns = []

        # Analyze message lengths
        lengths = [len(sample) for sample in samples]
        if len(set(lengths)) == 1:
            patterns.append(
                ProtocolPattern(
                    pattern_type="fixed_length",
                    description=f"All messages have fixed length of {lengths[0]} bytes",
                    frequency=len(samples),
                    confidence=1.0,
                )
            )

        # Analyze common byte sequences
        byte_sequences: Dict[bytes, int] = {}
        for sample in samples:
            for i in range(len(sample) - 3):
                seq = sample[i : i + 4]
                byte_sequences[seq] = byte_sequences.get(seq, 0) + 1

        # Find common sequences (appear in >50% of samples)
        for seq, count in sorted(byte_sequences.items(), key=lambda x: x[1], reverse=True)[:5]:
            if count > len(samples) * 0.5:
                patterns.append(
                    ProtocolPattern(
                        pattern_type="common_sequence",
                        description=f"Common byte sequence: {seq.hex()}",
                        frequency=count,
                        confidence=count / len(samples),
                    )
                )

        # Analyze header patterns (first N bytes)
        header_size = min(16, min(len(s) for s in samples))
        headers = [sample[:header_size] for sample in samples]

        # Check for magic numbers
        first_bytes: Dict[bytes, int] = {}
        for header in headers:
            if len(header) >= 4:
                magic = header[:4]
                first_bytes[magic] = first_bytes.get(magic, 0) + 1

        for magic, count in first_bytes.items():
            if count > len(samples) * 0.7:  # Appears in >70% of samples
                patterns.append(
                    ProtocolPattern(
                        pattern_type="magic_number",
                        description=f"Magic number: {magic.hex()}",
                        frequency=count,
                        confidence=count / len(samples),
                    )
                )

        return patterns

    async def infer_protocol_structure(self, samples: List[bytes], patterns: List[ProtocolPattern]) -> List[ProtocolField]:
        """
        Infer protocol field structure using LLM.

        Args:
            samples: List of protocol message samples
            patterns: Previously identified patterns

        Returns:
            List of inferred protocol fields
        """
        # Prepare sample data for LLM analysis
        sample_hex = [sample.hex() for sample in samples[:10]]  # First 10 samples

        llm_request = LLMRequest(
            prompt=f"""
            Analyze these protocol message samples and infer the field structure:

            Samples (hex):
            {chr(10).join(f"Sample {i+1}: {hex_data}" for i, hex_data in enumerate(sample_hex))}

            Identified Patterns:
            {chr(10).join(f"- {p.pattern_type}: {p.description}" for p in patterns)}

            Please identify:
            1. Field boundaries and offsets
            2. Field types (integer, string, binary, etc.)
            3. Field purposes and descriptions
            4. Any length fields or delimiters
            5. Checksum or validation fields

            Provide your analysis in JSON format with fields array.
            """,
            feature_domain="legacy_whisperer",
            context={"analysis_type": "structure_inference"},
            max_tokens=2000,
            temperature=0.1,
        )

        response = await self.llm_service.process_request(llm_request)

        # Parse LLM response to extract fields
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            analysis = json.loads(content.strip())

            fields = []
            for field_data in analysis.get("fields", []):
                fields.append(
                    ProtocolField(
                        name=field_data.get("name", "unknown"),
                        offset=field_data.get("offset", 0),
                        length=field_data.get("length", 0),
                        field_type=field_data.get("type", "binary"),
                        description=field_data.get("description", ""),
                        confidence=field_data.get("confidence", 0.7),
                    )
                )

            return fields

        except Exception as e:
            self.logger.warning(f"Failed to parse LLM field analysis: {e}")
            # Return basic field structure as fallback
            return [
                ProtocolField(
                    name="header",
                    offset=0,
                    length=16,
                    field_type="binary",
                    description="Protocol header",
                    confidence=0.5,
                )
            ]

    async def identify_message_types(self, samples: List[bytes], fields: List[ProtocolField]) -> List[Dict[str, Any]]:
        """
        Identify different message types in the protocol.

        Args:
            samples: List of protocol message samples
            fields: Identified protocol fields

        Returns:
            List of message type definitions
        """
        message_types = []

        # Group samples by potential type indicators
        type_groups: Dict[str, List[bytes]] = {}
        for sample in samples:
            # Use first few bytes as type indicator
            if len(sample) >= 4:
                type_indicator = sample[:4].hex()
                if type_indicator not in type_groups:
                    type_groups[type_indicator] = []
                type_groups[type_indicator].append(sample)

        # Create message type definitions
        for i, (indicator, group_samples) in enumerate(type_groups.items()):
            message_types.append(
                {
                    "type_id": i + 1,
                    "type_indicator": indicator,
                    "sample_count": len(group_samples),
                    "description": f"Message type {i + 1}",
                    "fields": [f.name for f in fields],
                }
            )

        return message_types

    async def determine_characteristics(self, samples: List[bytes]) -> Dict[str, bool]:
        """
        Determine protocol characteristics.

        Args:
            samples: List of protocol message samples

        Returns:
            Dictionary of protocol characteristics
        """
        characteristics = {
            "is_binary": True,  # Assume binary unless proven otherwise
            "is_stateful": False,
            "uses_encryption": False,
            "has_checksums": False,
        }

        # Check if text-based
        try:
            for sample in samples[:5]:
                sample.decode("ascii")
            characteristics["is_binary"] = False
        except:
            pass

        # Check for high entropy (possible encryption)
        for sample in samples[:10]:
            if len(sample) > 0:
                entropy = len(set(sample)) / len(sample)
                if entropy > 0.9:
                    characteristics["uses_encryption"] = True
                    break

        # Check for checksum patterns (last few bytes often checksums)
        if len(samples) > 5:
            last_bytes = [sample[-4:] for sample in samples if len(sample) >= 4]
            if len(set(last_bytes)) == len(last_bytes):
                characteristics["has_checksums"] = True

        return characteristics

    def assess_complexity(
        self,
        fields: List[ProtocolField],
        patterns: List[ProtocolPattern],
        message_types: List[Dict[str, Any]],
    ) -> ProtocolComplexity:
        """
        Assess protocol complexity.

        Args:
            fields: Identified protocol fields
            patterns: Identified patterns
            message_types: Identified message types

        Returns:
            Protocol complexity level
        """
        complexity_score = 0

        # Factor in number of fields
        complexity_score += len(fields) * 0.3

        # Factor in number of message types
        complexity_score += len(message_types) * 0.4

        # Factor in pattern complexity
        complexity_score += len(patterns) * 0.2

        # Factor in field types diversity
        field_types = set(f.field_type for f in fields)
        complexity_score += len(field_types) * 0.1

        if complexity_score < 5:
            return ProtocolComplexity.SIMPLE
        elif complexity_score < 10:
            return ProtocolComplexity.MODERATE
        elif complexity_score < 20:
            return ProtocolComplexity.COMPLEX
        else:
            return ProtocolComplexity.HIGHLY_COMPLEX

    def calculate_confidence(
        self,
        sample_count: int,
        patterns: List[ProtocolPattern],
        fields: List[ProtocolField],
        message_types: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate overall confidence score.

        Args:
            sample_count: Number of samples analyzed
            patterns: Identified patterns
            fields: Identified fields
            message_types: Identified message types

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0

        # Sample size factor
        if sample_count >= 100:
            confidence += 0.3
        elif sample_count >= 50:
            confidence += 0.2
        elif sample_count >= 20:
            confidence += 0.1

        # Pattern confidence
        if patterns:
            avg_pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
            confidence += avg_pattern_confidence * 0.3

        # Field confidence
        if fields:
            avg_field_confidence = sum(f.confidence for f in fields) / len(fields)
            confidence += avg_field_confidence * 0.3

        # Message type diversity
        if message_types:
            confidence += min(0.1, len(message_types) * 0.02)

        return min(1.0, confidence)
