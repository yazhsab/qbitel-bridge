"""
QBITEL Bridge - Legacy Whisperer Multi-Agent System

A modern multi-agent architecture for legacy system understanding
and modernization, inspired by frameworks like AutoGen and CrewAI.

Agents:
- ProtocolAnalystAgent: Analyzes traffic patterns and protocol structure
- DocumentationAgent: Generates comprehensive documentation
- RiskAssessorAgent: Evaluates modernization risks
- CodeGeneratorAgent: Generates adapter code and tests
- OrchestratorAgent: Coordinates agent collaboration

Features:
- Autonomous tool use
- Inter-agent communication
- Shared memory and context
- Self-reflection and error correction
- Human-in-the-loop support
"""

from .base import (
    AgentRole,
    AgentMessage,
    AgentMemory,
    AgentTool,
    ToolResult,
    BaseAgent,
    AgentConfig,
)

from .protocol_analyst import ProtocolAnalystAgent
from .documentation_agent import DocumentationAgent
from .risk_assessor import RiskAssessorAgent
from .code_generator import CodeGeneratorAgent
from .orchestrator_agent import OrchestratorAgent, MultiAgentOrchestrator

from .tools import (
    TrafficAnalysisTool,
    PatternRecognitionTool,
    DocumentationSearchTool,
    RiskCalculatorTool,
    CodeGenerationTool,
    TestGenerationTool,
)

from .memory import (
    SharedMemory,
    ConversationMemory,
    EpisodicMemory,
    SemanticMemory,
)

__all__ = [
    # Base classes
    "AgentRole",
    "AgentMessage",
    "AgentMemory",
    "AgentTool",
    "ToolResult",
    "BaseAgent",
    "AgentConfig",
    # Agents
    "ProtocolAnalystAgent",
    "DocumentationAgent",
    "RiskAssessorAgent",
    "CodeGeneratorAgent",
    "OrchestratorAgent",
    "MultiAgentOrchestrator",
    # Tools
    "TrafficAnalysisTool",
    "PatternRecognitionTool",
    "DocumentationSearchTool",
    "RiskCalculatorTool",
    "CodeGenerationTool",
    "TestGenerationTool",
    # Memory
    "SharedMemory",
    "ConversationMemory",
    "EpisodicMemory",
    "SemanticMemory",
]
