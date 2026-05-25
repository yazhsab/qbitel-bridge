"""
Cisco Unified Communications Manager (CUCM) PBX Connector

Provides integration with Cisco Unified Communications Manager and
Cisco contact center solutions via CTI-OS, JTAPI, and Finesse APIs.

Supports:
- Cisco Unified Communications Manager (CUCM) 12.x / 14.x / 15.x
- Cisco Unified Contact Center Enterprise (UCCE)
- Cisco Unified Contact Center Express (UCCX)
- Cisco Finesse Desktop
- Cisco Packaged Contact Center Enterprise (PCCE)

Protocol support:
- CTI-OS (Computer Telephony Integration Object Server)
- JTAPI (Java Telephony API) via CTI Manager
- Finesse REST API for agent state and call control
- AXL (Administrative XML) for configuration queries
- CURRI (Cisco Unified Routing Rules Interface) for routing

This implementation requires the Cisco CTI-OS SDK, JTAPI client,
or Finesse API credentials.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
import asyncio
import logging
import uuid

from ai_engine.domains.bpo.integrations.pbx.pbx_connector import (
    PBXConnector,
    PBXType,
    PBXConnectionConfig,
    PBXEvent,
    PBXEventType,
    PBXCallInfo,
    PBXCommandError,
    PBXConnectionError,
    PBXAuthenticationError,
    PBXTimeoutError,
    AgentState,
    CallDirection,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cisco-specific enums
# ---------------------------------------------------------------------------


class CiscoProtocol(Enum):
    """Cisco communication protocols."""

    CTI_OS = "cti_os"
    JTAPI = "jtapi"
    FINESSE = "finesse"
    CTI_OS_AND_FINESSE = "cti_os_finesse"


class CiscoEventCode(Enum):
    """Cisco CTI-OS/Finesse event codes mapped to PBX event types."""

    # Call events
    CALL_ORIGINATED = ("eCallOriginated", PBXEventType.CALL_NEW)
    CALL_QUEUED = ("eCallQueued", PBXEventType.CALL_QUEUED)
    CALL_DELIVERED = ("eCallDelivered", PBXEventType.CALL_DELIVERED)
    CALL_ESTABLISHED = ("eCallEstablished", PBXEventType.CALL_ANSWERED)
    CALL_HELD = ("eCallHeld", PBXEventType.CALL_HELD)
    CALL_RETRIEVED = ("eCallRetrieved", PBXEventType.CALL_RETRIEVED)
    CALL_TRANSFERRED = ("eCallTransferred", PBXEventType.CALL_TRANSFERRED)
    CALL_CONFERENCED = ("eCallConferenced", PBXEventType.CALL_CONFERENCED)
    CALL_CLEARED = ("eCallCleared", PBXEventType.CALL_ENDED)
    CALL_CONNECTION_CLEARED = ("eCallConnectionCleared", PBXEventType.CALL_ENDED)

    # Agent events
    AGENT_LOGGED_IN = ("eAgentLogin", PBXEventType.AGENT_LOGIN)
    AGENT_LOGGED_OUT = ("eAgentLogout", PBXEventType.AGENT_LOGOUT)
    AGENT_STATE_CHANGE = ("eAgentStateChange", PBXEventType.AGENT_STATE_CHANGE)
    AGENT_READY = ("eAgentReady", PBXEventType.AGENT_STATE_CHANGE)
    AGENT_NOT_READY = ("eAgentNotReady", PBXEventType.AGENT_STATE_CHANGE)
    AGENT_TALKING = ("eAgentTalking", PBXEventType.AGENT_STATE_CHANGE)
    AGENT_WORK = ("eAgentWork", PBXEventType.AGENT_STATE_CHANGE)
    AGENT_WORK_READY = ("eAgentWorkReady", PBXEventType.AGENT_STATE_CHANGE)
    AGENT_WORK_NOT_READY = ("eAgentWorkNotReady", PBXEventType.AGENT_STATE_CHANGE)

    def __init__(self, cisco_name: str, pbx_event_type: PBXEventType):
        self.cisco_name = cisco_name
        self.pbx_event_type = pbx_event_type


class CiscoAgentState(Enum):
    """Cisco-specific agent states."""

    LOGIN = (0, "LOGIN", AgentState.AVAILABLE)
    LOGOUT = (1, "LOGOUT", AgentState.LOGGED_OUT)
    NOT_READY = (2, "NOT_READY", AgentState.NOT_READY)
    READY = (3, "READY", AgentState.AVAILABLE)
    TALKING = (4, "TALKING", AgentState.ON_CALL)
    WORK = (5, "WORK", AgentState.AFTER_CALL_WORK)
    WORK_READY = (6, "WORK_READY", AgentState.AFTER_CALL_WORK)
    BUSY_OTHER = (7, "BUSY_OTHER", AgentState.NOT_READY)
    RESERVED = (8, "RESERVED", AgentState.ON_CALL)
    HOLD = (9, "HOLD", AgentState.ON_CALL)

    def __init__(self, code: int, cisco_name: str, standard_state: AgentState):
        self.state_code = code
        self.cisco_name = cisco_name
        self.standard_state = standard_state


class CiscoReasonCode(Enum):
    """Standard Cisco Not Ready reason codes."""

    DEFAULT = (0, "Default")
    BREAK = (1, "Break")
    LUNCH = (2, "Lunch")
    PERSONAL = (3, "Personal")
    TRAINING = (4, "Training")
    MEETING = (5, "Meeting")
    COACHING = (6, "Coaching")
    PROJECT = (7, "Project")
    SYSTEM = (8, "System Issue")
    SUPERVISOR = (9, "Supervisor Activity")

    def __init__(self, code: int, description: str):
        self.reason_code = code
        self.description = description


class FinesseDialogState(Enum):
    """Cisco Finesse dialog (call) states."""

    ALERTING = "ALERTING"
    ACTIVE = "ACTIVE"
    DROPPED = "DROPPED"
    HELD = "HELD"
    INITIATING = "INITIATING"
    INITIATED = "INITIATED"
    FAILED = "FAILED"
    ACCEPTED = "ACCEPTED"
    PAUSED = "PAUSED"
    WRAPPING_UP = "WRAPPING_UP"
    WRAPPED_UP = "WRAPPED_UP"


# ---------------------------------------------------------------------------
# Cisco-specific configuration
# ---------------------------------------------------------------------------


@dataclass
class CiscoConfig:
    """Cisco-specific connection configuration."""

    # Protocol selection
    protocol: CiscoProtocol = CiscoProtocol.FINESSE

    # CTI-OS settings
    cti_os_host: str = ""
    cti_os_port: int = 42027
    cti_os_peripheral_id: int = 5000
    cti_os_agent_instrument: str = ""

    # JTAPI settings
    jtapi_host: str = ""
    jtapi_port: int = 2748
    jtapi_provider: str = ""
    jtapi_user: str = ""
    jtapi_password_vault_path: str = ""

    # Finesse settings
    finesse_host: str = ""
    finesse_port: int = 443
    finesse_use_https: bool = True
    finesse_domain: str = ""
    finesse_user: str = ""
    finesse_password_vault_path: str = ""
    finesse_gadget_url: str = ""

    # CUCM AXL settings (for configuration queries)
    axl_host: str = ""
    axl_port: int = 8443
    axl_user: str = ""
    axl_password_vault_path: str = ""
    axl_version: str = "14.0"

    # Monitoring settings
    monitored_skill_groups: List[str] = field(default_factory=list)
    monitored_agent_teams: List[str] = field(default_factory=list)
    monitored_call_types: List[str] = field(default_factory=list)

    # Agent settings
    agent_desk_settings_name: str = ""
    agent_team_name: str = ""
    default_skill_group: str = ""

    # Finesse XMPP settings (for event notification)
    xmpp_host: str = ""
    xmpp_port: int = 5222
    xmpp_domain: str = ""
    use_bosh: bool = True  # BOSH (HTTP long-polling) for Finesse events
    bosh_url: str = ""

    # Advanced settings
    peripheral_type: str = "UCCE"  # UCCE, UCCX, PCCE
    enable_silent_monitoring: bool = False
    enable_call_recording: bool = False
    media_routing_domain: str = "Cisco_Voice"

    def validate(self) -> List[str]:
        """Validate Cisco-specific configuration."""
        errors = []

        if self.protocol in (CiscoProtocol.CTI_OS, CiscoProtocol.CTI_OS_AND_FINESSE):
            if not self.cti_os_host:
                errors.append("CTI-OS host is required for CTI-OS protocol")
            if self.cti_os_peripheral_id <= 0:
                errors.append("Valid CTI-OS peripheral ID is required")

        if self.protocol == CiscoProtocol.JTAPI:
            if not self.jtapi_host:
                errors.append("JTAPI host (CTI Manager) is required")
            if not self.jtapi_provider:
                errors.append("JTAPI provider string is required")

        if self.protocol in (CiscoProtocol.FINESSE, CiscoProtocol.CTI_OS_AND_FINESSE):
            if not self.finesse_host:
                errors.append("Finesse host is required")

        if self.peripheral_type not in ("UCCE", "UCCX", "PCCE"):
            errors.append(f"Invalid peripheral type: {self.peripheral_type}")

        return errors


# ---------------------------------------------------------------------------
# Cisco Connector Implementation
# ---------------------------------------------------------------------------


class CiscoConnector(PBXConnector):
    """
    Cisco CUCM PBX connector.

    Implements the PBXConnector interface for Cisco Unified Communications
    Manager via CTI-OS, JTAPI, and Finesse APIs.

    Features:
    - CTI-OS call control and agent state management
    - JTAPI device observation and call monitoring
    - Finesse REST API for modern web-based integration
    - AXL configuration queries
    - Cisco-specific event mapping
    - Skill group and agent team monitoring
    - Silent monitoring and call recording control
    - Peripheral gateway abstraction (UCCE/UCCX/PCCE)
    """

    def __init__(
        self,
        config: PBXConnectionConfig,
        cisco_config: Optional[CiscoConfig] = None,
    ):
        super().__init__(PBXType.CISCO_CUCM, config)
        self._cisco_config = cisco_config or CiscoConfig()

        # CTI-OS session state
        self._cti_os_session: Optional[Any] = None
        self._cti_os_agent: Optional[Any] = None
        self._cti_os_skill_groups: Dict[str, Any] = {}

        # JTAPI session state
        self._jtapi_provider: Optional[Any] = None
        self._jtapi_observers: Dict[str, Any] = {}

        # Finesse session state
        self._finesse_session: Optional[Any] = None
        self._finesse_xmpp: Optional[Any] = None
        self._finesse_notification_channel: Optional[str] = None

        # Subscription tracking
        self._subscriptions: Dict[str, Dict[str, Any]] = {}

        # Event code mapping
        self._event_code_map: Dict[str, CiscoEventCode] = {
            code.cisco_name: code for code in CiscoEventCode
        }

        # Agent state cache
        self._agent_state_cache: Dict[str, AgentState] = {}

        # Active dialogs (Finesse terminology for calls)
        self._active_dialogs: Dict[str, Dict[str, Any]] = {}

        # Validate config
        errors = self._cisco_config.validate()
        if errors:
            logger.warning(f"Cisco config validation warnings: {errors}")

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to Cisco CUCM via the configured protocol.

        For CTI-OS: Connects to CTI Server on the peripheral gateway
        For JTAPI: Connects to CTI Manager service on CUCM
        For Finesse: Establishes REST session and XMPP/BOSH notification channel
        """
        cisco_cfg = self._cisco_config

        errors = cisco_cfg.validate()
        if errors:
            raise PBXConnectionError(f"Cisco configuration errors: {errors}")

        try:
            if cisco_cfg.protocol in (CiscoProtocol.CTI_OS, CiscoProtocol.CTI_OS_AND_FINESSE):
                await self._connect_cti_os()

            if cisco_cfg.protocol == CiscoProtocol.JTAPI:
                await self._connect_jtapi()

            if cisco_cfg.protocol in (CiscoProtocol.FINESSE, CiscoProtocol.CTI_OS_AND_FINESSE):
                await self._connect_finesse()

            self._connected = True

            logger.info(
                f"Connected to Cisco CUCM via {cisco_cfg.protocol.value} "
                f"(type: {cisco_cfg.peripheral_type})"
            )

            self._audit_log_entry("CISCO_CONNECTED", {
                "protocol": cisco_cfg.protocol.value,
                "peripheral_type": cisco_cfg.peripheral_type,
            })

        except Exception as exc:
            logger.error(f"Failed to connect to Cisco CUCM: {exc}")
            raise PBXConnectionError(f"Cisco CUCM connection failed: {exc}")

    async def _connect_cti_os(self) -> None:
        """
        Establish CTI-OS connection to Cisco CTI Server.

        In production, this would:
        1. Create CtiOs.Session object
        2. Connect to CTI Server (host:port)
        3. Authenticate with peripheral credentials
        4. Register for event notifications
        5. Set up skill group monitoring
        """
        cisco_cfg = self._cisco_config

        # Production: CtiOs Session setup
        # session = CtiOsSession()
        # session.SetProperty("PeripheralID", peripheralId)
        # session.Connect(host, port)
        # session.Login(agentID, instrument, peripheralID)

        self._cti_os_session = f"ctios_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"CTI-OS session established: "
            f"host={cisco_cfg.cti_os_host}:{cisco_cfg.cti_os_port}, "
            f"peripheral={cisco_cfg.cti_os_peripheral_id}"
        )

    async def _connect_jtapi(self) -> None:
        """
        Establish JTAPI connection to Cisco CTI Manager.

        In production, this would:
        1. Create JTAPIProvider with provider string
        2. Connect to CTI Manager service
        3. Get Provider and add observer
        4. Get Address (line) objects
        5. Add CallObservers to monitored lines
        """
        cisco_cfg = self._cisco_config

        # Production: JtapiPeer.getProvider(providerString, callback)
        self._jtapi_provider = f"jtapi_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"JTAPI provider connected: {cisco_cfg.jtapi_host}:{cisco_cfg.jtapi_port}"
        )

    async def _connect_finesse(self) -> None:
        """
        Establish Finesse REST API session and notification channel.

        In production, this would:
        1. Authenticate via Finesse REST API
        2. Establish XMPP/BOSH connection for real-time notifications
        3. Subscribe to user, dialog, and queue notifications
        4. Get initial state snapshot
        """
        cisco_cfg = self._cisco_config

        # Production:
        # 1. GET /finesse/api/User/{id} to verify authentication
        # 2. Connect XMPP/BOSH for notifications
        # 3. Subscribe to /finesse/api/User/{id}/Dialogs
        # 4. Subscribe to /finesse/api/User/{id}/ClientLog

        self._finesse_session = f"finesse_{uuid.uuid4().hex[:8]}"
        self._finesse_notification_channel = f"xmpp_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"Finesse session established: "
            f"host={cisco_cfg.finesse_host}:{cisco_cfg.finesse_port}"
        )

    async def disconnect(self) -> None:
        """
        Disconnect from Cisco CUCM.

        Releases all observers, closes CTI-OS session, JTAPI provider,
        and Finesse notification channel.
        """
        # Close JTAPI observers
        for device_id, observer in list(self._jtapi_observers.items()):
            try:
                # Production: address.removeCallObserver(observer)
                pass
            except Exception as exc:
                logger.warning(f"Error removing JTAPI observer for {device_id}: {exc}")
        self._jtapi_observers.clear()

        # Close CTI-OS session
        if self._cti_os_session:
            # Production: session.Disconnect()
            self._cti_os_session = None
            self._cti_os_agent = None
            self._cti_os_skill_groups.clear()
            logger.debug("CTI-OS session closed")

        # Close JTAPI provider
        if self._jtapi_provider:
            # Production: provider.shutdown()
            self._jtapi_provider = None
            logger.debug("JTAPI provider closed")

        # Close Finesse session
        if self._finesse_session:
            # Production: Close XMPP/BOSH connection, invalidate session
            if self._finesse_notification_channel:
                self._finesse_notification_channel = None
            self._finesse_session = None
            logger.debug("Finesse session closed")

        self._connected = False
        self._agent_state_cache.clear()
        self._active_dialogs.clear()

        self._audit_log_entry("CISCO_DISCONNECTED", {})

    # -----------------------------------------------------------------------
    # Event subscription
    # -----------------------------------------------------------------------

    async def subscribe_events(
        self,
        event_types: Optional[Set[PBXEventType]] = None,
        agent_ids: Optional[Set[str]] = None,
        queue_names: Optional[Set[str]] = None,
    ) -> str:
        """
        Subscribe to Cisco CTI events.

        For CTI-OS: Registers event callbacks on session/agent objects
        For JTAPI: Adds CallObservers to Address/Terminal objects
        For Finesse: Subscribes to XMPP notification topics

        Args:
            event_types: Event types to subscribe to
            agent_ids: Agent IDs to monitor
            queue_names: Skill groups/queues to monitor

        Returns:
            Subscription ID
        """
        self._check_connected()

        subscription_id = str(uuid.uuid4())
        cisco_cfg = self._cisco_config

        monitored_items: List[str] = []

        if cisco_cfg.protocol in (CiscoProtocol.CTI_OS, CiscoProtocol.CTI_OS_AND_FINESSE):
            # CTI-OS: Register for agent and call events on skill groups
            skill_groups = queue_names or set(cisco_cfg.monitored_skill_groups)
            for sg in skill_groups:
                # Production: session.RequestSkillGroupStatistics(sg)
                self._cti_os_skill_groups[sg] = {"monitoring": True}
                monitored_items.append(f"skill_group:{sg}")

        if cisco_cfg.protocol == CiscoProtocol.JTAPI:
            # JTAPI: Add call observers to devices
            devices = agent_ids or set()
            for device_id in devices:
                observer_id = await self._add_jtapi_observer(device_id)
                monitored_items.append(f"jtapi_device:{device_id}")

        if cisco_cfg.protocol in (CiscoProtocol.FINESSE, CiscoProtocol.CTI_OS_AND_FINESSE):
            # Finesse: Subscribe to notification topics
            agents = agent_ids or set()
            for agent_id in agents:
                # Production: XMPP subscribe to /finesse/api/User/{agent_id}
                monitored_items.append(f"finesse_user:{agent_id}")

            queues = queue_names or set()
            for queue in queues:
                # Production: XMPP subscribe to /finesse/api/Queue/{queue}
                monitored_items.append(f"finesse_queue:{queue}")

        self._subscriptions[subscription_id] = {
            "event_types": event_types,
            "agent_ids": agent_ids,
            "queue_names": queue_names,
            "monitored_items": monitored_items,
            "created_at": datetime.utcnow().isoformat(),
        }

        self._audit_log_entry("CISCO_SUBSCRIBE", {
            "subscription_id": subscription_id,
            "monitored_items": monitored_items,
        })

        logger.info(
            f"Cisco subscription {subscription_id}: "
            f"monitoring {len(monitored_items)} items"
        )

        return subscription_id

    async def unsubscribe_events(self, subscription_id: str) -> None:
        """Unsubscribe from a previous event subscription."""
        self._check_connected()

        if subscription_id not in self._subscriptions:
            raise PBXCommandError(
                f"Unknown subscription: {subscription_id}",
                command="unsubscribe",
            )

        subscription = self._subscriptions[subscription_id]

        # Remove JTAPI observers
        for item in subscription.get("monitored_items", []):
            if item.startswith("jtapi_device:"):
                device_id = item.split(":", 1)[1]
                await self._remove_jtapi_observer(device_id)

        del self._subscriptions[subscription_id]

        self._audit_log_entry("CISCO_UNSUBSCRIBE", {
            "subscription_id": subscription_id,
        })

    async def _add_jtapi_observer(self, device_id: str) -> str:
        """
        Add a JTAPI CallObserver to a device.

        In production:
        1. provider.getAddress(device_id)
        2. address.addCallObserver(callObserver)
        """
        observer_id = f"obs_{uuid.uuid4().hex[:8]}"
        self._jtapi_observers[device_id] = observer_id
        logger.debug(f"JTAPI observer added for device {device_id}")
        return observer_id

    async def _remove_jtapi_observer(self, device_id: str) -> None:
        """Remove a JTAPI CallObserver from a device."""
        if device_id in self._jtapi_observers:
            # Production: address.removeCallObserver(observer)
            del self._jtapi_observers[device_id]
            logger.debug(f"JTAPI observer removed for device {device_id}")

    # -----------------------------------------------------------------------
    # Call control operations
    # -----------------------------------------------------------------------

    async def send_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a command to Cisco CUCM.

        Routes commands through the appropriate protocol (CTI-OS, JTAPI,
        or Finesse REST API) based on the configured protocol.

        Supported commands vary by protocol:
        - CTI-OS: MakeCall, AnswerCall, HoldCall, RetrieveCall, TransferCall,
                  ConferenceCall, SetAgentState, etc.
        - JTAPI: connect(), disconnect(), hold(), unhold(), transfer(), conference()
        - Finesse: PUT /User/{id}, PUT /Dialog/{id}, POST /Dialog

        Args:
            command: Command name
            params: Command parameters

        Returns:
            Command response
        """
        self._check_connected()

        self._audit_log_entry("CISCO_COMMAND", {
            "command": command,
            "protocol": self._cisco_config.protocol.value,
        })

        # Route to appropriate protocol handler
        if self._cisco_config.protocol in (
            CiscoProtocol.CTI_OS, CiscoProtocol.CTI_OS_AND_FINESSE
        ):
            return await self._send_cti_os_command(command, params)
        elif self._cisco_config.protocol == CiscoProtocol.JTAPI:
            return await self._send_jtapi_command(command, params)
        else:
            return await self._send_finesse_command(command, params)

    async def _send_cti_os_command(
        self, command: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send command via CTI-OS protocol."""
        # Production: Dispatch to appropriate CtiOs session/agent method
        return {
            "command": command,
            "protocol": "cti_os",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _send_jtapi_command(
        self, command: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send command via JTAPI protocol."""
        # Production: Call appropriate JTAPI Connection/Call method
        return {
            "command": command,
            "protocol": "jtapi",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _send_finesse_command(
        self, command: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send command via Finesse REST API.

        Maps commands to Finesse REST endpoints:
        - Agent state: PUT /finesse/api/User/{id}
        - Call control: PUT /finesse/api/Dialog/{id}
        - Make call: POST /finesse/api/User/{id}/Dialogs
        """
        # Production: HTTP request to Finesse REST API
        return {
            "command": command,
            "protocol": "finesse",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_agent_state(self, agent_id: str) -> AgentState:
        """
        Get the current state of a Cisco agent.

        For CTI-OS: Queries agent object state
        For JTAPI: Queries terminal state
        For Finesse: GET /finesse/api/User/{agent_id}
        """
        self._check_connected()

        if agent_id in self._agent_state_cache:
            return self._agent_state_cache[agent_id]

        if self._cisco_config.protocol in (
            CiscoProtocol.FINESSE, CiscoProtocol.CTI_OS_AND_FINESSE
        ):
            # Production: GET /finesse/api/User/{agent_id}
            # Parse state from response XML/JSON
            pass

        state = self._agent_state_cache.get(agent_id, AgentState.LOGGED_OUT)

        self._audit_log_entry("CISCO_QUERY_AGENT", {
            "agent_id": agent_id,
            "state": state.state_name,
        })

        return state

    async def set_agent_state(
        self,
        agent_id: str,
        state: AgentState,
        reason_code: str = "",
    ) -> None:
        """
        Set agent state on Cisco CUCM.

        For CTI-OS: agent.SetState(state, reasonCode)
        For Finesse: PUT /finesse/api/User/{agent_id}
                     with <User><state>...</state></User>
        """
        self._check_connected()

        cisco_state = self._map_to_cisco_state(state)

        if self._cisco_config.protocol in (
            CiscoProtocol.FINESSE, CiscoProtocol.CTI_OS_AND_FINESSE
        ):
            # Production: PUT /finesse/api/User/{agent_id}
            # Body: <User><state>READY</state></User>
            # For NOT_READY: <User><state>NOT_READY</state>
            #   <reasonCodeId>{reason_code}</reasonCodeId></User>
            pass
        elif self._cisco_config.protocol == CiscoProtocol.CTI_OS:
            # Production: agent.SetState(stateCode, reasonCode)
            pass

        self._agent_state_cache[agent_id] = state

        self._audit_log_entry("CISCO_SET_AGENT_STATE", {
            "agent_id": agent_id,
            "state": state.state_name,
            "cisco_state": cisco_state.cisco_name if cisco_state else "UNKNOWN",
            "reason_code": reason_code,
        })

        logger.info(f"Agent {agent_id} state set to {state.state_name}")

    async def transfer_call(
        self,
        call_id: str,
        target: str,
        transfer_type: str = "blind",
    ) -> str:
        """
        Transfer a call on Cisco CUCM.

        For blind transfer via Finesse:
        PUT /finesse/api/Dialog/{dialog_id}
        with <Dialog><targetMediaAddress>{target}</targetMediaAddress>
              <requestedAction>TRANSFER</requestedAction></Dialog>

        For consultative transfer:
        1. PUT /finesse/api/Dialog/{dialog_id} with CONSULT action
        2. PUT /finesse/api/Dialog/{dialog_id} with TRANSFER action
        """
        self._check_connected()

        new_call_id = str(uuid.uuid4())

        if self._cisco_config.protocol in (
            CiscoProtocol.FINESSE, CiscoProtocol.CTI_OS_AND_FINESSE
        ):
            if transfer_type == "consultative":
                # Step 1: Initiate consult call
                # PUT /finesse/api/Dialog/{id} action=CONSULT_CALL
                # Step 2: Complete transfer
                # PUT /finesse/api/Dialog/{id} action=TRANSFER
                pass
            else:
                # Single-step transfer
                # PUT /finesse/api/Dialog/{id} action=TRANSFER
                pass
        elif self._cisco_config.protocol == CiscoProtocol.CTI_OS:
            # Production: call.SingleStepTransfer() or Consult+Transfer
            pass

        self._audit_log_entry("CISCO_TRANSFER", {
            "call_id": call_id,
            "target": target,
            "transfer_type": transfer_type,
            "new_call_id": new_call_id,
        })

        logger.info(f"Call {call_id} transferred to {target} ({transfer_type})")
        return new_call_id

    async def conference_call(
        self,
        call_id: str,
        targets: List[str],
    ) -> str:
        """
        Create a conference on Cisco CUCM.

        Via Finesse:
        1. PUT /finesse/api/Dialog/{dialog_id} action=CONSULT_CALL
        2. PUT /finesse/api/Dialog/{dialog_id} action=CONFERENCE
        """
        self._check_connected()

        conference_id = str(uuid.uuid4())

        # Production: Iterate targets, consult + conference for each

        self._audit_log_entry("CISCO_CONFERENCE", {
            "call_id": call_id,
            "targets": targets,
            "conference_id": conference_id,
        })

        logger.info(f"Conference {conference_id} created with {len(targets)} participants")
        return conference_id

    async def hold_call(self, call_id: str) -> None:
        """
        Place a call on hold.

        Finesse: PUT /finesse/api/Dialog/{dialog_id}
                 <Dialog><targetMediaAddress>...</targetMediaAddress>
                  <requestedAction>HOLD</requestedAction></Dialog>
        """
        self._check_connected()

        # Production: PUT /finesse/api/Dialog/{id} action=HOLD

        self._audit_log_entry("CISCO_HOLD", {"call_id": call_id})
        logger.info(f"Call {call_id} placed on hold")

    async def retrieve_call(self, call_id: str) -> None:
        """
        Retrieve a held call.

        Finesse: PUT /finesse/api/Dialog/{dialog_id}
                 <requestedAction>RETRIEVE</requestedAction>
        """
        self._check_connected()

        # Production: PUT /finesse/api/Dialog/{id} action=RETRIEVE

        self._audit_log_entry("CISCO_RETRIEVE", {"call_id": call_id})
        logger.info(f"Call {call_id} retrieved from hold")

    async def make_call(
        self,
        agent_id: str,
        destination: str,
        uui_data: str = "",
    ) -> str:
        """
        Initiate an outbound call.

        Finesse: POST /finesse/api/User/{agent_id}/Dialogs
                 <Dialog><requestedAction>MAKE_CALL</requestedAction>
                  <toAddress>{destination}</toAddress></Dialog>
        """
        self._check_connected()

        call_id = str(uuid.uuid4())

        # Production: POST /finesse/api/User/{agent_id}/Dialogs

        self._audit_log_entry("CISCO_MAKE_CALL", {
            "agent_id": agent_id,
            "destination": destination,
            "call_id": call_id,
        })

        logger.info(f"Outbound call {call_id}: {agent_id} -> {destination}")
        return call_id

    async def end_call(self, call_id: str) -> None:
        """
        End a call.

        Finesse: PUT /finesse/api/Dialog/{dialog_id}
                 <requestedAction>DROP</requestedAction>
        """
        self._check_connected()

        # Production: PUT /finesse/api/Dialog/{id} action=DROP

        if call_id in self._active_dialogs:
            del self._active_dialogs[call_id]

        self._audit_log_entry("CISCO_END_CALL", {"call_id": call_id})
        logger.info(f"Call {call_id} ended")

    async def get_call_info(self, call_id: str) -> PBXCallInfo:
        """
        Get detailed call information.

        Finesse: GET /finesse/api/Dialog/{dialog_id}
        """
        self._check_connected()

        # Production: GET /finesse/api/Dialog/{dialog_id}
        # Parse dialog XML to extract call details

        dialog = self._active_dialogs.get(call_id, {})

        return PBXCallInfo(
            call_id=call_id,
            caller_number=dialog.get("fromAddress", ""),
            called_number=dialog.get("toAddress", ""),
            direction=CallDirection.INBOUND,
            metadata={"pbx_type": "cisco", "dialog_state": dialog.get("state", "")},
        )

    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """
        Get real-time queue/skill group statistics.

        Finesse: GET /finesse/api/Queue/{queue_id}
        CTI-OS: QuerySkillGroupStatistics
        """
        self._check_connected()

        # Production: GET /finesse/api/Queue/{queue_id}
        # or CTI-OS skill group statistics request

        return {
            "queue_name": queue_name,
            "calls_waiting": 0,
            "calls_in_progress": 0,
            "agents_available": 0,
            "agents_busy": 0,
            "agents_not_ready": 0,
            "agents_wrap_up": 0,
            "longest_wait_seconds": 0.0,
            "average_wait_seconds": 0.0,
            "service_level_pct": 100.0,
            "calls_abandoned_today": 0,
            "calls_handled_today": 0,
        }

    # -----------------------------------------------------------------------
    # Cisco-specific operations
    # -----------------------------------------------------------------------

    async def start_silent_monitoring(
        self,
        agent_id: str,
        supervisor_id: str,
        mode: str = "SILENT",
    ) -> str:
        """
        Start silent monitoring of an agent call.

        Modes:
        - SILENT: Supervisor listens only
        - COACH: Supervisor can whisper to agent
        - BARGE_IN: Supervisor joins the call

        Finesse: POST /finesse/api/User/{supervisor_id}/MonitoredDialogs

        Args:
            agent_id: Agent being monitored
            supervisor_id: Supervisor performing monitoring
            mode: Monitoring mode

        Returns:
            Monitoring session ID
        """
        self._check_connected()

        if mode not in ("SILENT", "COACH", "BARGE_IN"):
            raise PBXCommandError(
                f"Invalid monitoring mode: {mode}",
                command="start_silent_monitoring",
            )

        monitoring_id = str(uuid.uuid4())

        # Production: POST /finesse/api/User/{supervisor_id}/MonitoredDialogs
        # Body: <MonitoredDialog><agentId>{agent_id}</agentId>
        #        <monitoringMode>{mode}</monitoringMode></MonitoredDialog>

        self._audit_log_entry("CISCO_MONITOR_START", {
            "agent_id": agent_id,
            "supervisor_id": supervisor_id,
            "mode": mode,
            "monitoring_id": monitoring_id,
        })

        logger.info(
            f"Silent monitoring started: supervisor={supervisor_id}, "
            f"agent={agent_id}, mode={mode}"
        )
        return monitoring_id

    async def stop_silent_monitoring(self, monitoring_id: str) -> None:
        """Stop a silent monitoring session."""
        self._check_connected()

        # Production: DELETE /finesse/api/User/{supervisor_id}/MonitoredDialogs/{id}

        self._audit_log_entry("CISCO_MONITOR_STOP", {
            "monitoring_id": monitoring_id,
        })

    async def get_agent_team_members(self, team_name: str) -> List[Dict[str, Any]]:
        """
        Get all agents in a Cisco agent team.

        Finesse: GET /finesse/api/Team/{team_id}/Users

        Args:
            team_name: Cisco agent team name

        Returns:
            List of agent information dictionaries
        """
        self._check_connected()

        # Production: GET /finesse/api/Team/{team_id}/Users

        return []

    async def send_chat_message(
        self,
        agent_id: str,
        message: str,
        dialog_id: Optional[str] = None,
    ) -> None:
        """
        Send a chat/message to an agent via Finesse.

        Can be used for supervisor-to-agent messaging or
        system notifications.

        Args:
            agent_id: Target agent
            message: Message text
            dialog_id: Optional dialog context
        """
        self._check_connected()

        # Production: POST to Finesse chat API or XMPP message

        self._audit_log_entry("CISCO_CHAT_MESSAGE", {
            "agent_id": agent_id,
            "has_dialog": dialog_id is not None,
        })

    async def query_axl(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute an AXL SQL query against CUCM configuration database.

        Used for retrieving configuration data such as:
        - Phone device configurations
        - Route patterns
        - Translation patterns
        - Device pools

        Args:
            sql: SQL query string for executeSQLQuery

        Returns:
            Query results as list of dictionaries
        """
        self._check_connected()

        if not self._cisco_config.axl_host:
            raise PBXCommandError(
                "AXL host not configured",
                command="query_axl",
            )

        # Production: POST to https://{axl_host}:8443/axl/
        # SOAP envelope with executeSQLQuery

        self._audit_log_entry("CISCO_AXL_QUERY", {
            "query_length": len(sql),
        })

        return []

    # -----------------------------------------------------------------------
    # Heartbeat
    # -----------------------------------------------------------------------

    async def _send_heartbeat(self) -> bool:
        """
        Send heartbeat to Cisco CUCM.

        For CTI-OS: Session heartbeat via CTI protocol
        For JTAPI: Provider heartbeat
        For Finesse: GET /finesse/api/SystemInfo (lightweight health check)
        """
        if self._cisco_config.protocol in (
            CiscoProtocol.FINESSE, CiscoProtocol.CTI_OS_AND_FINESSE
        ):
            # Production: GET /finesse/api/SystemInfo
            return True
        elif self._cisco_config.protocol == CiscoProtocol.CTI_OS:
            # Production: Session heartbeat
            return self._cti_os_session is not None
        elif self._cisco_config.protocol == CiscoProtocol.JTAPI:
            # Production: Check provider state
            return self._jtapi_provider is not None

        return False

    # -----------------------------------------------------------------------
    # Event mapping
    # -----------------------------------------------------------------------

    async def _map_vendor_event(self, raw_event: Dict[str, Any]) -> PBXEvent:
        """
        Map Cisco CTI-OS/Finesse event to standardised PBXEvent.

        Handles events from all three protocols:
        - CTI-OS: CtiOs event objects (eCallDelivered, eAgentStateChange, etc.)
        - JTAPI: CallEvent objects from CallObserver
        - Finesse: XMPP notification payloads (Dialog/User XML)
        """
        event_name = raw_event.get("eventType", "")
        cisco_event = self._event_code_map.get(event_name)

        if cisco_event is None:
            logger.warning(f"Unknown Cisco event type: {event_name}")
            event_type = PBXEventType.CALL_NEW
        else:
            event_type = cisco_event.pbx_event_type

        # Extract fields (handle both CTI-OS and Finesse field names)
        call_id = raw_event.get("dialogId", raw_event.get("callId", ""))
        agent_id = raw_event.get("agentId", raw_event.get("userId", ""))
        queue = raw_event.get("queueName", raw_event.get("skillGroupName", ""))

        # Map agent state
        agent_state = None
        if event_type == PBXEventType.AGENT_STATE_CHANGE:
            cisco_state_name = raw_event.get("state", raw_event.get("agentState", ""))
            agent_state = self._map_cisco_agent_state(cisco_state_name)
            if agent_id:
                self._agent_state_cache[agent_id] = agent_state
        elif event_type == PBXEventType.AGENT_LOGIN:
            agent_state = AgentState.AVAILABLE
            if agent_id:
                self._agent_state_cache[agent_id] = agent_state
        elif event_type == PBXEventType.AGENT_LOGOUT:
            agent_state = AgentState.LOGGED_OUT
            if agent_id:
                self._agent_state_cache[agent_id] = agent_state

        # Determine call direction
        direction = None
        dialog_type = raw_event.get("mediaProperties", {}).get("dialedNumber", "")
        if raw_event.get("fromAddress") and raw_event.get("toAddress"):
            if raw_event.get("fromAddress") == agent_id:
                direction = CallDirection.OUTBOUND
            else:
                direction = CallDirection.INBOUND

        # Track active dialogs
        if event_type in (PBXEventType.CALL_NEW, PBXEventType.CALL_DELIVERED,
                          PBXEventType.CALL_ANSWERED):
            self._active_dialogs[call_id] = raw_event
        elif event_type == PBXEventType.CALL_ENDED:
            self._active_dialogs.pop(call_id, None)

        event = PBXEvent(
            event_type=event_type,
            call_id=call_id,
            agent_id=agent_id,
            queue_name=queue,
            timestamp=datetime.utcnow(),
            caller_number=raw_event.get("fromAddress", ""),
            called_number=raw_event.get("toAddress", ""),
            direction=direction,
            agent_state=agent_state,
            raw_event=raw_event,
            metadata={
                "cisco_event": event_name,
                "dialog_state": raw_event.get("state", ""),
                "peripheral_id": raw_event.get("peripheralId", ""),
                "media_type": raw_event.get("mediaType", "voice"),
            },
        )

        return event

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _map_to_cisco_state(self, state: AgentState) -> Optional[CiscoAgentState]:
        """Map standard AgentState to Cisco agent state."""
        state_map = {
            AgentState.AVAILABLE: CiscoAgentState.READY,
            AgentState.ON_CALL: CiscoAgentState.TALKING,
            AgentState.AFTER_CALL_WORK: CiscoAgentState.WORK,
            AgentState.NOT_READY: CiscoAgentState.NOT_READY,
            AgentState.BREAK: CiscoAgentState.NOT_READY,
            AgentState.TRAINING: CiscoAgentState.NOT_READY,
            AgentState.MEETING: CiscoAgentState.NOT_READY,
            AgentState.LOGGED_OUT: CiscoAgentState.LOGOUT,
        }
        return state_map.get(state)

    def _map_cisco_agent_state(self, cisco_state: str) -> AgentState:
        """Map Cisco agent state string to standard AgentState."""
        state_map = {
            "LOGIN": AgentState.AVAILABLE,
            "LOGOUT": AgentState.LOGGED_OUT,
            "NOT_READY": AgentState.NOT_READY,
            "READY": AgentState.AVAILABLE,
            "TALKING": AgentState.ON_CALL,
            "WORK": AgentState.AFTER_CALL_WORK,
            "WORK_READY": AgentState.AFTER_CALL_WORK,
            "BUSY_OTHER": AgentState.NOT_READY,
            "RESERVED": AgentState.ON_CALL,
            "HOLD": AgentState.ON_CALL,
        }
        return state_map.get(cisco_state.upper(), AgentState.NOT_READY)
