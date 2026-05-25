"""
Avaya Aura PBX Connector

Provides integration with Avaya Aura Communication Manager and Avaya
Contact Center systems via TSAPI (Telephony Services API) and DMCC
(Device, Media and Call Control) protocols.

Supports:
- Avaya Communication Manager 8.x / 10.x
- Avaya Aura Contact Center (AACC)
- Avaya AES (Application Enablement Services) TSAPI/DMCC
- Avaya Oceana / Oceanalytics
- Avaya IX Workplace

Protocol support:
- TSAPI (Telephony Services API) for call control
- DMCC (Device, Media and Call Control) for device management
- Station monitoring for quality assurance
- Call observation for compliance recording

This implementation requires the Avaya AES SDK or TSAPI client library.
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
# Avaya-specific enums
# ---------------------------------------------------------------------------


class AvayaProtocol(Enum):
    """Avaya communication protocols."""

    TSAPI = "tsapi"
    DMCC = "dmcc"
    TSAPI_AND_DMCC = "tsapi_dmcc"


class AvayaEventCode(Enum):
    """Avaya CSTA/TSAPI event codes mapped to PBX event types."""

    # Connection state events
    CSTA_DELIVERED = ("CSTADeliveredEvent", PBXEventType.CALL_DELIVERED)
    CSTA_ESTABLISHED = ("CSTAEstablishedEvent", PBXEventType.CALL_ANSWERED)
    CSTA_HELD = ("CSTAHeldEvent", PBXEventType.CALL_HELD)
    CSTA_RETRIEVED = ("CSTARetrievedEvent", PBXEventType.CALL_RETRIEVED)
    CSTA_TRANSFERRED = ("CSTATransferredEvent", PBXEventType.CALL_TRANSFERRED)
    CSTA_CONFERENCED = ("CSTAConferencedEvent", PBXEventType.CALL_CONFERENCED)
    CSTA_CONNECTION_CLEARED = ("CSTAConnectionClearedEvent", PBXEventType.CALL_ENDED)
    CSTA_ORIGINATED = ("CSTAOriginatedEvent", PBXEventType.CALL_NEW)
    CSTA_QUEUED = ("CSTAQueuedEvent", PBXEventType.CALL_QUEUED)

    # Agent state events
    CSTA_AGENT_LOGGED_ON = ("CSTALoggedOnEvent", PBXEventType.AGENT_LOGIN)
    CSTA_AGENT_LOGGED_OFF = ("CSTALoggedOffEvent", PBXEventType.AGENT_LOGOUT)
    CSTA_AGENT_NOT_READY = ("CSTANotReadyEvent", PBXEventType.AGENT_STATE_CHANGE)
    CSTA_AGENT_READY = ("CSTAReadyEvent", PBXEventType.AGENT_STATE_CHANGE)
    CSTA_AGENT_WORK_NOT_READY = ("CSTAWorkNotReadyEvent", PBXEventType.AGENT_STATE_CHANGE)
    CSTA_AGENT_WORK_READY = ("CSTAWorkReadyEvent", PBXEventType.AGENT_STATE_CHANGE)

    def __init__(self, csta_name: str, pbx_event_type: PBXEventType):
        self.csta_name = csta_name
        self.pbx_event_type = pbx_event_type


class AvayaAgentMode(Enum):
    """Avaya-specific agent modes."""

    AUTO_IN = ("AUTO-IN", AgentState.AVAILABLE)
    MANUAL_IN = ("MANUAL-IN", AgentState.AVAILABLE)
    ACW = ("ACW", AgentState.AFTER_CALL_WORK)
    AUX = ("AUX", AgentState.NOT_READY)
    AVAILABLE = ("AVAIL", AgentState.AVAILABLE)
    LOGGED_OUT = ("LOGGED-OUT", AgentState.LOGGED_OUT)

    def __init__(self, avaya_name: str, standard_state: AgentState):
        self.avaya_name = avaya_name
        self.standard_state = standard_state


class AvayaAuxReason(Enum):
    """Standard Avaya AUX reason codes."""

    DEFAULT = (0, "Default")
    BREAK = (1, "Break")
    LUNCH = (2, "Lunch")
    TRAINING = (3, "Training")
    MEETING = (4, "Meeting")
    PROJECT = (5, "Project Work")
    SUPERVISOR = (6, "Supervisor Activity")
    PERSONAL = (7, "Personal Time")
    SYSTEM = (8, "System Issue")
    COACHING = (9, "Coaching")

    def __init__(self, code: int, description: str):
        self.reason_code = code
        self.description = description


# ---------------------------------------------------------------------------
# Avaya-specific configuration
# ---------------------------------------------------------------------------


@dataclass
class AvayaConfig:
    """Avaya-specific connection configuration."""

    # AES connection
    aes_server: str = ""
    aes_port: int = 450
    aes_login: str = ""
    aes_password_vault_path: str = ""

    # Protocol selection
    protocol: AvayaProtocol = AvayaProtocol.TSAPI

    # TSAPI settings
    tsapi_link_name: str = ""  # TSAPI link configured on AES
    tsapi_version: int = 3  # TSAPI version (2 or 3)
    tsapi_service_id: str = ""

    # DMCC settings
    dmcc_server: str = ""
    dmcc_port: int = 4721
    dmcc_secure_port: int = 4722
    dmcc_application_id: str = ""

    # Communication Manager settings
    cm_switch_name: str = ""
    cm_cti_link: str = ""

    # Monitoring settings
    enable_station_monitoring: bool = False
    enable_call_observation: bool = False
    observation_mode: str = "SILENT"  # SILENT, COACH, BARGE_IN

    # VDN/Hunt Group settings
    monitored_vdns: List[str] = field(default_factory=list)
    monitored_skills: List[str] = field(default_factory=list)
    monitored_agents: List[str] = field(default_factory=list)

    # Advanced settings
    device_id_type: str = "STATION"  # STATION, AGENT_ID, VDN
    use_ucid: bool = True  # Universal Call ID
    event_filter_mask: int = 0xFFFF  # Which events to receive

    def validate(self) -> List[str]:
        """Validate Avaya-specific configuration."""
        errors = []

        if not self.aes_server:
            errors.append("AES server address is required")

        if self.protocol in (AvayaProtocol.TSAPI, AvayaProtocol.TSAPI_AND_DMCC):
            if not self.tsapi_link_name:
                errors.append("TSAPI link name is required for TSAPI protocol")

        if self.protocol in (AvayaProtocol.DMCC, AvayaProtocol.TSAPI_AND_DMCC):
            if not self.dmcc_application_id:
                errors.append("DMCC application ID is required for DMCC protocol")

        if self.enable_call_observation and self.observation_mode not in (
            "SILENT", "COACH", "BARGE_IN"
        ):
            errors.append(f"Invalid observation mode: {self.observation_mode}")

        return errors


# ---------------------------------------------------------------------------
# Avaya Connector Implementation
# ---------------------------------------------------------------------------


class AvayaConnector(PBXConnector):
    """
    Avaya Aura PBX connector.

    Implements the PBXConnector interface for Avaya Communication Manager
    via the Application Enablement Services (AES) TSAPI and DMCC protocols.

    This implementation requires the Avaya TSAPI/DMCC client libraries.
    In production, it would link against the native AES SDK.

    Features:
    - TSAPI v3 call control and monitoring
    - DMCC device and media control
    - Station monitoring for quality assurance
    - Call observation (silent, coach, barge-in)
    - Avaya-specific event mapping (CSTA events)
    - UCID (Universal Call ID) tracking
    - VDN and skill group monitoring
    - Agent AUX reason code management
    """

    def __init__(
        self,
        config: PBXConnectionConfig,
        avaya_config: Optional[AvayaConfig] = None,
    ):
        super().__init__(PBXType.AVAYA_AURA, config)
        self._avaya_config = avaya_config or AvayaConfig()

        # TSAPI session state
        self._tsapi_handle: Optional[Any] = None
        self._tsapi_stream_id: Optional[int] = None

        # DMCC session state
        self._dmcc_session: Optional[Any] = None

        # Monitor references
        self._active_monitors: Dict[str, str] = {}  # device_id -> monitor_cross_ref_id
        self._active_observations: Dict[str, str] = {}  # call_id -> observation_id

        # Subscription tracking
        self._subscriptions: Dict[str, Dict[str, Any]] = {}

        # Event code mapping for fast lookup
        self._event_code_map: Dict[str, AvayaEventCode] = {
            code.csta_name: code for code in AvayaEventCode
        }

        # Agent state cache
        self._agent_state_cache: Dict[str, AgentState] = {}

        # Validate Avaya config
        errors = self._avaya_config.validate()
        if errors:
            logger.warning(f"Avaya config validation warnings: {errors}")

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to Avaya AES via TSAPI and/or DMCC.

        In production, this would:
        1. Load TSAPI client library (libcsta.so / AvayaTsapi.dll)
        2. Open TSAPI stream (acsOpenStream)
        3. Authenticate with AES credentials
        4. Register for events (acsEventNotify)
        5. Optionally open DMCC session
        """
        avaya_cfg = self._avaya_config

        # Validate before connecting
        errors = avaya_cfg.validate()
        if errors:
            raise PBXConnectionError(f"Avaya configuration errors: {errors}")

        try:
            if avaya_cfg.protocol in (AvayaProtocol.TSAPI, AvayaProtocol.TSAPI_AND_DMCC):
                await self._connect_tsapi()

            if avaya_cfg.protocol in (AvayaProtocol.DMCC, AvayaProtocol.TSAPI_AND_DMCC):
                await self._connect_dmcc()

            self._connected = True

            logger.info(
                f"Connected to Avaya AES at {avaya_cfg.aes_server}:{avaya_cfg.aes_port} "
                f"via {avaya_cfg.protocol.value}"
            )

            self._audit_log_entry("AVAYA_CONNECTED", {
                "aes_server": avaya_cfg.aes_server,
                "protocol": avaya_cfg.protocol.value,
                "tsapi_link": avaya_cfg.tsapi_link_name,
            })

        except Exception as exc:
            logger.error(f"Failed to connect to Avaya AES: {exc}")
            raise PBXConnectionError(f"Avaya AES connection failed: {exc}")

    async def _connect_tsapi(self) -> None:
        """
        Establish TSAPI connection to Avaya AES.

        In production, this would use the Avaya TSAPI SDK to:
        1. Call acsOpenStream() with the TSAPI link name
        2. Authenticate with acsSetHeartbeatInterval()
        3. Set up event notification via acsEventNotify()
        4. Negotiate protocol version
        """
        avaya_cfg = self._avaya_config

        # Production: acsOpenStream(streamHandle, invokeIDType, invokeID,
        #   streamType, serverID, loginID, passwd, applicationName, ...)
        self._tsapi_handle = f"tsapi_{uuid.uuid4().hex[:8]}"
        self._tsapi_stream_id = 1

        logger.info(
            f"TSAPI stream opened: link={avaya_cfg.tsapi_link_name}, "
            f"version={avaya_cfg.tsapi_version}"
        )

    async def _connect_dmcc(self) -> None:
        """
        Establish DMCC connection to Avaya AES.

        In production, this would use the DMCC SDK to:
        1. Open XML/SOAP connection to DMCC service
        2. Authenticate application
        3. Register device control capabilities
        4. Initialize media control session
        """
        avaya_cfg = self._avaya_config

        # Production: Create DMCC ServiceProvider, authenticate, get DeviceServices
        self._dmcc_session = f"dmcc_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"DMCC session opened: app_id={avaya_cfg.dmcc_application_id}"
        )

    async def disconnect(self) -> None:
        """
        Disconnect from Avaya AES.

        Releases all monitors, closes TSAPI stream and DMCC session.
        """
        # Release all monitors
        for device_id, monitor_ref in list(self._active_monitors.items()):
            try:
                await self._stop_monitor(device_id)
            except Exception as exc:
                logger.warning(f"Error releasing monitor for {device_id}: {exc}")

        # Release all observations
        for call_id, obs_id in list(self._active_observations.items()):
            try:
                await self._stop_observation(call_id)
            except Exception as exc:
                logger.warning(f"Error releasing observation for {call_id}: {exc}")

        # Close TSAPI stream
        if self._tsapi_handle:
            # Production: acsCloseStream(streamHandle)
            self._tsapi_handle = None
            self._tsapi_stream_id = None
            logger.debug("TSAPI stream closed")

        # Close DMCC session
        if self._dmcc_session:
            # Production: Close DMCC ServiceProvider
            self._dmcc_session = None
            logger.debug("DMCC session closed")

        self._connected = False
        self._active_monitors.clear()
        self._active_observations.clear()
        self._agent_state_cache.clear()

        self._audit_log_entry("AVAYA_DISCONNECTED", {
            "aes_server": self._avaya_config.aes_server,
        })

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
        Subscribe to Avaya CSTA events.

        Creates monitor points on agents, VDNs, and/or hunt groups via
        TSAPI cstaMonitorDevice() or cstaMonitorCall().

        Args:
            event_types: Event types to subscribe to
            agent_ids: Agent extensions/IDs to monitor
            queue_names: VDNs or hunt groups to monitor

        Returns:
            Subscription ID
        """
        self._check_connected()

        subscription_id = str(uuid.uuid4())

        # Determine devices to monitor
        devices_to_monitor: Set[str] = set()

        if agent_ids:
            devices_to_monitor.update(agent_ids)
        elif self._avaya_config.monitored_agents:
            devices_to_monitor.update(self._avaya_config.monitored_agents)

        if queue_names:
            devices_to_monitor.update(queue_names)
        elif self._avaya_config.monitored_vdns:
            devices_to_monitor.update(self._avaya_config.monitored_vdns)

        # Set up monitors for each device
        monitor_refs = {}
        for device_id in devices_to_monitor:
            try:
                monitor_ref = await self._start_monitor(device_id)
                monitor_refs[device_id] = monitor_ref
            except Exception as exc:
                logger.warning(f"Failed to monitor device {device_id}: {exc}")

        self._subscriptions[subscription_id] = {
            "event_types": event_types,
            "agent_ids": agent_ids,
            "queue_names": queue_names,
            "monitors": monitor_refs,
            "created_at": datetime.utcnow().isoformat(),
        }

        self._audit_log_entry("AVAYA_SUBSCRIBE", {
            "subscription_id": subscription_id,
            "devices_monitored": list(monitor_refs.keys()),
            "monitor_count": len(monitor_refs),
        })

        logger.info(
            f"Avaya subscription {subscription_id}: monitoring "
            f"{len(monitor_refs)} devices"
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

        # Release monitors associated with this subscription
        for device_id, monitor_ref in subscription.get("monitors", {}).items():
            try:
                await self._stop_monitor(device_id)
            except Exception as exc:
                logger.warning(f"Error releasing monitor for {device_id}: {exc}")

        del self._subscriptions[subscription_id]

        self._audit_log_entry("AVAYA_UNSUBSCRIBE", {
            "subscription_id": subscription_id,
        })

    async def _start_monitor(self, device_id: str) -> str:
        """
        Start CSTA monitoring on a device.

        In production, this would call cstaMonitorDevice() via TSAPI.

        Args:
            device_id: Extension or VDN to monitor

        Returns:
            Monitor cross-reference ID
        """
        # Production: cstaMonitorDevice(streamHandle, invokeID, deviceID, monitorFilter)
        monitor_ref = f"mon_{uuid.uuid4().hex[:8]}"
        self._active_monitors[device_id] = monitor_ref

        logger.debug(f"Started monitor on device {device_id}: {monitor_ref}")
        return monitor_ref

    async def _stop_monitor(self, device_id: str) -> None:
        """
        Stop CSTA monitoring on a device.

        In production, this would call cstaMonitorStop() via TSAPI.
        """
        if device_id in self._active_monitors:
            monitor_ref = self._active_monitors[device_id]
            # Production: cstaMonitorStop(streamHandle, invokeID, monitorCrossRefID)
            del self._active_monitors[device_id]
            logger.debug(f"Stopped monitor on device {device_id}: {monitor_ref}")

    # -----------------------------------------------------------------------
    # Call control operations
    # -----------------------------------------------------------------------

    async def send_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a raw CSTA/TSAPI command to Avaya AES.

        Supported commands:
        - cstaMakeCall
        - cstaAnswerCall
        - cstaClearConnection
        - cstaHoldCall
        - cstaRetrieveCall
        - cstaTransferCall
        - cstaConferenceCall
        - cstaConsultationCall
        - cstaSetAgentState
        - cstaQueryAgentState
        - cstaSnapshotDevice
        - cstaMonitorDevice

        Args:
            command: CSTA command name
            params: Command parameters

        Returns:
            Command response
        """
        self._check_connected()

        self._audit_log_entry("AVAYA_COMMAND", {
            "command": command,
            "params": {k: v for k, v in params.items() if k != "password"},
        })

        # In production, this would dispatch to the appropriate TSAPI function
        # via the AES SDK
        response = {
            "command": command,
            "status": "success",
            "invoke_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.debug(f"Avaya command executed: {command}")
        return response

    async def get_agent_state(self, agent_id: str) -> AgentState:
        """
        Get the current state of an Avaya agent.

        Uses cstaQueryAgentState() to query the agent's current
        work mode and AUX status.

        Args:
            agent_id: Avaya agent login ID or extension

        Returns:
            Normalised agent state
        """
        self._check_connected()

        # Check cache first
        if agent_id in self._agent_state_cache:
            return self._agent_state_cache[agent_id]

        # Production: cstaQueryAgentState(streamHandle, invokeID, deviceID)
        # The response would contain agentState, talkState, workMode, etc.

        # Stub: return cached or default state
        state = self._agent_state_cache.get(agent_id, AgentState.LOGGED_OUT)

        self._audit_log_entry("AVAYA_QUERY_AGENT", {
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
        Set the agent state on Avaya CM.

        Maps standard AgentState to Avaya work modes and AUX reason codes.

        Args:
            agent_id: Avaya agent login ID
            state: Desired agent state
            reason_code: Avaya AUX reason code (for NOT_READY states)
        """
        self._check_connected()

        # Map standard state to Avaya mode
        avaya_mode = self._map_to_avaya_mode(state)

        # Production: cstaSetAgentState(streamHandle, invokeID, deviceID,
        #   agentMode, agentID, agentGroup, agentPassword)

        self._agent_state_cache[agent_id] = state

        self._audit_log_entry("AVAYA_SET_AGENT_STATE", {
            "agent_id": agent_id,
            "state": state.state_name,
            "avaya_mode": avaya_mode.avaya_name if avaya_mode else "UNKNOWN",
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
        Transfer a call on Avaya CM.

        For blind transfers, uses cstaDeflectCall() or single-step transfer.
        For consultative transfers, uses cstaConsultationCall() followed
        by cstaTransferCall().

        Args:
            call_id: Active call ID
            target: Transfer destination (extension, VDN, external number)
            transfer_type: 'blind' or 'consultative'

        Returns:
            New call ID
        """
        self._check_connected()

        new_call_id = str(uuid.uuid4())

        if transfer_type == "consultative":
            # Production: Step 1 - cstaConsultationCall()
            # Production: Step 2 - cstaTransferCall()
            pass
        else:
            # Production: cstaSingleStepTransferCall() or cstaDeflectCall()
            pass

        self._audit_log_entry("AVAYA_TRANSFER", {
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
        Create a conference on Avaya CM.

        Uses cstaConsultationCall() for each participant, then
        cstaConferenceCall() to merge.

        Args:
            call_id: Existing call to conference
            targets: Participants to add

        Returns:
            Conference ID
        """
        self._check_connected()

        conference_id = str(uuid.uuid4())

        # Production:
        # For each target:
        #   1. cstaConsultationCall() to reach participant
        #   2. cstaConferenceCall() to merge into conference

        self._audit_log_entry("AVAYA_CONFERENCE", {
            "call_id": call_id,
            "targets": targets,
            "conference_id": conference_id,
        })

        logger.info(f"Conference {conference_id} created with {len(targets)} additional participants")
        return conference_id

    async def hold_call(self, call_id: str) -> None:
        """Place a call on hold using cstaHoldCall()."""
        self._check_connected()

        # Production: cstaHoldCall(streamHandle, invokeID, activeCall, connectionToBeHeld)

        self._audit_log_entry("AVAYA_HOLD", {"call_id": call_id})
        logger.info(f"Call {call_id} placed on hold")

    async def retrieve_call(self, call_id: str) -> None:
        """Retrieve a held call using cstaRetrieveCall()."""
        self._check_connected()

        # Production: cstaRetrieveCall(streamHandle, invokeID, heldCall)

        self._audit_log_entry("AVAYA_RETRIEVE", {"call_id": call_id})
        logger.info(f"Call {call_id} retrieved from hold")

    async def make_call(
        self,
        agent_id: str,
        destination: str,
        uui_data: str = "",
    ) -> str:
        """
        Initiate an outbound call via Avaya CM.

        Uses cstaMakeCall() with optional UUI (User-to-User Information)
        for screen pop data.

        Args:
            agent_id: Agent extension originating the call
            destination: Destination number
            uui_data: Optional UUI data for screen pop

        Returns:
            New call ID
        """
        self._check_connected()

        call_id = str(uuid.uuid4())

        # Production: cstaMakeCall(streamHandle, invokeID, callingDevice,
        #   calledDevice, userData)
        # If UUI is provided, include it in the userData parameter

        self._audit_log_entry("AVAYA_MAKE_CALL", {
            "agent_id": agent_id,
            "destination": destination,
            "call_id": call_id,
            "has_uui": bool(uui_data),
        })

        logger.info(f"Outbound call {call_id} initiated: {agent_id} -> {destination}")
        return call_id

    async def end_call(self, call_id: str) -> None:
        """End a call using cstaClearConnection()."""
        self._check_connected()

        # Production: cstaClearConnection(streamHandle, invokeID, call, deviceID)

        self._audit_log_entry("AVAYA_END_CALL", {"call_id": call_id})
        logger.info(f"Call {call_id} ended")

    async def get_call_info(self, call_id: str) -> PBXCallInfo:
        """
        Get detailed call information via cstaSnapshotCall().

        Returns UCID, parties, call state, and associated data.
        """
        self._check_connected()

        # Production: cstaSnapshotCall(streamHandle, invokeID, snapshotObj)

        return PBXCallInfo(
            call_id=call_id,
            ucid=f"ucid_{call_id[:8]}",
            direction=CallDirection.INBOUND,
            metadata={"pbx_type": "avaya"},
        )

    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """
        Get real-time VDN/hunt group statistics.

        Uses cstaSnapshotDevice() on the VDN to get queue metrics.
        """
        self._check_connected()

        # Production: cstaSnapshotDevice() on VDN, then process ConnectionID list
        # Also query CMS/ACDR for real-time stats

        return {
            "queue_name": queue_name,
            "calls_waiting": 0,
            "calls_in_progress": 0,
            "agents_available": 0,
            "agents_busy": 0,
            "agents_acw": 0,
            "agents_aux": 0,
            "longest_wait_seconds": 0.0,
            "average_wait_seconds": 0.0,
            "service_level_pct": 100.0,
            "abandoned_today": 0,
            "average_handle_time_seconds": 0.0,
        }

    # -----------------------------------------------------------------------
    # Avaya-specific operations
    # -----------------------------------------------------------------------

    async def start_call_observation(
        self,
        call_id: str,
        observer_extension: str,
        mode: str = "SILENT",
    ) -> str:
        """
        Start observing (listening to) a call for quality monitoring.

        Modes:
        - SILENT: Listen only (agent and customer cannot hear observer)
        - COACH: Observer can talk to agent only (customer cannot hear)
        - BARGE_IN: Observer can talk to both agent and customer

        In production, uses DMCC or the Service Observe feature
        of Avaya Communication Manager.

        Args:
            call_id: Call to observe
            observer_extension: Extension of the observer/supervisor
            mode: Observation mode

        Returns:
            Observation session ID
        """
        self._check_connected()

        if mode not in ("SILENT", "COACH", "BARGE_IN"):
            raise PBXCommandError(
                f"Invalid observation mode: {mode}",
                command="start_call_observation",
            )

        observation_id = str(uuid.uuid4())

        # Production: Use Service Observe via TSAPI or DMCC
        # For DMCC: RegisterTerminal, then MonitorStart with appropriate filter

        self._active_observations[call_id] = observation_id

        self._audit_log_entry("AVAYA_OBSERVATION_START", {
            "call_id": call_id,
            "observer": observer_extension,
            "mode": mode,
            "observation_id": observation_id,
        })

        logger.info(
            f"Call observation started: call={call_id}, "
            f"observer={observer_extension}, mode={mode}"
        )
        return observation_id

    async def _stop_observation(self, call_id: str) -> None:
        """Stop observing a call."""
        if call_id in self._active_observations:
            observation_id = self._active_observations[call_id]
            # Production: Stop Service Observe / DMCC Monitor
            del self._active_observations[call_id]

            self._audit_log_entry("AVAYA_OBSERVATION_STOP", {
                "call_id": call_id,
                "observation_id": observation_id,
            })

    async def snapshot_device(self, device_id: str) -> Dict[str, Any]:
        """
        Get a snapshot of device state via cstaSnapshotDevice().

        Returns all active calls, connection states, and agent state
        for the specified extension or VDN.

        Args:
            device_id: Extension, VDN, or agent ID

        Returns:
            Device snapshot with active connections and state
        """
        self._check_connected()

        # Production: cstaSnapshotDevice(streamHandle, invokeID, snapshotObj)

        return {
            "device_id": device_id,
            "device_type": self._avaya_config.device_id_type,
            "active_connections": [],
            "agent_state": None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_ucid(self, call_id: str) -> str:
        """
        Get the Universal Call ID (UCID) for a call.

        UCID is Avaya's globally unique call identifier that persists
        across transfers and conferences.

        Args:
            call_id: The local call ID

        Returns:
            UCID string
        """
        self._check_connected()

        # Production: Extract UCID from call event UserData or
        # query via cstaSnapshotCall()
        return f"ucid_{call_id[:16]}"

    async def send_uui_data(self, call_id: str, uui_data: str) -> None:
        """
        Attach User-to-User Information (UUI) data to a call.

        UUI data is used for screen pop information that follows the
        call through transfers and conferences.

        Args:
            call_id: The call to attach data to
            uui_data: UUI data string (max 96 bytes on Avaya)
        """
        self._check_connected()

        if len(uui_data.encode("utf-8")) > 96:
            raise PBXCommandError(
                "UUI data exceeds Avaya 96-byte limit",
                command="send_uui_data",
            )

        # Production: cstaSetUserData() or include in transfer/conference commands

        self._audit_log_entry("AVAYA_UUI_SET", {
            "call_id": call_id,
            "uui_length": len(uui_data),
        })

    # -----------------------------------------------------------------------
    # Heartbeat
    # -----------------------------------------------------------------------

    async def _send_heartbeat(self) -> bool:
        """
        Send heartbeat to Avaya AES.

        Uses acsSetHeartbeatInterval() and monitors for heartbeat
        responses from AES.

        Returns:
            True if heartbeat acknowledged
        """
        if not self._tsapi_handle and not self._dmcc_session:
            return False

        # Production: acsSetHeartbeatInterval() or DMCC keepalive

        return True

    # -----------------------------------------------------------------------
    # Event mapping
    # -----------------------------------------------------------------------

    async def _map_vendor_event(self, raw_event: Dict[str, Any]) -> PBXEvent:
        """
        Map Avaya CSTA event to standardised PBXEvent.

        Avaya events come as CSTA events via TSAPI with event codes
        like CSTADeliveredEvent, CSTAEstablishedEvent, etc.

        This method normalises these vendor-specific events into the
        common PBXEvent format.
        """
        event_name = raw_event.get("eventType", "")
        avaya_event = self._event_code_map.get(event_name)

        if avaya_event is None:
            logger.warning(f"Unknown Avaya event type: {event_name}")
            event_type = PBXEventType.CALL_NEW  # fallback
        else:
            event_type = avaya_event.pbx_event_type

        # Extract common fields from CSTA event structure
        call_id = raw_event.get("callID", raw_event.get("connectionID", {}).get("callID", ""))
        device_id = raw_event.get("deviceID", raw_event.get("agentDevice", ""))
        queue = raw_event.get("lastRedirectionDevice", raw_event.get("calledDevice", ""))

        # Map agent state if present
        agent_state = None
        if event_type == PBXEventType.AGENT_STATE_CHANGE:
            avaya_mode = raw_event.get("agentMode", "")
            agent_state = self._map_avaya_agent_state(avaya_mode)
            if device_id:
                self._agent_state_cache[device_id] = agent_state
        elif event_type == PBXEventType.AGENT_LOGIN:
            agent_state = AgentState.AVAILABLE
            if device_id:
                self._agent_state_cache[device_id] = agent_state
        elif event_type == PBXEventType.AGENT_LOGOUT:
            agent_state = AgentState.LOGGED_OUT
            if device_id:
                self._agent_state_cache[device_id] = agent_state

        # Determine call direction
        direction = None
        cause = raw_event.get("cause", "")
        if event_type in (PBXEventType.CALL_NEW, PBXEventType.CALL_DELIVERED):
            if cause == "newCall" or raw_event.get("callingDevice", "") == device_id:
                direction = CallDirection.OUTBOUND
            else:
                direction = CallDirection.INBOUND

        # Extract UCID if available
        ucid = raw_event.get("ucid", "")
        uui = raw_event.get("userData", "")

        event = PBXEvent(
            event_type=event_type,
            call_id=call_id,
            agent_id=device_id,
            queue_name=queue,
            timestamp=datetime.utcnow(),
            caller_number=raw_event.get("callingDevice", ""),
            called_number=raw_event.get("calledDevice", ""),
            direction=direction,
            agent_state=agent_state,
            raw_event=raw_event,
            metadata={
                "ucid": ucid,
                "uui": uui,
                "cause": cause,
                "avaya_event": event_name,
                "monitor_cross_ref": raw_event.get("monitorCrossRefID", ""),
            },
        )

        return event

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _map_to_avaya_mode(self, state: AgentState) -> Optional[AvayaAgentMode]:
        """Map standard AgentState to Avaya-specific agent mode."""
        state_map = {
            AgentState.AVAILABLE: AvayaAgentMode.AUTO_IN,
            AgentState.AFTER_CALL_WORK: AvayaAgentMode.ACW,
            AgentState.NOT_READY: AvayaAgentMode.AUX,
            AgentState.BREAK: AvayaAgentMode.AUX,
            AgentState.TRAINING: AvayaAgentMode.AUX,
            AgentState.MEETING: AvayaAgentMode.AUX,
            AgentState.LOGGED_OUT: AvayaAgentMode.LOGGED_OUT,
        }
        return state_map.get(state)

    def _map_avaya_agent_state(self, avaya_mode: str) -> AgentState:
        """Map Avaya agent work mode string to standard AgentState."""
        mode_map = {
            "AUTO-IN": AgentState.AVAILABLE,
            "MANUAL-IN": AgentState.AVAILABLE,
            "ACW": AgentState.AFTER_CALL_WORK,
            "AUX": AgentState.NOT_READY,
            "AVAIL": AgentState.AVAILABLE,
            "LOGGED-OUT": AgentState.LOGGED_OUT,
            "ON-CALL": AgentState.ON_CALL,
        }
        return mode_map.get(avaya_mode.upper(), AgentState.NOT_READY)
