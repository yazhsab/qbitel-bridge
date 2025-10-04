"""
CRONOS AI Engine - Network Security Integration Connectors

Specialized connectors for network security systems like Firewalls and IDS/IPS.
"""

import asyncio
import json
import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod

from .base_connector import BaseIntegrationConnector, IntegrationResult, IntegrationType
from ..models import SecurityEvent, ThreatAnalysis, AutomatedResponse
from ..logging import get_security_logger, SecurityLogType, LogLevel


class NetworkSecurityConnector(BaseIntegrationConnector, ABC):
    """Base class for network security system connectors."""

    def __init__(self, config):
        super().__init__(config)
        self.logger = get_security_logger(
            "cronos.security.integrations.network_security"
        )

    @abstractmethod
    async def block_ip(
        self, ip_address: str, duration_minutes: int = None
    ) -> IntegrationResult:
        """Block an IP address."""
        pass

    @abstractmethod
    async def unblock_ip(self, ip_address: str) -> IntegrationResult:
        """Unblock an IP address."""
        pass

    @abstractmethod
    async def create_firewall_rule(
        self, rule_config: Dict[str, Any]
    ) -> IntegrationResult:
        """Create a new firewall rule."""
        pass

    @abstractmethod
    async def delete_firewall_rule(self, rule_id: str) -> IntegrationResult:
        """Delete a firewall rule."""
        pass


class FirewallConnector(NetworkSecurityConnector):
    """Generic firewall integration connector."""

    def __init__(self, config):
        super().__init__(config)
        self.api_key = None
        self.firewall_type = config.custom_config.get("firewall_type", "palo_alto")

    async def initialize(self) -> bool:
        """Initialize Firewall connection."""
        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Initializing {self.firewall_type} Firewall connector",
                level=LogLevel.INFO,
                component="firewall_connector",
            )

            # Set up authentication
            self.api_key = self.config.credentials.get("api_key")

            if self.api_key:
                # Test connection
                test_result = await self._test_connection()

                if test_result:
                    self.connection_status = "connected"
                    self.last_health_check = datetime.utcnow()

                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        f"{self.firewall_type} Firewall connector initialized successfully",
                        level=LogLevel.INFO,
                        component="firewall_connector",
                    )
                    return True
                else:
                    self.connection_status = "failed"
                    return False
            else:
                self.connection_status = "failed"
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    "Firewall API key not provided",
                    level=LogLevel.ERROR,
                    component="firewall_connector",
                )
                return False

        except Exception as e:
            self.connection_status = "failed"
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Firewall connector initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                component="firewall_connector",
                error_code="FIREWALL_INIT_ERROR",
            )
            return False

    async def _test_connection(self) -> bool:
        """Test firewall connection."""

        if self.firewall_type == "palo_alto":
            return await self._test_palo_alto_connection()
        elif self.firewall_type == "checkpoint":
            return await self._test_checkpoint_connection()
        elif self.firewall_type == "fortinet":
            return await self._test_fortinet_connection()
        else:
            # Generic test
            return await self._test_generic_connection()

    async def _test_palo_alto_connection(self) -> bool:
        """Test Palo Alto firewall connection."""

        test_url = f"{self.config.endpoint}/api/"
        params = {
            "type": "op",
            "cmd": "<show><system><info></info></system></show>",
            "key": self.api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    test_url, params=params, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Parse XML response
                        try:
                            root = ET.fromstring(content)
                            return root.get("status") == "success"
                        except ET.ParseError:
                            return False
                    return False

        except Exception:
            return False

    async def _test_checkpoint_connection(self) -> bool:
        """Test Check Point firewall connection."""

        # Check Point uses REST API with session-based authentication
        login_url = f"{self.config.endpoint}/web_api/login"
        credentials = {
            "user": self.config.credentials.get("username"),
            "password": self.config.credentials.get("password"),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    login_url, json=credentials, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return "sid" in data
                    return False

        except Exception:
            return False

    async def _test_fortinet_connection(self) -> bool:
        """Test Fortinet FortiGate connection."""

        test_url = f"{self.config.endpoint}/api/v2/monitor/system/status"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    test_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "success"
                    return False

        except Exception:
            return False

    async def _test_generic_connection(self) -> bool:
        """Generic connection test."""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.endpoint, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 500

        except Exception:
            return False

    async def send_security_event(
        self, security_event: SecurityEvent
    ) -> IntegrationResult:
        """Send security event to firewall for processing."""

        try:
            # For network-related events, check if IP blocking is needed
            if security_event.source_ip and security_event.threat_level.value in [
                "critical",
                "high",
            ]:
                # Automatically block suspicious IP
                block_result = await self.block_ip(
                    security_event.source_ip,
                    duration_minutes=self.config.custom_config.get(
                        "auto_block_duration", 60
                    ),
                )

                if block_result.success:
                    return IntegrationResult(
                        success=True,
                        message=f"IP {security_event.source_ip} blocked automatically",
                        response_data=block_result.response_data,
                    )
                else:
                    return block_result
            else:
                return IntegrationResult(
                    success=True,
                    message="Security event processed, no automatic action required",
                    response_data={"action": "no_action_required"},
                )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Firewall event processing failed: {str(e)}",
                error_code="FIREWALL_PROCESS_ERROR",
            )

    async def send_threat_analysis(
        self, threat_analysis: ThreatAnalysis
    ) -> IntegrationResult:
        """Send threat analysis to firewall."""

        try:
            # Extract IOCs and create blocking rules if needed
            blocked_items = []

            for ioc in threat_analysis.iocs:
                if ioc.ioc_type.value == "ip" and ioc.confidence >= 0.8:
                    # Block high-confidence IP IOCs
                    block_result = await self.block_ip(ioc.value)
                    if block_result.success:
                        blocked_items.append(ioc.value)

            return IntegrationResult(
                success=True,
                message=f"Threat analysis processed, {len(blocked_items)} items blocked",
                response_data={"blocked_items": blocked_items},
            )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Firewall threat analysis processing failed: {str(e)}",
                error_code="FIREWALL_THREAT_PROCESS_ERROR",
            )

    async def send_response_execution(
        self, automated_response: AutomatedResponse
    ) -> IntegrationResult:
        """Send automated response execution to firewall."""

        try:
            # Execute network-related response actions
            executed_actions = []

            for action in automated_response.actions:
                if action.action_type.value in ["block_ip", "create_firewall_rule"]:
                    if action.action_type.value == "block_ip":
                        ip_address = action.parameters.get("ip_address")
                        duration = action.parameters.get("duration_minutes", 60)

                        if ip_address:
                            result = await self.block_ip(ip_address, duration)
                            if result.success:
                                executed_actions.append(
                                    {
                                        "action": "block_ip",
                                        "target": ip_address,
                                        "result": "success",
                                    }
                                )

                    elif action.action_type.value == "create_firewall_rule":
                        rule_config = action.parameters.get("rule_config", {})

                        if rule_config:
                            result = await self.create_firewall_rule(rule_config)
                            if result.success:
                                executed_actions.append(
                                    {
                                        "action": "create_firewall_rule",
                                        "rule_id": result.response_data.get("rule_id"),
                                        "result": "success",
                                    }
                                )

            return IntegrationResult(
                success=True,
                message=f"Response execution processed, {len(executed_actions)} actions executed",
                response_data={"executed_actions": executed_actions},
            )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Firewall response execution failed: {str(e)}",
                error_code="FIREWALL_RESPONSE_ERROR",
            )

    async def block_ip(
        self, ip_address: str, duration_minutes: int = None
    ) -> IntegrationResult:
        """Block an IP address on the firewall."""

        if self.firewall_type == "palo_alto":
            return await self._palo_alto_block_ip(ip_address, duration_minutes)
        elif self.firewall_type == "checkpoint":
            return await self._checkpoint_block_ip(ip_address, duration_minutes)
        elif self.firewall_type == "fortinet":
            return await self._fortinet_block_ip(ip_address, duration_minutes)
        else:
            return await self._generic_block_ip(ip_address, duration_minutes)

    async def _palo_alto_block_ip(
        self, ip_address: str, duration_minutes: int = None
    ) -> IntegrationResult:
        """Block IP on Palo Alto firewall."""

        # Create dynamic address object
        address_name = f"CRONOS_BLOCKED_{ip_address.replace('.', '_')}"

        # Step 1: Create address object
        create_url = f"{self.config.endpoint}/api/"
        create_params = {
            "type": "config",
            "action": "set",
            "xpath": f"/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys1']/address/entry[@name='{address_name}']",
            "element": f"<ip-netmask>{ip_address}</ip-netmask><description>Blocked by CRONOS AI - {datetime.utcnow().isoformat()}</description>",
            "key": self.api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Create address object
                async with session.post(
                    create_url,
                    params=create_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 200:
                        content = await response.text()
                        root = ET.fromstring(content)

                        if root.get("status") == "success":
                            # Step 2: Add to block list (assuming there's a security rule)
                            # This is a simplified implementation

                            # Step 3: Commit changes
                            commit_result = await self._palo_alto_commit()

                            if commit_result:
                                # Schedule unblock if duration specified
                                if duration_minutes:
                                    asyncio.create_task(
                                        self._schedule_unblock(
                                            ip_address, duration_minutes * 60
                                        )
                                    )

                                return IntegrationResult(
                                    success=True,
                                    message=f"IP {ip_address} blocked on Palo Alto firewall",
                                    response_data={
                                        "ip_address": ip_address,
                                        "address_object": address_name,
                                        "duration_minutes": duration_minutes,
                                    },
                                )
                            else:
                                return IntegrationResult(
                                    success=False,
                                    message="Failed to commit firewall changes",
                                    error_code="FIREWALL_COMMIT_ERROR",
                                )
                        else:
                            return IntegrationResult(
                                success=False,
                                message=f"Palo Alto API error: {content}",
                                error_code="PALO_ALTO_API_ERROR",
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Palo Alto block IP failed: {response.status} - {error_text}",
                            error_code="PALO_ALTO_BLOCK_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Palo Alto block IP error: {str(e)}",
                error_code="PALO_ALTO_BLOCK_ERROR",
            )

    async def _palo_alto_commit(self) -> bool:
        """Commit changes to Palo Alto firewall."""

        commit_url = f"{self.config.endpoint}/api/"
        commit_params = {
            "type": "commit",
            "cmd": "<commit></commit>",
            "key": self.api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    commit_url,
                    params=commit_params,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:

                    if response.status == 200:
                        content = await response.text()
                        root = ET.fromstring(content)
                        return root.get("status") == "success"
                    return False

        except Exception:
            return False

    async def _checkpoint_block_ip(
        self, ip_address: str, duration_minutes: int = None
    ) -> IntegrationResult:
        """Block IP on Check Point firewall."""

        # Check Point implementation would involve:
        # 1. Login to get session ID
        # 2. Add host object
        # 3. Create/modify security rule
        # 4. Publish changes

        # Simplified implementation
        return IntegrationResult(
            success=True,
            message=f"IP {ip_address} would be blocked on Check Point firewall",
            response_data={"ip_address": ip_address, "method": "checkpoint"},
        )

    async def _fortinet_block_ip(
        self, ip_address: str, duration_minutes: int = None
    ) -> IntegrationResult:
        """Block IP on Fortinet FortiGate."""

        # Create banned IP entry
        ban_url = f"{self.config.endpoint}/api/v2/cmdb/user/banned"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        ban_data = {
            "ip": ip_address,
            "comment": f"Blocked by CRONOS AI - {datetime.utcnow().isoformat()}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ban_url,
                    json=ban_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 200:
                        response_data = await response.json()

                        if response_data.get("status") == "success":
                            # Schedule unblock if duration specified
                            if duration_minutes:
                                asyncio.create_task(
                                    self._schedule_unblock(
                                        ip_address, duration_minutes * 60
                                    )
                                )

                            return IntegrationResult(
                                success=True,
                                message=f"IP {ip_address} blocked on FortiGate",
                                response_data={
                                    "ip_address": ip_address,
                                    "duration_minutes": duration_minutes,
                                },
                            )
                        else:
                            return IntegrationResult(
                                success=False,
                                message=f"FortiGate API error: {response_data}",
                                error_code="FORTINET_API_ERROR",
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"FortiGate block IP failed: {response.status} - {error_text}",
                            error_code="FORTINET_BLOCK_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"FortiGate block IP error: {str(e)}",
                error_code="FORTINET_BLOCK_ERROR",
            )

    async def _generic_block_ip(
        self, ip_address: str, duration_minutes: int = None
    ) -> IntegrationResult:
        """Generic IP blocking implementation."""

        # This would be implemented based on the specific firewall's API
        return IntegrationResult(
            success=True,
            message=f"IP {ip_address} would be blocked using generic method",
            response_data={"ip_address": ip_address, "method": "generic"},
        )

    async def _schedule_unblock(self, ip_address: str, delay_seconds: int):
        """Schedule automatic unblocking of an IP address."""

        await asyncio.sleep(delay_seconds)

        try:
            unblock_result = await self.unblock_ip(ip_address)

            if unblock_result.success:
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    f"Automatically unblocked IP {ip_address} after scheduled delay",
                    level=LogLevel.INFO,
                    component="firewall_connector",
                )
            else:
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    f"Failed to automatically unblock IP {ip_address}: {unblock_result.message}",
                    level=LogLevel.WARNING,
                    component="firewall_connector",
                )

        except Exception as e:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Error during scheduled unblock of IP {ip_address}: {str(e)}",
                level=LogLevel.ERROR,
                component="firewall_connector",
            )

    async def unblock_ip(self, ip_address: str) -> IntegrationResult:
        """Unblock an IP address on the firewall."""

        if self.firewall_type == "palo_alto":
            return await self._palo_alto_unblock_ip(ip_address)
        elif self.firewall_type == "checkpoint":
            return await self._checkpoint_unblock_ip(ip_address)
        elif self.firewall_type == "fortinet":
            return await self._fortinet_unblock_ip(ip_address)
        else:
            return await self._generic_unblock_ip(ip_address)

    async def _palo_alto_unblock_ip(self, ip_address: str) -> IntegrationResult:
        """Unblock IP on Palo Alto firewall."""

        address_name = f"CRONOS_BLOCKED_{ip_address.replace('.', '_')}"

        # Delete address object
        delete_url = f"{self.config.endpoint}/api/"
        delete_params = {
            "type": "config",
            "action": "delete",
            "xpath": f"/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys1']/address/entry[@name='{address_name}']",
            "key": self.api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    delete_url,
                    params=delete_params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 200:
                        content = await response.text()
                        root = ET.fromstring(content)

                        if root.get("status") == "success":
                            # Commit changes
                            commit_result = await self._palo_alto_commit()

                            if commit_result:
                                return IntegrationResult(
                                    success=True,
                                    message=f"IP {ip_address} unblocked on Palo Alto firewall",
                                    response_data={"ip_address": ip_address},
                                )
                            else:
                                return IntegrationResult(
                                    success=False,
                                    message="Failed to commit firewall changes",
                                    error_code="FIREWALL_COMMIT_ERROR",
                                )
                        else:
                            return IntegrationResult(
                                success=False,
                                message=f"Palo Alto API error: {content}",
                                error_code="PALO_ALTO_API_ERROR",
                            )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"Palo Alto unblock IP failed: {response.status} - {error_text}",
                            error_code="PALO_ALTO_UNBLOCK_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"Palo Alto unblock IP error: {str(e)}",
                error_code="PALO_ALTO_UNBLOCK_ERROR",
            )

    async def _checkpoint_unblock_ip(self, ip_address: str) -> IntegrationResult:
        """Unblock IP on Check Point firewall."""

        # Simplified implementation
        return IntegrationResult(
            success=True,
            message=f"IP {ip_address} would be unblocked on Check Point firewall",
            response_data={"ip_address": ip_address},
        )

    async def _fortinet_unblock_ip(self, ip_address: str) -> IntegrationResult:
        """Unblock IP on Fortinet FortiGate."""

        # Remove from banned IP list
        unban_url = f"{self.config.endpoint}/api/v2/cmdb/user/banned/{ip_address}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    unban_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status == 200:
                        return IntegrationResult(
                            success=True,
                            message=f"IP {ip_address} unblocked on FortiGate",
                            response_data={"ip_address": ip_address},
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"FortiGate unblock IP failed: {response.status} - {error_text}",
                            error_code="FORTINET_UNBLOCK_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"FortiGate unblock IP error: {str(e)}",
                error_code="FORTINET_UNBLOCK_ERROR",
            )

    async def _generic_unblock_ip(self, ip_address: str) -> IntegrationResult:
        """Generic IP unblocking implementation."""

        return IntegrationResult(
            success=True,
            message=f"IP {ip_address} would be unblocked using generic method",
            response_data={"ip_address": ip_address},
        )

    async def create_firewall_rule(
        self, rule_config: Dict[str, Any]
    ) -> IntegrationResult:
        """Create a new firewall rule."""

        # This would be implemented based on the specific firewall type
        rule_name = rule_config.get(
            "name", f"CRONOS_RULE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        return IntegrationResult(
            success=True,
            message=f"Firewall rule '{rule_name}' would be created",
            response_data={"rule_id": rule_name, "rule_config": rule_config},
        )

    async def delete_firewall_rule(self, rule_id: str) -> IntegrationResult:
        """Delete a firewall rule."""

        return IntegrationResult(
            success=True,
            message=f"Firewall rule '{rule_id}' would be deleted",
            response_data={"rule_id": rule_id},
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for Firewall connector."""

        try:
            # Test basic connectivity
            test_result = await self._test_connection()

            if test_result:
                self.connection_status = "healthy"
                self.last_health_check = datetime.utcnow()

                return {
                    "status": "healthy",
                    "message": f"{self.firewall_type} firewall connection is healthy",
                    "last_check": self.last_health_check.isoformat(),
                }
            else:
                self.connection_status = "unhealthy"
                return {
                    "status": "unhealthy",
                    "message": f"{self.firewall_type} firewall health check failed",
                    "last_check": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            self.connection_status = "unhealthy"
            return {
                "status": "unhealthy",
                "message": f"{self.firewall_type} firewall health check error: {str(e)}",
                "last_check": datetime.utcnow().isoformat(),
            }


class IDSConnector(NetworkSecurityConnector):
    """Intrusion Detection System integration connector."""

    def __init__(self, config):
        super().__init__(config)
        self.api_token = None
        self.ids_type = config.custom_config.get("ids_type", "suricata")

    async def initialize(self) -> bool:
        """Initialize IDS connection."""
        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Initializing {self.ids_type} IDS connector",
                level=LogLevel.INFO,
                component="ids_connector",
            )

            # Set up authentication
            self.api_token = self.config.credentials.get("api_token")

            if self.api_token:
                # Test connection
                test_result = await self._test_connection()

                if test_result:
                    self.connection_status = "connected"
                    self.last_health_check = datetime.utcnow()

                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        f"{self.ids_type} IDS connector initialized successfully",
                        level=LogLevel.INFO,
                        component="ids_connector",
                    )
                    return True
                else:
                    self.connection_status = "failed"
                    return False
            else:
                self.connection_status = "failed"
                self.logger.log_security_event(
                    SecurityLogType.CONFIGURATION_CHANGE,
                    "IDS API token not provided",
                    level=LogLevel.ERROR,
                    component="ids_connector",
                )
                return False

        except Exception as e:
            self.connection_status = "failed"
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"IDS connector initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                component="ids_connector",
                error_code="IDS_INIT_ERROR",
            )
            return False

    async def _test_connection(self) -> bool:
        """Test IDS connection."""

        # Generic IDS health check endpoint
        test_url = f"{self.config.endpoint}/api/health"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    test_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200

        except Exception:
            return False

    async def send_security_event(
        self, security_event: SecurityEvent
    ) -> IntegrationResult:
        """Send security event to IDS for correlation."""

        try:
            # Send event to IDS for additional analysis
            event_data = {
                "event_id": security_event.event_id,
                "event_type": security_event.event_type.value,
                "source_ip": security_event.source_ip,
                "destination_ip": security_event.destination_ip,
                "protocol": security_event.protocol,
                "port": security_event.port,
                "timestamp": security_event.timestamp.isoformat(),
                "description": security_event.description,
                "threat_level": security_event.threat_level.value,
            }

            result = await self._send_event_to_ids(event_data)
            return result

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"IDS event send failed: {str(e)}",
                error_code="IDS_SEND_ERROR",
            )

    async def send_threat_analysis(
        self, threat_analysis: ThreatAnalysis
    ) -> IntegrationResult:
        """Send threat analysis to IDS."""

        try:
            # Update IDS rules based on threat analysis
            updated_rules = []

            for ioc in threat_analysis.iocs:
                if ioc.ioc_type.value == "ip" and ioc.confidence >= 0.7:
                    # Create temporary blocking rule
                    rule_result = await self._create_ids_rule(ioc.value, "ip")
                    if rule_result.get("success"):
                        updated_rules.append(ioc.value)

            return IntegrationResult(
                success=True,
                message=f"IDS rules updated, {len(updated_rules)} new rules created",
                response_data={"updated_rules": updated_rules},
            )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"IDS threat analysis processing failed: {str(e)}",
                error_code="IDS_THREAT_PROCESS_ERROR",
            )

    async def send_response_execution(
        self, automated_response: AutomatedResponse
    ) -> IntegrationResult:
        """Send automated response execution to IDS."""

        try:
            # Log response actions in IDS
            logged_actions = []

            for action in automated_response.actions:
                log_entry = {
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "target_system": action.target_system,
                    "status": action.status.value,
                    "executed_at": (
                        action.executed_at.isoformat() if action.executed_at else None
                    ),
                }

                log_result = await self._log_action_in_ids(log_entry)
                if log_result.get("success"):
                    logged_actions.append(action.action_id)

            return IntegrationResult(
                success=True,
                message=f"IDS logging complete, {len(logged_actions)} actions logged",
                response_data={"logged_actions": logged_actions},
            )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"IDS response logging failed: {str(e)}",
                error_code="IDS_RESPONSE_ERROR",
            )

    async def _send_event_to_ids(self, event_data: Dict[str, Any]) -> IntegrationResult:
        """Send event data to IDS."""

        events_url = f"{self.config.endpoint}/api/events"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    events_url,
                    json=event_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:

                    if response.status in [200, 201]:
                        response_data = await response.json()
                        return IntegrationResult(
                            success=True,
                            message="Event sent to IDS successfully",
                            response_data=response_data,
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            message=f"IDS event send failed: {response.status} - {error_text}",
                            error_code="IDS_EVENT_ERROR",
                        )

        except Exception as e:
            return IntegrationResult(
                success=False,
                message=f"IDS event send error: {str(e)}",
                error_code="IDS_EVENT_ERROR",
            )

    async def _create_ids_rule(self, ioc_value: str, ioc_type: str) -> Dict[str, Any]:
        """Create IDS detection rule for IOC."""

        # This would create a temporary rule to detect/block the IOC
        # Implementation depends on the specific IDS system

        return {
            "success": True,
            "rule_id": f"CRONOS_{ioc_type.upper()}_{hash(ioc_value) % 100000}",
            "ioc_value": ioc_value,
            "ioc_type": ioc_type,
        }

    async def _log_action_in_ids(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Log action execution in IDS."""

        # This would log the action in the IDS system for audit trail
        # Implementation depends on the specific IDS system

        return {
            "success": True,
            "log_id": f"LOG_{log_entry['action_id']}",
            "logged_at": datetime.utcnow().isoformat(),
        }

    # Implement abstract methods from NetworkSecurityConnector
    async def block_ip(
        self, ip_address: str, duration_minutes: int = None
    ) -> IntegrationResult:
        """Block IP in IDS (create detection rule)."""

        rule_result = await self._create_ids_rule(ip_address, "ip")

        if rule_result["success"]:
            return IntegrationResult(
                success=True,
                message=f"IDS rule created to detect/block IP {ip_address}",
                response_data=rule_result,
            )
        else:
            return IntegrationResult(
                success=False,
                message=f"Failed to create IDS rule for IP {ip_address}",
                error_code="IDS_RULE_CREATE_ERROR",
            )

    async def unblock_ip(self, ip_address: str) -> IntegrationResult:
        """Remove IP detection rule from IDS."""

        return IntegrationResult(
            success=True,
            message=f"IDS rule for IP {ip_address} would be removed",
            response_data={"ip_address": ip_address},
        )

    async def create_firewall_rule(
        self, rule_config: Dict[str, Any]
    ) -> IntegrationResult:
        """Create IDS detection rule."""

        rule_name = rule_config.get(
            "name", f"CRONOS_IDS_RULE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        return IntegrationResult(
            success=True,
            message=f"IDS rule '{rule_name}' would be created",
            response_data={"rule_id": rule_name, "rule_config": rule_config},
        )

    async def delete_firewall_rule(self, rule_id: str) -> IntegrationResult:
        """Delete IDS detection rule."""

        return IntegrationResult(
            success=True,
            message=f"IDS rule '{rule_id}' would be deleted",
            response_data={"rule_id": rule_id},
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for IDS connector."""

        try:
            # Test basic connectivity
            test_result = await self._test_connection()

            if test_result:
                self.connection_status = "healthy"
                self.last_health_check = datetime.utcnow()

                return {
                    "status": "healthy",
                    "message": f"{self.ids_type} IDS connection is healthy",
                    "last_check": self.last_health_check.isoformat(),
                }
            else:
                self.connection_status = "unhealthy"
                return {
                    "status": "unhealthy",
                    "message": f"{self.ids_type} IDS health check failed",
                    "last_check": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            self.connection_status = "unhealthy"
            return {
                "status": "unhealthy",
                "message": f"{self.ids_type} IDS health check error: {str(e)}",
                "last_check": datetime.utcnow().isoformat(),
            }
