"""
CTI (Computer Telephony Integration) Message Generator

Generates realistic CTI protocol messages for training the protocol
discovery and field detection models.

Supports three CTI protocol variants:
1. CSTA XML (ECMA-269): Standard XML-based telephony events
2. TSAPI SOAP: Avaya TSAPI-style SOAP/XML messages
3. Finesse REST: Cisco Finesse-style REST/JSON messages

Message types generated:
- csta_make_call: CSTA MakeCall request
- csta_answer_call: CSTA AnswerCall request
- csta_clear_call: CSTA ClearCall request
- csta_hold_call: CSTA HoldCall request
- csta_transfer_call: CSTA TransferCall request
- csta_monitor_start: CSTA MonitorStart request
- csta_event_delivered: CSTA DeliveredEvent notification
- tsapi_request: TSAPI SOAP envelope request
- finesse_dialog: Cisco Finesse dialog JSON

State machines per cti_message.py:
- Call states: IDLE -> RINGING -> CONNECTED -> HELD/TRANSFERRING/WRAPPING_UP -> DISCONNECTED
- Agent states: LOGGED_OUT -> READY -> ON_CALL -> WRAPPING -> READY
"""

import json
import random
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CTIGenerator:
    """Generate realistic CTI protocol messages for ML training."""

    # Call states (from cti_message.py)
    CALL_STATES = [
        "IDLE", "RINGING", "CONNECTED", "HELD",
        "TRANSFERRING", "CONFERENCING", "WRAPPING_UP", "DISCONNECTED",
    ]

    # Agent states (from cti_message.py)
    AGENT_STATES = [
        "LOGGED_OUT", "READY", "NOT_READY", "ON_CALL", "WRAPPING", "BREAK",
    ]

    # Valid call state transitions
    CALL_TRANSITIONS = {
        "IDLE": ["RINGING"],
        "RINGING": ["CONNECTED", "DISCONNECTED"],
        "CONNECTED": ["HELD", "TRANSFERRING", "CONFERENCING", "WRAPPING_UP", "DISCONNECTED"],
        "HELD": ["CONNECTED", "DISCONNECTED"],
        "TRANSFERRING": ["CONNECTED", "DISCONNECTED"],
        "CONFERENCING": ["CONNECTED", "DISCONNECTED"],
        "WRAPPING_UP": ["DISCONNECTED", "IDLE"],
        "DISCONNECTED": ["IDLE"],
    }

    # Valid agent state transitions
    AGENT_TRANSITIONS = {
        "LOGGED_OUT": ["READY", "NOT_READY"],
        "READY": ["ON_CALL", "NOT_READY", "LOGGED_OUT"],
        "NOT_READY": ["READY", "BREAK", "LOGGED_OUT"],
        "ON_CALL": ["WRAPPING", "READY", "NOT_READY", "LOGGED_OUT"],
        "WRAPPING": ["READY", "NOT_READY", "LOGGED_OUT"],
        "BREAK": ["READY", "NOT_READY", "LOGGED_OUT"],
    }

    # Queue and skill groups
    QUEUES = ["sales", "support", "billing", "retention", "collections", "tech"]
    SKILL_GROUPS = ["english", "spanish", "french", "billing", "tech", "sales"]

    # Wrap-up codes
    WRAP_UP_CODES = [
        "RESOLVED", "ESCALATED", "CALLBACK", "TRANSFER", "VOICEMAIL",
        "SALE_COMPLETE", "NO_SALE", "INFO_REQUEST", "COMPLAINT",
        "PAYMENT_RECEIVED", "ACCOUNT_UPDATE", "TECH_ISSUE",
    ]

    # CTI server identifiers
    CTI_SERVERS = [
        "avaya-aes-01.corp.local",
        "cisco-cucm-01.corp.local",
        "genesys-sipserver-01.corp.local",
        "mitel-ips-01.corp.local",
        "avaya-cms-01.corp.local",
    ]

    # Direction types
    DIRECTIONS = ["inbound", "outbound", "internal"]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.agent_ids = [f"agent_{i:04d}" for i in range(1, 201)]
        self.tenant_ids = [f"tenant_{i:03d}" for i in range(1, 21)]
        self.extensions = [f"{random.randint(1000, 9999)}" for _ in range(200)]
        self.seq_counter = random.randint(1, 100000)

    def _next_seq(self) -> int:
        self.seq_counter += 1
        return self.seq_counter

    def _random_phone(self) -> str:
        return f"+1{random.randint(200, 999)}{random.randint(2000000, 9999999)}"

    def _random_call_id(self) -> str:
        return f"call-{uuid.uuid4().hex[:12]}"

    def _random_agent(self) -> str:
        return random.choice(self.agent_ids)

    def _random_queue(self) -> str:
        return random.choice(self.QUEUES)

    def _random_state_transition(self, transitions_map: Dict, current: str) -> str:
        """Get a valid next state from the transitions map."""
        valid = transitions_map.get(current, [])
        return random.choice(valid) if valid else current

    # ------------------------------------------------------------------
    # CSTA XML message generators
    # ------------------------------------------------------------------

    def generate_csta_make_call(self) -> Tuple[bytes, Dict]:
        """Generate a CSTA MakeCall request (XML)."""
        call_id = self._random_call_id()
        agent = self._random_agent()
        calling_device = random.choice(self.extensions)
        called_number = self._random_phone()
        seq = self._next_seq()

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<MakeCall xmlns="http://www.ecma-international.org/standards/ecma-269">\r\n'
            f'  <callingDevice>{calling_device}</callingDevice>\r\n'
            f'  <calledDirectoryNumber>{called_number}</calledDirectoryNumber>\r\n'
            f'  <userData>\r\n'
            f'    <entry key="callId">{call_id}</entry>\r\n'
            f'    <entry key="agentId">{agent}</entry>\r\n'
            f'    <entry key="queue">{self._random_queue()}</entry>\r\n'
            f'    <entry key="direction">outbound</entry>\r\n'
            f'  </userData>\r\n'
            f'  <correlatorData>{seq}</correlatorData>\r\n'
            '</MakeCall>\r\n'
        )

        message = xml.encode("utf-8")
        meta = {
            "cti_variant": "CSTA",
            "csta_operation": "MakeCall",
            "call_id": call_id,
            "agent_id": agent,
            "calling_device": calling_device,
            "called_number": called_number,
            "direction": "outbound",
            "call_state_before": "IDLE",
            "call_state_after": "RINGING",
            "sequence": seq,
        }
        fields = self._build_xml_fields(xml)
        return message, self._finalize_metadata("csta_make_call", message, meta, fields)

    def generate_csta_answer_call(self) -> Tuple[bytes, Dict]:
        """Generate a CSTA AnswerCall request (XML)."""
        call_id = self._random_call_id()
        agent = self._random_agent()
        device = random.choice(self.extensions)
        seq = self._next_seq()

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<AnswerCall xmlns="http://www.ecma-international.org/standards/ecma-269">\r\n'
            f'  <callToBeAnswered>\r\n'
            f'    <callID>{call_id}</callID>\r\n'
            f'    <deviceID>{device}</deviceID>\r\n'
            f'  </callToBeAnswered>\r\n'
            f'  <userData>\r\n'
            f'    <entry key="agentId">{agent}</entry>\r\n'
            f'  </userData>\r\n'
            f'  <correlatorData>{seq}</correlatorData>\r\n'
            '</AnswerCall>\r\n'
        )

        message = xml.encode("utf-8")
        meta = {
            "cti_variant": "CSTA",
            "csta_operation": "AnswerCall",
            "call_id": call_id,
            "agent_id": agent,
            "device_id": device,
            "call_state_before": "RINGING",
            "call_state_after": "CONNECTED",
            "agent_state_before": "READY",
            "agent_state_after": "ON_CALL",
            "sequence": seq,
        }
        fields = self._build_xml_fields(xml)
        return message, self._finalize_metadata("csta_answer_call", message, meta, fields)

    def generate_csta_clear_call(self) -> Tuple[bytes, Dict]:
        """Generate a CSTA ClearCall request (XML)."""
        call_id = self._random_call_id()
        agent = self._random_agent()
        device = random.choice(self.extensions)
        seq = self._next_seq()

        reasons = ["normalClearing", "userBusy", "callRejected", "noAnswer"]
        reason = random.choice(reasons)

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<ClearCall xmlns="http://www.ecma-international.org/standards/ecma-269">\r\n'
            f'  <callToBeClearedConnection>\r\n'
            f'    <callID>{call_id}</callID>\r\n'
            f'    <deviceID>{device}</deviceID>\r\n'
            f'  </callToBeClearedConnection>\r\n'
            f'  <reason>{reason}</reason>\r\n'
            f'  <correlatorData>{seq}</correlatorData>\r\n'
            '</ClearCall>\r\n'
        )

        message = xml.encode("utf-8")
        meta = {
            "cti_variant": "CSTA",
            "csta_operation": "ClearCall",
            "call_id": call_id,
            "agent_id": agent,
            "device_id": device,
            "clear_reason": reason,
            "call_state_before": "CONNECTED",
            "call_state_after": "DISCONNECTED",
            "sequence": seq,
        }
        fields = self._build_xml_fields(xml)
        return message, self._finalize_metadata("csta_clear_call", message, meta, fields)

    def generate_csta_hold_call(self) -> Tuple[bytes, Dict]:
        """Generate a CSTA HoldCall request (XML)."""
        call_id = self._random_call_id()
        agent = self._random_agent()
        device = random.choice(self.extensions)
        seq = self._next_seq()

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<HoldCall xmlns="http://www.ecma-international.org/standards/ecma-269">\r\n'
            f'  <callToBeHeld>\r\n'
            f'    <callID>{call_id}</callID>\r\n'
            f'    <deviceID>{device}</deviceID>\r\n'
            f'  </callToBeHeld>\r\n'
            f'  <userData>\r\n'
            f'    <entry key="agentId">{agent}</entry>\r\n'
            f'  </userData>\r\n'
            f'  <correlatorData>{seq}</correlatorData>\r\n'
            '</HoldCall>\r\n'
        )

        message = xml.encode("utf-8")
        meta = {
            "cti_variant": "CSTA",
            "csta_operation": "HoldCall",
            "call_id": call_id,
            "agent_id": agent,
            "device_id": device,
            "call_state_before": "CONNECTED",
            "call_state_after": "HELD",
            "sequence": seq,
        }
        fields = self._build_xml_fields(xml)
        return message, self._finalize_metadata("csta_hold_call", message, meta, fields)

    def generate_csta_transfer_call(self) -> Tuple[bytes, Dict]:
        """Generate a CSTA TransferCall request (XML)."""
        call_id = self._random_call_id()
        agent = self._random_agent()
        device = random.choice(self.extensions)
        transfer_to = random.choice(self.extensions)
        seq = self._next_seq()
        queue = self._random_queue()

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<TransferCall xmlns="http://www.ecma-international.org/standards/ecma-269">\r\n'
            f'  <heldCall>\r\n'
            f'    <callID>{call_id}</callID>\r\n'
            f'    <deviceID>{device}</deviceID>\r\n'
            f'  </heldCall>\r\n'
            f'  <activeCall>\r\n'
            f'    <callID>{self._random_call_id()}</callID>\r\n'
            f'    <deviceID>{transfer_to}</deviceID>\r\n'
            f'  </activeCall>\r\n'
            f'  <userData>\r\n'
            f'    <entry key="agentId">{agent}</entry>\r\n'
            f'    <entry key="targetQueue">{queue}</entry>\r\n'
            f'    <entry key="transferReason">customerRequest</entry>\r\n'
            f'  </userData>\r\n'
            f'  <correlatorData>{seq}</correlatorData>\r\n'
            '</TransferCall>\r\n'
        )

        message = xml.encode("utf-8")
        meta = {
            "cti_variant": "CSTA",
            "csta_operation": "TransferCall",
            "call_id": call_id,
            "agent_id": agent,
            "device_id": device,
            "transfer_to": transfer_to,
            "target_queue": queue,
            "call_state_before": "CONNECTED",
            "call_state_after": "TRANSFERRING",
            "sequence": seq,
        }
        fields = self._build_xml_fields(xml)
        return message, self._finalize_metadata("csta_transfer_call", message, meta, fields)

    def generate_csta_monitor_start(self) -> Tuple[bytes, Dict]:
        """Generate a CSTA MonitorStart request (XML)."""
        agent = self._random_agent()
        device = random.choice(self.extensions)
        monitor_type = random.choice(["call", "device", "callsViaDevice"])
        seq = self._next_seq()
        monitor_cross_ref = f"mon-{random.randint(10000, 99999)}"

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<MonitorStart xmlns="http://www.ecma-international.org/standards/ecma-269">\r\n'
            f'  <monitorObject>\r\n'
            f'    <deviceID>{device}</deviceID>\r\n'
            f'  </monitorObject>\r\n'
            f'  <monitorType>{monitor_type}</monitorType>\r\n'
            f'  <monitorCrossRefID>{monitor_cross_ref}</monitorCrossRefID>\r\n'
            f'  <userData>\r\n'
            f'    <entry key="supervisor">{agent}</entry>\r\n'
            f'  </userData>\r\n'
            f'  <correlatorData>{seq}</correlatorData>\r\n'
            '</MonitorStart>\r\n'
        )

        message = xml.encode("utf-8")
        meta = {
            "cti_variant": "CSTA",
            "csta_operation": "MonitorStart",
            "agent_id": agent,
            "device_id": device,
            "monitor_type": monitor_type,
            "monitor_cross_ref": monitor_cross_ref,
            "sequence": seq,
        }
        fields = self._build_xml_fields(xml)
        return message, self._finalize_metadata("csta_monitor_start", message, meta, fields)

    def generate_csta_event_delivered(self) -> Tuple[bytes, Dict]:
        """Generate a CSTA DeliveredEvent notification (XML)."""
        call_id = self._random_call_id()
        agent = self._random_agent()
        device = random.choice(self.extensions)
        caller_id = self._random_phone()
        queue = self._random_queue()
        skill = random.choice(self.SKILL_GROUPS)
        seq = self._next_seq()

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<DeliveredEvent xmlns="http://www.ecma-international.org/standards/ecma-269">\r\n'
            f'  <monitorCrossRefID>mon-{random.randint(10000, 99999)}</monitorCrossRefID>\r\n'
            f'  <connection>\r\n'
            f'    <callID>{call_id}</callID>\r\n'
            f'    <deviceID>{device}</deviceID>\r\n'
            f'  </connection>\r\n'
            f'  <alertingDevice>{device}</alertingDevice>\r\n'
            f'  <callingDevice>{caller_id}</callingDevice>\r\n'
            f'  <calledDevice>{device}</calledDevice>\r\n'
            f'  <lastRedirectionDevice>{queue}</lastRedirectionDevice>\r\n'
            f'  <localConnectionInfo>alerting</localConnectionInfo>\r\n'
            f'  <cause>newCall</cause>\r\n'
            f'  <userData>\r\n'
            f'    <entry key="agentId">{agent}</entry>\r\n'
            f'    <entry key="queue">{queue}</entry>\r\n'
            f'    <entry key="skillGroup">{skill}</entry>\r\n'
            f'    <entry key="direction">inbound</entry>\r\n'
            f'    <entry key="priority">{random.randint(1, 10)}</entry>\r\n'
            f'  </userData>\r\n'
            '</DeliveredEvent>\r\n'
        )

        message = xml.encode("utf-8")
        meta = {
            "cti_variant": "CSTA",
            "csta_operation": "DeliveredEvent",
            "call_id": call_id,
            "agent_id": agent,
            "device_id": device,
            "caller_id": caller_id,
            "queue": queue,
            "skill_group": skill,
            "direction": "inbound",
            "call_state_before": "IDLE",
            "call_state_after": "RINGING",
            "sequence": seq,
        }
        fields = self._build_xml_fields(xml)
        return message, self._finalize_metadata("csta_event_delivered", message, meta, fields)

    # ------------------------------------------------------------------
    # TSAPI SOAP message generator
    # ------------------------------------------------------------------

    def generate_tsapi_request(self) -> Tuple[bytes, Dict]:
        """Generate a TSAPI SOAP envelope request."""
        call_id = self._random_call_id()
        agent = self._random_agent()
        device = random.choice(self.extensions)
        queue = self._random_queue()
        seq = self._next_seq()

        operations = [
            "cstaMonitorStart", "cstaMonitorStop",
            "cstaMakeCall", "cstaAnswerCall",
            "cstaClearConnection", "cstaHoldCall",
            "cstaRetrieveCall", "cstaTransferCall",
            "cstaQueryDeviceInfo", "cstaSetAgentState",
        ]
        operation = random.choice(operations)
        agent_state = random.choice(self.AGENT_STATES)

        soap = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"\r\n'
            '  xmlns:tsapi="http://avaya.com/tsapi">\r\n'
            '  <soapenv:Header>\r\n'
            f'    <tsapi:sessionId>sess-{random.randint(10000, 99999)}</tsapi:sessionId>\r\n'
            f'    <tsapi:invokeId>{seq}</tsapi:invokeId>\r\n'
            f'    <tsapi:timestamp>{datetime.now().isoformat()}</tsapi:timestamp>\r\n'
            '  </soapenv:Header>\r\n'
            '  <soapenv:Body>\r\n'
            f'    <tsapi:{operation}>\r\n'
            f'      <callId>{call_id}</callId>\r\n'
            f'      <deviceId>{device}</deviceId>\r\n'
            f'      <agentId>{agent}</agentId>\r\n'
            f'      <queue>{queue}</queue>\r\n'
            f'      <agentState>{agent_state}</agentState>\r\n'
            f'    </tsapi:{operation}>\r\n'
            '  </soapenv:Body>\r\n'
            '</soapenv:Envelope>\r\n'
        )

        message = soap.encode("utf-8")
        meta = {
            "cti_variant": "TSAPI",
            "tsapi_operation": operation,
            "call_id": call_id,
            "agent_id": agent,
            "device_id": device,
            "queue": queue,
            "agent_state": agent_state,
            "sequence": seq,
        }
        fields = self._build_xml_fields(soap)
        return message, self._finalize_metadata("tsapi_request", message, meta, fields)

    # ------------------------------------------------------------------
    # Cisco Finesse REST/JSON message generator
    # ------------------------------------------------------------------

    def generate_finesse_dialog(self) -> Tuple[bytes, Dict]:
        """Generate a Cisco Finesse dialog JSON message."""
        call_id = self._random_call_id()
        agent = self._random_agent()
        ext = random.choice(self.extensions)
        caller_id = self._random_phone()
        queue = self._random_queue()
        call_state = random.choice(["ALERTING", "ACTIVE", "HELD", "DROPPED", "WRAPPING_UP"])
        direction = random.choice(self.DIRECTIONS)

        dialog = {
            "Dialog": {
                "id": call_id,
                "fromAddress": caller_id if direction == "inbound" else ext,
                "toAddress": ext if direction == "inbound" else caller_id,
                "state": call_state,
                "mediaType": "Voice",
                "associatedDialogUri": f"/finesse/api/Dialog/{call_id}",
                "mediaProperties": {
                    "callType": direction.upper(),
                    "DNIS": ext,
                    "wrapUpReason": (
                        random.choice(self.WRAP_UP_CODES)
                        if call_state == "WRAPPING_UP" else ""
                    ),
                    "queueName": queue,
                    "queueNumber": str(random.randint(100, 999)),
                    "callVariable1": f"tenant={random.choice(self.tenant_ids)}",
                    "callVariable2": f"skill={random.choice(self.SKILL_GROUPS)}",
                    "callVariable3": f"priority={random.randint(1, 10)}",
                },
                "participants": {
                    "Participant": [
                        {
                            "mediaAddress": caller_id,
                            "state": call_state,
                            "startTime": (
                                datetime.now() - timedelta(seconds=random.randint(0, 600))
                            ).isoformat(),
                        },
                        {
                            "mediaAddress": ext,
                            "mediaAddressType": "AGENT_DEVICE",
                            "state": call_state,
                            "startTime": (
                                datetime.now() - timedelta(seconds=random.randint(0, 300))
                            ).isoformat(),
                            "extension": ext,
                            "agentId": agent,
                        },
                    ]
                },
            }
        }

        json_str = json.dumps(dialog, indent=2, default=str)
        message = json_str.encode("utf-8")

        meta = {
            "cti_variant": "Finesse",
            "finesse_state": call_state,
            "call_id": call_id,
            "agent_id": agent,
            "extension": ext,
            "caller_id": caller_id,
            "queue": queue,
            "direction": direction,
            "wrap_up_reason": dialog["Dialog"]["mediaProperties"]["wrapUpReason"],
        }

        fields = [
            {"name": "json_body", "offset": 0, "length": len(message)},
            {"name": "dialog_id", "offset": json_str.find(call_id), "length": len(call_id)},
        ]

        return message, self._finalize_metadata("finesse_dialog", message, meta, fields)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_xml_fields(self, xml_str: str) -> List[Dict]:
        """Build field metadata from an XML message."""
        xml_bytes = xml_str.encode("utf-8")
        fields = [
            {"name": "xml_message", "offset": 0, "length": len(xml_bytes)},
        ]

        # Find XML declaration
        decl_end = xml_str.find("?>")
        if decl_end > 0:
            fields.append({"name": "xml_declaration", "offset": 0, "length": decl_end + 2})

        # Find root element
        first_tag_start = xml_str.find("<", decl_end + 2 if decl_end > 0 else 0)
        if first_tag_start >= 0:
            first_tag_end = xml_str.find(">", first_tag_start)
            if first_tag_end > 0:
                fields.append({
                    "name": "root_element",
                    "offset": first_tag_start,
                    "length": first_tag_end - first_tag_start + 1,
                })

        return fields

    def _finalize_metadata(
        self, msg_type: str, message: bytes, meta: Dict, fields: list
    ) -> Dict:
        """Create the final metadata dict."""
        meta.update({
            "protocol": "cti",
            "message_type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message),
            "fields": fields,
            "hash": hashlib.sha256(message).hexdigest(),
        })
        return meta

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of CTI messages."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = [
            ("csta_make_call", self.generate_csta_make_call, 0.10),
            ("csta_answer_call", self.generate_csta_answer_call, 0.10),
            ("csta_clear_call", self.generate_csta_clear_call, 0.10),
            ("csta_hold_call", self.generate_csta_hold_call, 0.08),
            ("csta_transfer_call", self.generate_csta_transfer_call, 0.08),
            ("csta_monitor_start", self.generate_csta_monitor_start, 0.07),
            ("csta_event_delivered", self.generate_csta_event_delivered, 0.12),
            ("tsapi_request", self.generate_tsapi_request, 0.15),
            ("finesse_dialog", self.generate_finesse_dialog, 0.20),
        ]

        dataset_metadata = {
            "protocol": "cti",
            "version": "CSTA/TSAPI/Finesse",
            "total_samples": num_samples,
            "samples_by_type": {},
            "generated_at": datetime.now().isoformat(),
        }

        sample_idx = 0
        for msg_type, generator, ratio in generators:
            count = int(num_samples * ratio)
            dataset_metadata["samples_by_type"][msg_type] = count

            for i in range(count):
                message, metadata = generator()
                metadata["sample_index"] = sample_idx
                metadata["message_type"] = msg_type

                bin_path = output_path / f"cti_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = output_path / f"cti_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate CTI dataset."""
    generator = CTIGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "cti"

    print("Generating CTI dataset...")
    metadata = generator.generate_dataset(num_samples=1000, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} samples")
    print(f"Output directory: {output_dir}")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")


if __name__ == "__main__":
    main()
