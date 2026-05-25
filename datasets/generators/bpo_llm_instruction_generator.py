"""
BPO LLM Instruction Pair Generator

Generates instruction-response pairs for LLM fine-tuning on BPO/call center
security domains. Produces a JSON array of instruction pairs.

Categories (100 pairs each, 500 total):
1. security_policy_generation: Generate security policies for BPO environments
2. fraud_pattern_analysis: Analyze toll fraud patterns and CDR anomalies
3. pci_compliance_assessment: Assess PCI-DSS compliance for voice channels
4. call_anomaly_analysis: Analyze call flow anomalies and suspicious patterns
5. protocol_identification: Identify and classify BPO protocol messages

Each pair includes:
- pair_id: Unique identifier
- category: One of 5 categories
- difficulty: basic / intermediate / advanced
- instruction: The prompt/question
- context: Optional context data
- response: Expected model response
"""

import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class BPOLLMInstructionGenerator:
    """Generate LLM fine-tuning instruction pairs for BPO security domains."""

    CATEGORIES = [
        "security_policy_generation",
        "fraud_pattern_analysis",
        "pci_compliance_assessment",
        "call_anomaly_analysis",
        "protocol_identification",
    ]

    DIFFICULTIES = ["basic", "intermediate", "advanced"]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Security Policy Generation
    # ------------------------------------------------------------------

    def _generate_security_policy_pairs(self, count: int) -> List[Dict]:
        """Generate security policy instruction pairs."""
        templates = [
            {
                "difficulty": "basic",
                "instructions": [
                    "Write a password policy for BPO agent desktop systems.",
                    "Create a clean desk policy for contact center agents handling sensitive data.",
                    "Draft an acceptable use policy for BPO agent workstations.",
                    "Write a policy for handling customer personally identifiable information (PII) in a call center.",
                    "Create a visitor access policy for a contact center floor.",
                ],
                "contexts": [
                    "Environment: 500-seat contact center with 24/7 operations. Agents handle financial services accounts.",
                    "Environment: Healthcare BPO with HIPAA and PCI-DSS requirements. Agents access patient and payment data.",
                    "Environment: Multi-tenant BPO facility serving retail and banking clients. Shared infrastructure.",
                ],
                "responses": [
                    "## BPO Agent Desktop Password Policy\n\n### Purpose\nEstablish password requirements for all agent desktop systems to prevent unauthorized access.\n\n### Requirements\n1. **Minimum Length**: 12 characters\n2. **Complexity**: Must include uppercase, lowercase, number, and special character\n3. **Rotation**: Every 90 days, no reuse of last 12 passwords\n4. **Lockout**: Account locks after 5 failed attempts for 30 minutes\n5. **Session Timeout**: Auto-lock after 5 minutes of inactivity\n6. **MFA**: Required for supervisor and admin accounts\n7. **Shared Accounts**: Prohibited - each agent must have individual credentials\n\n### Enforcement\n- Active Directory Group Policy enforces all requirements\n- Compliance monitored via SIEM alerts\n- Violations result in immediate account suspension",
                    "## Clean Desk Policy for Contact Center Operations\n\n### Purpose\nPrevent unauthorized access to sensitive customer data through physical security controls.\n\n### Requirements\n1. **No Written Notes**: Agents must not write down customer PII, card numbers, or account details\n2. **Screen Lock**: Agents must lock screens (Win+L) when leaving desks, even briefly\n3. **Printed Materials**: All printouts containing customer data must be shredded immediately after use\n4. **Personal Devices**: No personal phones, cameras, or USB drives on the operations floor\n5. **End of Shift**: All work materials must be secured; nothing left on desks\n6. **Visitor Escort**: All visitors must be escorted and cannot access agent workstations\n\n### Monitoring\n- Random floor audits by supervisors (weekly)\n- CCTV monitoring of operations floor\n- Violations tracked and reported to compliance team",
                ],
            },
            {
                "difficulty": "intermediate",
                "instructions": [
                    "Design a data loss prevention (DLP) policy for a BPO handling credit card transactions over voice channels.",
                    "Create an incident response procedure for a suspected toll fraud attack on the PBX.",
                    "Write a security policy for remote/work-from-home BPO agents.",
                    "Draft a network segmentation policy for a multi-tenant BPO platform.",
                    "Create a call recording retention and encryption policy.",
                ],
                "contexts": [
                    "The BPO processes 50,000 calls/day across 3 shifts. PCI-DSS Level 1 certified. Uses Avaya Aura CM and Genesys Cloud.",
                    "BPO experienced a PBX hack last quarter resulting in $45,000 in fraudulent international calls to premium rate numbers.",
                    "30% of agents transitioned to WFH during COVID. Using VPN + Citrix VDI. Handling financial services and healthcare.",
                ],
                "responses": [
                    "## DLP Policy for Voice Channel Card Transactions\n\n### Scope\nAll voice channels where payment card data is captured via DTMF or spoken digits.\n\n### Controls\n1. **DTMF Masking**: All DTMF tones carrying card data must be masked using flat-tone replacement before reaching agent headsets or recording systems\n2. **Recording Pause**: Call recording automatically pauses during PCI scope (payment capture) and resumes after\n3. **Agent Screen Masking**: CRM displays only first 6 / last 4 digits of card numbers\n4. **Network Monitoring**: Real-time DLP rules scan for 13-19 digit sequences matching Luhn validation\n5. **Chat/Email Filtering**: Card number patterns blocked in agent chat tools\n6. **Log Scrubbing**: All application logs sanitized to remove potential PAN data\n\n### Monitoring\n- PCI scope entry/exit logged with timestamps\n- DTMF masking effectiveness validated via test calls (monthly)\n- DLP rule triggers reviewed daily by security team\n- Quarterly penetration testing of voice and data channels",
                    "## Incident Response: Toll Fraud Attack on PBX\n\n### Detection Phase (0-15 min)\n1. SIEM alert triggers on high-volume international calls to premium prefixes\n2. Verify alert: Check CDR spike for IRSF patterns (+252, +232, +870 destinations)\n3. Confirm: Compare against normal baseline traffic\n\n### Containment (15-30 min)\n1. Block outbound international calling immediately via trunk group restrictions\n2. Disable compromised voicemail ports and DISA access\n3. Enable enhanced logging on all PBX interfaces\n4. Notify carrier to block premium rate destinations\n\n### Eradication (30 min - 4 hours)\n1. Reset all PBX admin credentials\n2. Audit all voicemail boxes for weak PINs\n3. Review DISA configuration and disable if not required\n4. Update trunk group ACLs\n5. Patch PBX software to latest version\n\n### Recovery (4-24 hours)\n1. Re-enable international calling with new restrictions\n2. Implement real-time CDR monitoring with $100/hour threshold alerts\n3. Deploy toll fraud detection ML model\n\n### Post-Incident\n- Document total fraud charges and file carrier dispute\n- Update security policies\n- Conduct agent training on social engineering awareness",
                ],
            },
            {
                "difficulty": "advanced",
                "instructions": [
                    "Design a zero-trust security architecture for a cloud-based multi-tenant BPO platform.",
                    "Create a comprehensive security operations center (SOC) runbook for BPO-specific threats.",
                    "Write an AI/ML governance policy for automated security decision-making in contact centers.",
                    "Design a post-quantum cryptography migration plan for SIP/SRTP communications in a large BPO.",
                    "Create a supply chain security policy for third-party integrations in a BPO tech stack.",
                ],
                "contexts": [
                    "Enterprise BPO: 10,000 agents, 5 countries, $2B annual revenue. Moving from on-premise to cloud. Handles healthcare, financial services, and government contracts.",
                    "SOC team of 12 analysts monitoring 50,000 endpoints and 100,000 daily calls. Current MTTD is 4.5 hours; target is 15 minutes.",
                ],
                "responses": [
                    "## Zero-Trust Architecture for Cloud BPO Platform\n\n### Core Principles\n1. **Never Trust, Always Verify**: Every request authenticated regardless of network location\n2. **Least Privilege**: Agents access only their tenant's data and required tools\n3. **Assume Breach**: Design for detection and containment, not just prevention\n\n### Identity Layer\n- SAML 2.0/OIDC SSO for all agent and supervisor access\n- MFA mandatory (hardware tokens for admins, app-based for agents)\n- Continuous authentication via behavioral biometrics (typing patterns, mouse movement)\n- Session tokens with 15-minute expiry, automatic re-authentication\n\n### Network Layer\n- Micro-segmentation: Each tenant in isolated VLAN/VPC\n- SIP/RTP traffic encrypted (SRTP with AES-256-GCM)\n- No direct internet access from agent workstations\n- SDP (Software Defined Perimeter) for all management access\n\n### Data Layer\n- Encryption at rest (AES-256) and in transit (TLS 1.3)\n- PCI scope: DTMF masking + recording pause + screen masking\n- DLP policies on all egress points\n- Real-time PAN detection with automatic redaction\n\n### Monitoring\n- SIEM with BPO-specific detection rules\n- UEBA for agent behavioral anomalies\n- ML-based toll fraud detection on CDR streams\n- Automated incident response playbooks\n\n### Compliance\n- Continuous PCI-DSS, HIPAA, SOC 2 compliance monitoring\n- Automated evidence collection for audits\n- Quarterly penetration testing with BPO-specific scenarios",
                ],
            },
        ]

        pairs = []
        for _ in range(count):
            template = random.choice(templates)
            difficulty = template["difficulty"]
            instruction = random.choice(template["instructions"])
            context = random.choice(template["contexts"])
            response = random.choice(template["responses"])

            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "security_policy_generation",
                "difficulty": difficulty,
                "instruction": instruction,
                "context": context,
                "response": response,
            })
        return pairs

    # ------------------------------------------------------------------
    # Fraud Pattern Analysis
    # ------------------------------------------------------------------

    def _generate_fraud_analysis_pairs(self, count: int) -> List[Dict]:
        """Generate fraud pattern analysis instruction pairs."""
        fraud_types = [
            "IRSF", "PBX_HACK", "CALL_TRANSFER", "WANGIRI", "SUBSCRIPTION",
            "CALL_PUMPING", "ARBITRAGE", "BYPASS", "CLIP_MANIPULATION",
            "TOLL_FREE_ABUSE",
        ]

        instructions = [
            f"Analyze this CDR pattern and identify if it indicates {ft} fraud. Explain the indicators."
            for ft in fraud_types
        ] + [
            "Given this set of CDRs, calculate the fraud risk score and explain your reasoning.",
            "Identify the fraud pattern type from the following call detail records.",
            "Compare these CDRs against known IRSF patterns and flag suspicious records.",
            "What premium rate prefixes are being targeted in this CDR data?",
            "Explain the difference between IRSF and Call Pumping fraud patterns.",
            "How would you detect Wangiri callback fraud from CDR data alone?",
            "Design a real-time fraud detection rule for PBX hack attempts.",
            "What CDR features are most predictive of toll fraud?",
            "Analyze the temporal pattern of these calls to detect automated dialing.",
            "Identify the geographic distribution anomaly in this CDR dataset.",
        ]

        contexts = [
            (
                "CDR Sample: 15 calls to +252 (Somalia) in 10 minutes from trunk T001, "
                "average duration 45 min, total cost $2,340, all from extension 4501 after hours."
            ),
            (
                "CDR Sample: 200 calls to +1-900 numbers in 1 hour, duration 1-3 seconds each, "
                "originating from 5 different voicemail boxes, no agent logged in."
            ),
            (
                "CDR Sample: Agent transfers 8 calls to +870 (satellite) numbers in a shift, "
                "each lasting 30+ minutes. Normal transfer rate is 2/day to domestic numbers."
            ),
            (
                "CDR Sample: 500 missed calls from +232 (Sierra Leone) in 30 minutes to "
                "toll-free numbers, followed by 50 callbacks to the same prefix."
            ),
            (
                "CDR Sample: 100 calls/hour from SIM gateway, call duration exactly 60 seconds, "
                "all to same +388 (IPRN) prefix, A-number rotates every 10 calls."
            ),
        ]

        responses = [
            (
                "## Fraud Analysis: IRSF (International Revenue Share Fraud)\n\n"
                "### Indicators Detected\n"
                "1. **Destination**: +252 (Somalia) - known IRSF premium rate destination\n"
                "2. **Volume**: 15 calls in 10 minutes - abnormal burst pattern\n"
                "3. **Duration**: Average 45 minutes - artificially extended to maximize revenue share\n"
                "4. **Timing**: After-hours activity suggests unauthorized access\n"
                "5. **Source**: Single extension (4501) - likely compromised credentials\n\n"
                "### Risk Score: 95/100 (Critical)\n\n"
                "### Recommended Actions\n"
                "- Immediately block outbound to +252 prefix\n"
                "- Disable extension 4501 pending investigation\n"
                "- Check voicemail configuration for unauthorized forwarding\n"
                "- File carrier dispute for $2,340 in charges\n"
                "- Review access logs for extension 4501"
            ),
            (
                "## Fraud Analysis: PBX Hack via Voicemail\n\n"
                "### Indicators Detected\n"
                "1. **Pattern**: Short-duration calls (1-3 sec) to +1-900 premium numbers\n"
                "2. **Volume**: 200 calls/hour - automated dialer pattern\n"
                "3. **Source**: 5 voicemail boxes - suggests brute-force compromise\n"
                "4. **Agent Activity**: No agents logged in - unauthorized access\n\n"
                "### Risk Score: 98/100 (Critical)\n\n"
                "### Pattern Classification: PBX_HACK\n"
                "This matches the classic voicemail-to-PBX hack pattern where attackers "
                "compromise voicemail boxes with weak PINs, then use DISA/auto-attendant "
                "to make outbound calls to premium rate numbers."
            ),
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "fraud_pattern_analysis",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    # ------------------------------------------------------------------
    # PCI Compliance Assessment
    # ------------------------------------------------------------------

    def _generate_pci_compliance_pairs(self, count: int) -> List[Dict]:
        """Generate PCI compliance assessment instruction pairs."""
        instructions = [
            "Assess the PCI-DSS compliance of this DTMF payment capture flow.",
            "Review this call recording configuration for PCI-DSS 4.0 compliance.",
            "Identify PCI compliance gaps in this contact center architecture.",
            "Evaluate the agent screen masking implementation for PCI compliance.",
            "Is this DTMF masking configuration sufficient for PCI-DSS Requirement 3.3?",
            "Assess whether this recording pause/resume implementation meets PCI-DSS 3.4.",
            "Review this PCI scope management flow and identify risks.",
            "What PCI-DSS requirements apply to DTMF relay via SIP INFO?",
            "Evaluate this IVR payment flow against PCI-DSS SAQ requirements.",
            "How should call recordings be encrypted to meet PCI-DSS Requirement 3.5?",
            "Assess the risk of PAN exposure in this agent desktop configuration.",
            "Review this compliance report and suggest improvements.",
        ]

        contexts = [
            (
                "Configuration: DTMF masking mode=clamp, recording_pause=enabled, "
                "screen_masking=first6_last4, SRTP=enabled, agent_can_hear_tones=false, "
                "card_data_stored=tokenized"
            ),
            (
                "Configuration: DTMF masking mode=disabled, recording_pause=manual, "
                "screen_masking=disabled, SRTP=disabled, agent_can_hear_tones=true, "
                "card_data_stored=encrypted_pan"
            ),
            (
                "IVR Flow: Customer enters card number via DTMF -> system validates Luhn -> "
                "amount confirmed -> payment processed. Recording continues throughout. "
                "DTMF tones passed to agent headset."
            ),
            (
                "Architecture: SIP trunk -> SBC -> CUCM -> Agent Phone. Recording at SBC "
                "level. No DTMF interception point. Card numbers spoken by customer."
            ),
            (
                "Compliance Report: Overall score 72/100. DTMF masking: 95, Recording pause: 60, "
                "Screen masking: 85, Data protection: 48. Violations: pan_in_logs (15 events), "
                "recording_not_paused (8 events)."
            ),
        ]

        responses = [
            (
                "## PCI-DSS Compliance Assessment\n\n"
                "### Findings\n"
                "**COMPLIANT** - This configuration meets PCI-DSS 4.0 requirements:\n"
                "- Req 3.3: DTMF clamping prevents card data capture in audio ✅\n"
                "- Req 3.4: Recording pause prevents PAN in recordings ✅\n"
                "- Req 3.5: SRTP encryption protects data in transit ✅\n"
                "- Req 8.3: Agent cannot hear/see full card numbers ✅\n"
                "- Req 10.2: Tokenization reduces PCI scope ✅\n\n"
                "### Score: 95/100\n"
                "### Recommendation: Continue monitoring. Consider adding ML-based PAN "
                "detection as an additional layer."
            ),
            (
                "## PCI-DSS Compliance Assessment\n\n"
                "### CRITICAL FINDINGS - NON-COMPLIANT\n\n"
                "**Violations:**\n"
                "1. **Req 3.3 FAIL**: DTMF masking disabled - card digits captured in audio stream\n"
                "2. **Req 3.4 FAIL**: Recording pause is manual - agents forget to pause 30% of the time\n"
                "3. **Req 3.5 FAIL**: SRTP disabled - card data transmitted unencrypted\n"
                "4. **Req 8.3 FAIL**: Agent can hear DTMF tones - full card number audible\n"
                "5. **Req 10.2 WARNING**: Storing encrypted PAN instead of tokens increases scope\n\n"
                "### Score: 15/100\n"
                "### Required Actions:\n"
                "- Immediately enable DTMF masking (flat_tone or clamp mode)\n"
                "- Automate recording pause on PCI scope entry\n"
                "- Enable SRTP on all voice channels\n"
                "- Implement agent screen masking\n"
                "- Migrate from PAN storage to tokenization"
            ),
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "pci_compliance_assessment",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    # ------------------------------------------------------------------
    # Call Anomaly Analysis
    # ------------------------------------------------------------------

    def _generate_call_anomaly_pairs(self, count: int) -> List[Dict]:
        """Generate call anomaly analysis instruction pairs."""
        instructions = [
            "Analyze this call flow for anomalies and potential security issues.",
            "This agent's call pattern deviates from baseline. What could this indicate?",
            "Identify the anomaly type from these call metrics.",
            "Is this hold time pattern suspicious? Explain your analysis.",
            "Analyze the transfer pattern and determine if it indicates social engineering.",
            "What security alerts should be generated from this call behavior?",
            "Compare this agent's metrics against the team baseline and flag anomalies.",
            "Detect potential account takeover indicators in this call sequence.",
            "Analyze this after-hours call activity for insider threat indicators.",
            "Evaluate this call routing pattern for potential toll fraud relay.",
        ]

        contexts = [
            (
                "Agent AGT1234: 45 calls today (baseline: 28), avg hold time 8.5 min "
                "(baseline: 1.2 min), 12 transfers to external numbers (baseline: 0.5), "
                "3 calls to +252 prefix. Shift: night."
            ),
            (
                "Call flow: Customer verified identity -> agent placed on hold 6 times "
                "(total 22 min) -> 3 supervisor escalations -> call transferred to external "
                "number +1-876-xxx-xxxx -> call lasted 47 minutes."
            ),
            (
                "Agent metrics deviation: Wrap-up time 45 sec (baseline 120 sec), CSAT 2.1 "
                "(baseline 4.2), call recordings show agent rushing through verification. "
                "4 customer complaints about identity verification being skipped."
            ),
            (
                "After-hours activity: Extension 3502 made 25 outbound calls between 2-4 AM. "
                "No agent logged in to the CTI system. Calls to 8 different countries. "
                "PBX admin account accessed from VPN at 1:45 AM."
            ),
        ]

        responses = [
            (
                "## Call Anomaly Analysis\n\n"
                "### Anomaly Type: Potential Insider Toll Fraud\n\n"
                "### Indicators\n"
                "1. **Volume Anomaly**: 45 calls (60% above 28-call baseline) - HIGH\n"
                "2. **Hold Pattern**: 8.5 min avg hold (7x baseline) - CRITICAL\n"
                "   - Extended holds often used to mask unauthorized calls\n"
                "3. **Transfer Pattern**: 12 external transfers (24x baseline) - CRITICAL\n"
                "   - Transfers to international numbers are a key IRSF indicator\n"
                "4. **Destination**: 3 calls to +252 (Somalia premium rate) - CRITICAL\n"
                "5. **Timing**: Night shift - lower supervision\n\n"
                "### Risk Score: 92/100\n"
                "### Recommended Actions\n"
                "- Immediately block agent's external transfer capability\n"
                "- Review all 12 external transfers for destination analysis\n"
                "- Pull call recordings for the +252 calls\n"
                "- Initiate HR investigation\n"
                "- Block +252 prefix on all trunks"
            ),
            (
                "## Call Anomaly Analysis\n\n"
                "### Anomaly Type: Social Engineering / Account Takeover Attempt\n\n"
                "### Indicators\n"
                "1. **Hold Pattern**: 6 holds totaling 22 minutes - caller stalling\n"
                "2. **Escalation**: 3 supervisor escalations - attempting authority bypass\n"
                "3. **Transfer**: External transfer to +1-876 (Jamaica - known IRSF prefix)\n"
                "4. **Duration**: 47-minute call - unusually long for standard transaction\n\n"
                "### Risk Score: 85/100\n"
                "### Pattern: Likely social engineering where caller manipulated agent into "
                "transferring to a premium rate number disguised as a 'callback number'.\n\n"
                "### Recommended Actions\n"
                "- Review call recording for social engineering tactics\n"
                "- Flag the destination number for fraud analysis\n"
                "- Retrain agent on transfer verification procedures\n"
                "- Add +1-876 to premium rate watchlist"
            ),
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "call_anomaly_analysis",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    # ------------------------------------------------------------------
    # Protocol Identification
    # ------------------------------------------------------------------

    def _generate_protocol_identification_pairs(self, count: int) -> List[Dict]:
        """Generate protocol identification instruction pairs."""
        instructions = [
            "Identify the protocol and message type from this binary/text payload.",
            "Classify this message as SIP, RTP, TN3270e, CTI, or IVR and explain why.",
            "What protocol fields can you identify in this message?",
            "Parse the header structure and identify the protocol version.",
            "Is this a request or response message? What protocol does it belong to?",
            "Identify the BPO-specific extensions in this protocol message.",
            "What security-relevant fields exist in this protocol message?",
            "Classify this traffic pattern as voice signaling, media, or control.",
            "Identify the vendor from the protocol message characteristics.",
            "What call state transition does this CTI message represent?",
        ]

        contexts = [
            "Payload (hex): 80 60 00 2A 00 00 0A 00 12 34 56 78 [160 bytes audio]",
            "INVITE sip:4001@pbx.example.com SIP/2.0\\r\\nVia: SIP/2.0/UDP 10.0.1.5:5060\\r\\nX-CC-Tenant: tenant_001\\r\\nX-CC-Queue: sales",
            "Payload (hex): 00 00 00 00 15 F5 C3 [EBCDIC screen data with SBA orders]",
            '<?xml version="1.0"?><DeliveredEvent xmlns="http://www.ecma-international.org/standards/ecma-269"><callID>call-abc123</callID>',
            '<vxml version="2.1"><form id="payment"><field name="card_number" type="digits?length=16">',
            "MRCP/2.0 450 RECOGNIZE 12345\\r\\nChannel-Identifier: ch-001@speechrecog\\r\\nContent-Type: application/srgs+xml",
            '{"Dialog":{"id":"call-xyz","state":"ACTIVE","mediaType":"Voice","mediaProperties":{"callType":"INBOUND"}}}',
        ]

        responses = [
            (
                "## Protocol Identification: RTP (Real-time Transport Protocol)\n\n"
                "### Evidence\n"
                "- First byte 0x80: Version=2, Padding=0, Extension=0, CC=0\n"
                "- Second byte 0x60: Marker=0, Payload Type=96 (dynamic video)\n"
                "- Bytes 2-3: Sequence number 0x002A = 42\n"
                "- Bytes 4-7: Timestamp 0x00000A00\n"
                "- Bytes 8-11: SSRC 0x12345678\n"
                "- 160 bytes audio payload follows 12-byte header\n\n"
                "### Classification: rtp_audio (standard RTP audio packet)\n"
                "### Protocol Version: RFC 3550 v2"
            ),
            (
                "## Protocol Identification: SIP (Session Initiation Protocol)\n\n"
                "### Evidence\n"
                "- Request line: INVITE method with SIP/2.0 version\n"
                "- Standard SIP headers: Via, From, To, Call-ID\n"
                "- BPO Extensions: X-CC-Tenant (tenant_001), X-CC-Queue (sales)\n"
                "- Transport: UDP port 5060\n\n"
                "### Classification: sip_invite (session initiation request)\n"
                "### BPO Context: Call center tenant_001, sales queue\n"
                "### Vendor Detection: Cannot determine from this snippet alone"
            ),
            (
                "## Protocol Identification: TN3270e (Terminal Network Extended)\n\n"
                "### Evidence\n"
                "- 5-byte TN3270e header: data_type=0x00 (DATA_3270)\n"
                "- SBA (Set Buffer Address) orders present in payload\n"
                "- EBCDIC-encoded screen content (not ASCII)\n"
                "- Characteristic 3270 write command structure\n\n"
                "### Classification: tn3270e_data_3270 (screen write)\n"
                "### Protocol Version: RFC 2355"
            ),
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "protocol_identification",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of LLM instruction pairs."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pairs_per_category = num_samples // len(self.CATEGORIES)
        remainder = num_samples % len(self.CATEGORIES)

        all_pairs = []
        samples_by_category = {}

        category_generators = {
            "security_policy_generation": self._generate_security_policy_pairs,
            "fraud_pattern_analysis": self._generate_fraud_analysis_pairs,
            "pci_compliance_assessment": self._generate_pci_compliance_pairs,
            "call_anomaly_analysis": self._generate_call_anomaly_pairs,
            "protocol_identification": self._generate_protocol_identification_pairs,
        }

        for idx, category in enumerate(self.CATEGORIES):
            count = pairs_per_category + (1 if idx < remainder else 0)
            generator = category_generators[category]
            pairs = generator(count)
            all_pairs.extend(pairs)
            samples_by_category[category] = count

        # Shuffle all pairs
        random.shuffle(all_pairs)

        # Write to JSON file
        pairs_file = output_path / "instruction_pairs.json"
        with open(pairs_file, "w") as f:
            json.dump(all_pairs, f, indent=2, default=str)

        dataset_metadata = {
            "protocol": "bpo_llm_pairs",
            "version": "1.0",
            "total_samples": len(all_pairs),
            "samples_by_type": samples_by_category,
            "difficulties": {
                d: sum(1 for p in all_pairs if p["difficulty"] == d)
                for d in self.DIFFICULTIES
            },
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate BPO LLM instruction pairs dataset."""
    generator = BPOLLMInstructionGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "security_events" / "bpo_llm_pairs"

    print("Generating BPO LLM instruction pairs...")
    metadata = generator.generate_dataset(num_samples=500, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} instruction pairs")
    print(f"Output directory: {output_dir}")
    for category, count in metadata["samples_by_type"].items():
        print(f"  - {category}: {count} pairs")
    print(f"\nDifficulty distribution:")
    for difficulty, count in metadata["difficulties"].items():
        print(f"  - {difficulty}: {count} pairs")


if __name__ == "__main__":
    main()
