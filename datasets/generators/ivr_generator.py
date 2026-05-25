"""
IVR (Interactive Voice Response) Protocol Document Generator

Generates realistic IVR documents for training the protocol discovery
and field detection models.

Document types generated:
1. vxml_form: VoiceXML form dialogs with input fields
2. vxml_menu: VoiceXML menu navigations with DTMF choices
3. mrcp_speak: MRCP SPEAK requests (TTS)
4. mrcp_recognize: MRCP RECOGNIZE requests (ASR/DTMF)
5. ccxml_event: CCXML event notifications
6. payment_capture_flow: PCI-compliant VoiceXML payment capture forms

Includes:
- SRGS grammars for speech/DTMF recognition
- TTS prompts with SSML markup
- PCI payment capture with DTMF masking annotations
- Realistic BPO IVR flow structures
"""

import json
import random
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class IVRGenerator:
    """Generate realistic IVR protocol documents for ML training."""

    # MRCP methods
    MRCP_SPEAK_METHODS = ["SPEAK", "STOP", "PAUSE", "RESUME", "BARGE-IN-OCCURRED"]
    MRCP_RECOGNIZE_METHODS = [
        "RECOGNIZE", "START-OF-INPUT", "RECOGNITION-COMPLETE",
        "DEFINE-GRAMMAR", "INTERPRET",
    ]

    # MRCP status codes
    MRCP_STATUS_CODES = {
        200: "Success",
        401: "Method Not Allowed",
        402: "Method Not Valid In This State",
        403: "Unsupported Parameter",
        404: "Illegal Value For Parameter",
        405: "Not Found",
    }

    # TTS voices
    TTS_VOICES = [
        "allison", "michael", "lisa", "james", "susan",
        "en-US-Standard-A", "en-US-Standard-B", "en-US-Standard-C",
        "en-US-Wavenet-A", "en-US-Wavenet-D",
    ]

    # IVR menu options
    MENU_OPTIONS = {
        "main_menu": [
            ("1", "For billing and payments"),
            ("2", "For technical support"),
            ("3", "For sales and new services"),
            ("4", "For account management"),
            ("5", "To speak with an agent"),
            ("0", "To hear these options again"),
        ],
        "billing_menu": [
            ("1", "To make a payment"),
            ("2", "To check your balance"),
            ("3", "For payment history"),
            ("4", "To set up autopay"),
            ("9", "To return to the main menu"),
        ],
        "tech_support_menu": [
            ("1", "For internet issues"),
            ("2", "For phone service"),
            ("3", "For cable or TV"),
            ("4", "For equipment problems"),
            ("9", "To return to the main menu"),
        ],
        "language_menu": [
            ("1", "For English"),
            ("2", "Para espanol"),
            ("3", "Pour francais"),
        ],
    }

    # CCXML event types
    CCXML_EVENTS = [
        "connection.alerting", "connection.connected",
        "connection.disconnected", "connection.failed",
        "dialog.started", "dialog.exit",
        "conference.joined", "conference.exited",
        "error.connection", "error.dialog",
    ]

    # Card brands for payment capture
    CARD_BRANDS = {
        "visa": {"prefix": "4", "length": 16},
        "mastercard": {"prefix": "5", "length": 16},
        "amex": {"prefix": "3", "length": 15},
        "discover": {"prefix": "6", "length": 16},
    }

    # Queue names
    QUEUES = ["sales", "support", "billing", "retention", "collections"]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.request_id_counter = random.randint(1, 100000)
        self.channel_ids = [
            f"ch-{random.randint(10000, 99999)}" for _ in range(50)
        ]

    def _next_request_id(self) -> int:
        self.request_id_counter += 1
        return self.request_id_counter

    def _random_phone(self) -> str:
        return f"+1{random.randint(200, 999)}{random.randint(2000000, 9999999)}"

    def _random_session_id(self) -> str:
        return f"sess-{uuid.uuid4().hex[:12]}"

    def _luhn_checksum(self, partial: str) -> str:
        """Generate a Luhn-valid check digit for a partial card number."""
        digits = [int(d) for d in partial]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        total = sum(odd_digits)
        for d in even_digits:
            total += sum(divmod(d * 2, 10))
        check = (10 - (total % 10)) % 10
        return str(check)

    def _generate_card_number(self, brand: str) -> str:
        """Generate a Luhn-valid card number for a given brand."""
        info = self.CARD_BRANDS[brand]
        prefix = info["prefix"]
        length = info["length"]
        partial = prefix + "".join(
            str(random.randint(0, 9)) for _ in range(length - len(prefix) - 1)
        )
        return partial + self._luhn_checksum(partial)

    # ------------------------------------------------------------------
    # VoiceXML generators
    # ------------------------------------------------------------------

    def generate_vxml_form(self) -> Tuple[bytes, Dict]:
        """Generate a VoiceXML form dialog."""
        session_id = self._random_session_id()
        caller = self._random_phone()

        form_types = ["account_lookup", "callback_request", "survey", "balance_inquiry"]
        form_type = random.choice(form_types)

        if form_type == "account_lookup":
            vxml = self._build_vxml_account_lookup(session_id, caller)
        elif form_type == "callback_request":
            vxml = self._build_vxml_callback_request(session_id, caller)
        elif form_type == "survey":
            vxml = self._build_vxml_survey(session_id, caller)
        else:
            vxml = self._build_vxml_balance_inquiry(session_id, caller)

        message = vxml.encode("utf-8")
        meta = {
            "ivr_document_type": "VoiceXML",
            "form_type": form_type,
            "session_id": session_id,
            "caller": caller,
            "has_grammar": True,
            "has_tts_prompt": True,
        }
        fields = self._build_xml_fields(vxml)
        return message, self._finalize_metadata("vxml_form", message, meta, fields)

    def generate_vxml_menu(self) -> Tuple[bytes, Dict]:
        """Generate a VoiceXML menu navigation."""
        session_id = self._random_session_id()
        menu_name = random.choice(list(self.MENU_OPTIONS.keys()))
        options = self.MENU_OPTIONS[menu_name]

        choices_xml = ""
        for digit, desc in options:
            choices_xml += (
                f'    <choice dtmf="{digit}" next="#action_{digit}">\r\n'
                f'      {desc}\r\n'
                f'    </choice>\r\n'
            )

        prompt_text = "Please select from the following options. "
        prompt_text += " ".join(f"Press {d} {desc}." for d, desc in options)

        vxml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<vxml version="2.1" xmlns="http://www.w3.org/2001/vxml">\r\n'
            f'  <!-- Session: {session_id} -->\r\n'
            f'  <menu id="{menu_name}" dtmf="true">\r\n'
            f'    <prompt bargein="true">\r\n'
            f'      <audio src="prompts/{menu_name}.wav">\r\n'
            f'        {prompt_text}\r\n'
            f'      </audio>\r\n'
            f'    </prompt>\r\n'
            f'{choices_xml}'
            f'    <noinput count="1">\r\n'
            f'      <prompt>I did not hear a selection. {prompt_text}</prompt>\r\n'
            f'      <reprompt/>\r\n'
            f'    </noinput>\r\n'
            f'    <noinput count="2">\r\n'
            f'      <prompt>Transferring you to an agent.</prompt>\r\n'
            f'      <goto next="#transfer_agent"/>\r\n'
            f'    </noinput>\r\n'
            f'    <nomatch>\r\n'
            f'      <prompt>That is not a valid selection. {prompt_text}</prompt>\r\n'
            f'      <reprompt/>\r\n'
            f'    </nomatch>\r\n'
            f'  </menu>\r\n'
            f'  <form id="transfer_agent">\r\n'
            f'    <transfer name="agent" dest="sip:{random.choice(self.QUEUES)}@cc.example.com"\r\n'
            f'             connecttimeout="30s" bridge="true"/>\r\n'
            f'  </form>\r\n'
            '</vxml>\r\n'
        )

        message = vxml.encode("utf-8")
        meta = {
            "ivr_document_type": "VoiceXML",
            "menu_name": menu_name,
            "menu_option_count": len(options),
            "session_id": session_id,
            "has_grammar": False,
            "has_tts_prompt": True,
            "has_noinput_handler": True,
            "has_nomatch_handler": True,
        }
        fields = self._build_xml_fields(vxml)
        return message, self._finalize_metadata("vxml_menu", message, meta, fields)

    def generate_mrcp_speak(self) -> Tuple[bytes, Dict]:
        """Generate an MRCP SPEAK request (TTS)."""
        channel_id = random.choice(self.channel_ids)
        request_id = self._next_request_id()
        voice = random.choice(self.TTS_VOICES)

        prompts = [
            "Thank you for calling. Your account balance is {balance} dollars.",
            "Please hold while I transfer you to the next available agent.",
            "Your estimated wait time is {wait} minutes.",
            "For quality assurance, this call may be recorded.",
            "Your payment of {amount} dollars has been processed successfully.",
            "I'm sorry, that is not a valid entry. Please try again.",
            "Your reference number is {ref}. Is there anything else I can help with?",
            "Thank you for your patience. An agent will be with you shortly.",
        ]
        prompt_template = random.choice(prompts)
        prompt_text = prompt_template.format(
            balance=f"{random.randint(10, 5000)}.{random.randint(0, 99):02d}",
            wait=str(random.randint(1, 15)),
            amount=f"{random.randint(10, 1000)}.{random.randint(0, 99):02d}",
            ref=f"REF{random.randint(100000, 999999)}",
        )

        # Build SSML body
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"\r\n'
            f'  xml:lang="en-US">\r\n'
            f'  <voice name="{voice}">\r\n'
            f'    <prosody rate="medium" pitch="default">\r\n'
            f'      {prompt_text}\r\n'
            f'    </prosody>\r\n'
            f'  </voice>\r\n'
            f'</speak>'
        )

        # MRCP message
        mrcp = (
            f"MRCP/2.0 {len(ssml) + 200} SPEAK {request_id}\r\n"
            f"Channel-Identifier: {channel_id}@speechsynth\r\n"
            f"Content-Type: application/ssml+xml\r\n"
            f"Content-Length: {len(ssml)}\r\n"
            f"Voice-Name: {voice}\r\n"
            f"Speech-Language: en-US\r\n"
            f"Kill-On-Barge-In: true\r\n"
            f"\r\n"
            f"{ssml}"
        )

        message = mrcp.encode("utf-8")
        meta = {
            "ivr_document_type": "MRCP",
            "mrcp_method": "SPEAK",
            "mrcp_version": "2.0",
            "channel_id": channel_id,
            "request_id": request_id,
            "voice": voice,
            "prompt_text": prompt_text,
            "has_ssml": True,
            "has_grammar": False,
            "has_tts_prompt": True,
        }
        fields = self._build_mrcp_fields(mrcp)
        return message, self._finalize_metadata("mrcp_speak", message, meta, fields)

    def generate_mrcp_recognize(self) -> Tuple[bytes, Dict]:
        """Generate an MRCP RECOGNIZE request (ASR/DTMF)."""
        channel_id = random.choice(self.channel_ids)
        request_id = self._next_request_id()

        # SRGS grammar for DTMF or speech
        grammar_types = ["dtmf_digits", "yes_no", "account_number", "date"]
        grammar_type = random.choice(grammar_types)

        if grammar_type == "dtmf_digits":
            grammar = (
                '<grammar mode="dtmf" version="1.0"\r\n'
                '  xmlns="http://www.w3.org/2001/06/grammar">\r\n'
                '  <rule id="digits" scope="public">\r\n'
                '    <one-of>\r\n'
                '      <item repeat="1-16"><ruleref uri="#digit"/></item>\r\n'
                '    </one-of>\r\n'
                '  </rule>\r\n'
                '  <rule id="digit">\r\n'
                '    <one-of>\r\n'
                '      <item>0</item><item>1</item><item>2</item>\r\n'
                '      <item>3</item><item>4</item><item>5</item>\r\n'
                '      <item>6</item><item>7</item><item>8</item>\r\n'
                '      <item>9</item>\r\n'
                '    </one-of>\r\n'
                '  </rule>\r\n'
                '</grammar>'
            )
        elif grammar_type == "yes_no":
            grammar = (
                '<grammar mode="voice" version="1.0"\r\n'
                '  xmlns="http://www.w3.org/2001/06/grammar"\r\n'
                '  xml:lang="en-US">\r\n'
                '  <rule id="yesno" scope="public">\r\n'
                '    <one-of>\r\n'
                '      <item>yes</item>\r\n'
                '      <item>yeah</item>\r\n'
                '      <item>correct</item>\r\n'
                '      <item>no</item>\r\n'
                '      <item>nope</item>\r\n'
                '      <item>negative</item>\r\n'
                '    </one-of>\r\n'
                '  </rule>\r\n'
                '</grammar>'
            )
        elif grammar_type == "account_number":
            grammar = (
                '<grammar mode="dtmf" version="1.0"\r\n'
                '  xmlns="http://www.w3.org/2001/06/grammar">\r\n'
                '  <rule id="account" scope="public">\r\n'
                '    <item repeat="10"><ruleref uri="#digit"/></item>\r\n'
                '  </rule>\r\n'
                '  <rule id="digit">\r\n'
                '    <one-of>\r\n'
                '      <item>0</item><item>1</item><item>2</item>\r\n'
                '      <item>3</item><item>4</item><item>5</item>\r\n'
                '      <item>6</item><item>7</item><item>8</item>\r\n'
                '      <item>9</item>\r\n'
                '    </one-of>\r\n'
                '  </rule>\r\n'
                '</grammar>'
            )
        else:  # date
            grammar = (
                '<grammar mode="voice" version="1.0"\r\n'
                '  xmlns="http://www.w3.org/2001/06/grammar"\r\n'
                '  xml:lang="en-US">\r\n'
                '  <rule id="date" scope="public">\r\n'
                '    <ruleref uri="#month"/> <ruleref uri="#day"/>\r\n'
                '  </rule>\r\n'
                '  <rule id="month">\r\n'
                '    <one-of>\r\n'
                '      <item>january</item><item>february</item>\r\n'
                '      <item>march</item><item>april</item>\r\n'
                '      <item>may</item><item>june</item>\r\n'
                '    </one-of>\r\n'
                '  </rule>\r\n'
                '  <rule id="day">\r\n'
                '    <one-of>\r\n'
                '      <item>first</item><item>second</item>\r\n'
                '      <item>third</item><item>fifteenth</item>\r\n'
                '      <item>thirtieth</item>\r\n'
                '    </one-of>\r\n'
                '  </rule>\r\n'
                '</grammar>'
            )

        mrcp = (
            f"MRCP/2.0 {len(grammar) + 250} RECOGNIZE {request_id}\r\n"
            f"Channel-Identifier: {channel_id}@speechrecog\r\n"
            f"Content-Type: application/srgs+xml\r\n"
            f"Content-Length: {len(grammar)}\r\n"
            f"No-Input-Timeout: 5000\r\n"
            f"Recognition-Timeout: 10000\r\n"
            f"Speech-Complete-Timeout: 1000\r\n"
            f"Sensitivity-Level: 0.5\r\n"
            f"Confidence-Threshold: 0.6\r\n"
            f"\r\n"
            f"{grammar}"
        )

        message = mrcp.encode("utf-8")
        meta = {
            "ivr_document_type": "MRCP",
            "mrcp_method": "RECOGNIZE",
            "mrcp_version": "2.0",
            "channel_id": channel_id,
            "request_id": request_id,
            "grammar_type": grammar_type,
            "grammar_mode": "dtmf" if grammar_type in ["dtmf_digits", "account_number"] else "voice",
            "has_grammar": True,
            "has_tts_prompt": False,
            "has_ssml": False,
        }
        fields = self._build_mrcp_fields(mrcp)
        return message, self._finalize_metadata("mrcp_recognize", message, meta, fields)

    def generate_ccxml_event(self) -> Tuple[bytes, Dict]:
        """Generate a CCXML event notification."""
        session_id = self._random_session_id()
        connection_id = f"conn-{uuid.uuid4().hex[:8]}"
        event_type = random.choice(self.CCXML_EVENTS)
        caller = self._random_phone()
        called = self._random_phone()

        event_data = ""
        if "connection" in event_type:
            event_data = (
                f'      <connectionid>{connection_id}</connectionid>\r\n'
                f'      <callerid>{caller}</callerid>\r\n'
                f'      <calledid>{called}</calledid>\r\n'
            )
        elif "dialog" in event_type:
            event_data = (
                f'      <dialogid>dlg-{uuid.uuid4().hex[:8]}</dialogid>\r\n'
                f'      <src>app://ivr/{random.choice(["main", "billing", "support"])}.vxml</src>\r\n'
            )
        elif "conference" in event_type:
            event_data = (
                f'      <conferenceid>conf-{random.randint(1000, 9999)}</conferenceid>\r\n'
                f'      <connectionid>{connection_id}</connectionid>\r\n'
            )
        elif "error" in event_type:
            event_data = (
                f'      <reason>Resource unavailable</reason>\r\n'
                f'      <connectionid>{connection_id}</connectionid>\r\n'
            )

        ccxml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<ccxml version="1.0" xmlns="http://www.w3.org/2002/09/ccxml">\r\n'
            f'  <eventprocessor>\r\n'
            f'    <transition event="{event_type}">\r\n'
            f'      <log expr="\'Event: {event_type} session: {session_id}\'"/>\r\n'
            f'{event_data}'
            f'    </transition>\r\n'
            f'  </eventprocessor>\r\n'
            '</ccxml>\r\n'
        )

        message = ccxml.encode("utf-8")
        meta = {
            "ivr_document_type": "CCXML",
            "ccxml_event": event_type,
            "session_id": session_id,
            "connection_id": connection_id,
            "caller": caller,
            "called": called,
            "has_grammar": False,
            "has_tts_prompt": False,
        }
        fields = self._build_xml_fields(ccxml)
        return message, self._finalize_metadata("ccxml_event", message, meta, fields)

    def generate_payment_capture_flow(self) -> Tuple[bytes, Dict]:
        """Generate a PCI-compliant VoiceXML payment capture flow."""
        session_id = self._random_session_id()
        brand = random.choice(list(self.CARD_BRANDS.keys()))
        card_length = self.CARD_BRANDS[brand]["length"]

        masking_mode = random.choice(["clamp", "flat_tone", "silence", "replace"])
        recording_action = random.choice(["pause", "stop"])

        vxml = (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<vxml version="2.1" xmlns="http://www.w3.org/2001/vxml">\r\n'
            f'  <!-- PCI Payment Capture Flow - Session: {session_id} -->\r\n'
            f'  <!-- PCI-DSS 4.0 Compliant: DTMF masking + recording pause -->\r\n'
            f'\r\n'
            f'  <property name="inputmodes" value="dtmf"/>\r\n'
            f'  <property name="interdigittimeout" value="5s"/>\r\n'
            f'  <property name="termchar" value="#"/>\r\n'
            f'\r\n'
            f'  <!-- Step 1: Pause recording for PCI scope -->\r\n'
            f'  <form id="pci_enter_scope">\r\n'
            f'    <block>\r\n'
            f'      <data name="recording_ctl"\r\n'
            f'            src="api://recording/{recording_action}"\r\n'
            f'            method="post"\r\n'
            f'            namelist="session_id"/>\r\n'
            f'      <data name="dtmf_mask_ctl"\r\n'
            f'            src="api://dtmf/mask/{masking_mode}"\r\n'
            f'            method="post"/>\r\n'
            f'      <goto next="#capture_card_number"/>\r\n'
            f'    </block>\r\n'
            f'  </form>\r\n'
            f'\r\n'
            f'  <!-- Step 2: Capture card number -->\r\n'
            f'  <form id="capture_card_number">\r\n'
            f'    <field name="card_number" type="digits?length={card_length}">\r\n'
            f'      <prompt bargein="true">\r\n'
            f'        Please enter your {card_length}-digit card number\r\n'
            f'        followed by the pound key.\r\n'
            f'      </prompt>\r\n'
            f'      <filled>\r\n'
            f'        <if cond="card_number.length == {card_length}">\r\n'
            f'          <goto next="#capture_expiry"/>\r\n'
            f'        <else/>\r\n'
            f'          <prompt>Invalid card number length. Please try again.</prompt>\r\n'
            f'          <clear namelist="card_number"/>\r\n'
            f'        </if>\r\n'
            f'      </filled>\r\n'
            f'      <noinput count="2">\r\n'
            f'        <prompt>Transferring you to an agent.</prompt>\r\n'
            f'        <goto next="#exit_pci_transfer"/>\r\n'
            f'      </noinput>\r\n'
            f'    </field>\r\n'
            f'  </form>\r\n'
            f'\r\n'
            f'  <!-- Step 3: Capture expiry date -->\r\n'
            f'  <form id="capture_expiry">\r\n'
            f'    <field name="expiry" type="digits?length=4">\r\n'
            f'      <prompt>Enter the expiration date as four digits. '
            f'Month then year.</prompt>\r\n'
            f'      <filled>\r\n'
            f'        <goto next="#capture_cvv"/>\r\n'
            f'      </filled>\r\n'
            f'    </field>\r\n'
            f'  </form>\r\n'
            f'\r\n'
            f'  <!-- Step 4: Capture CVV -->\r\n'
            f'  <form id="capture_cvv">\r\n'
            f'    <field name="cvv" type="digits?length={3 if brand != "amex" else 4}">\r\n'
            f'      <prompt>Enter the {3 if brand != "amex" else 4}-digit security '
            f'code from the {"back" if brand != "amex" else "front"} of your card.</prompt>\r\n'
            f'      <filled>\r\n'
            f'        <goto next="#process_payment"/>\r\n'
            f'      </filled>\r\n'
            f'    </field>\r\n'
            f'  </form>\r\n'
            f'\r\n'
            f'  <!-- Step 5: Process and exit PCI scope -->\r\n'
            f'  <form id="process_payment">\r\n'
            f'    <block>\r\n'
            f'      <data name="payment_result"\r\n'
            f'            src="api://payment/process"\r\n'
            f'            method="post"\r\n'
            f'            namelist="card_number expiry cvv"/>\r\n'
            f'      <data name="recording_resume"\r\n'
            f'            src="api://recording/resume"\r\n'
            f'            method="post"/>\r\n'
            f'      <data name="dtmf_unmask"\r\n'
            f'            src="api://dtmf/unmask"\r\n'
            f'            method="post"/>\r\n'
            f'      <prompt>Your payment has been processed. '
            f'Your reference number is <say-as interpret-as="characters">'
            f'REF{random.randint(100000, 999999)}</say-as>.</prompt>\r\n'
            f'    </block>\r\n'
            f'  </form>\r\n'
            f'\r\n'
            f'  <form id="exit_pci_transfer">\r\n'
            f'    <block>\r\n'
            f'      <data name="recording_resume" src="api://recording/resume" method="post"/>\r\n'
            f'      <data name="dtmf_unmask" src="api://dtmf/unmask" method="post"/>\r\n'
            f'    </block>\r\n'
            f'    <transfer name="agent" dest="sip:billing@cc.example.com"\r\n'
            f'             connecttimeout="30s" bridge="true"/>\r\n'
            f'  </form>\r\n'
            '</vxml>\r\n'
        )

        message = vxml.encode("utf-8")
        meta = {
            "ivr_document_type": "VoiceXML",
            "form_type": "payment_capture",
            "session_id": session_id,
            "card_brand": brand,
            "card_number_length": card_length,
            "dtmf_masking_mode": masking_mode,
            "recording_action": recording_action,
            "pci_compliant": True,
            "has_grammar": False,
            "has_tts_prompt": True,
        }
        fields = self._build_xml_fields(vxml)
        return message, self._finalize_metadata("payment_capture_flow", message, meta, fields)

    # ------------------------------------------------------------------
    # VoiceXML form builders
    # ------------------------------------------------------------------

    def _build_vxml_account_lookup(self, session_id: str, caller: str) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<vxml version="2.1" xmlns="http://www.w3.org/2001/vxml">\r\n'
            f'  <!-- Account Lookup - Session: {session_id} -->\r\n'
            f'  <form id="account_lookup">\r\n'
            f'    <field name="account_number" type="digits?length=10">\r\n'
            f'      <prompt bargein="true">\r\n'
            f'        Please enter your 10-digit account number\r\n'
            f'        followed by the pound key.\r\n'
            f'      </prompt>\r\n'
            f'      <filled>\r\n'
            f'        <data name="lookup_result"\r\n'
            f'              src="api://account/lookup"\r\n'
            f'              method="post"\r\n'
            f'              namelist="account_number"/>\r\n'
            f'        <prompt>\r\n'
            f'          Your account has been located.\r\n'
            f'          Transferring you to the next available agent.\r\n'
            f'        </prompt>\r\n'
            f'        <goto next="#transfer"/>\r\n'
            f'      </filled>\r\n'
            f'      <noinput count="2">\r\n'
            f'        <prompt>Transferring you to an agent.</prompt>\r\n'
            f'        <goto next="#transfer"/>\r\n'
            f'      </noinput>\r\n'
            f'    </field>\r\n'
            f'  </form>\r\n'
            f'  <form id="transfer">\r\n'
            f'    <transfer name="agent"\r\n'
            f'             dest="sip:{random.choice(self.QUEUES)}@cc.example.com"\r\n'
            f'             connecttimeout="30s" bridge="true"/>\r\n'
            f'  </form>\r\n'
            '</vxml>\r\n'
        )

    def _build_vxml_callback_request(self, session_id: str, caller: str) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<vxml version="2.1" xmlns="http://www.w3.org/2001/vxml">\r\n'
            f'  <!-- Callback Request - Session: {session_id} -->\r\n'
            f'  <form id="callback">\r\n'
            f'    <field name="callback_number" type="phone">\r\n'
            f'      <prompt>\r\n'
            f'        We can call you back when an agent is available.\r\n'
            f'        Please enter the phone number you would like us to\r\n'
            f'        call, followed by the pound key.\r\n'
            f'        Or press star to use your current number.\r\n'
            f'      </prompt>\r\n'
            f'      <filled>\r\n'
            f'        <if cond="callback_number == \'*\'">\r\n'
            f'          <assign name="callback_number" expr="\'{caller}\'"/>\r\n'
            f'        </if>\r\n'
            f'        <goto next="#confirm_callback"/>\r\n'
            f'      </filled>\r\n'
            f'    </field>\r\n'
            f'  </form>\r\n'
            f'  <form id="confirm_callback">\r\n'
            f'    <field name="confirm" type="boolean">\r\n'
            f'      <prompt>\r\n'
            f'        You will receive a callback at\r\n'
            f'        <say-as interpret-as="telephone">\r\n'
            f'          <value expr="callback_number"/>\r\n'
            f'        </say-as>.\r\n'
            f'        Is this correct? Say yes or no.\r\n'
            f'      </prompt>\r\n'
            f'      <filled>\r\n'
            f'        <data name="schedule_result"\r\n'
            f'              src="api://callback/schedule"\r\n'
            f'              method="post"\r\n'
            f'              namelist="callback_number"/>\r\n'
            f'        <prompt>Your callback has been scheduled. Goodbye.</prompt>\r\n'
            f'        <disconnect/>\r\n'
            f'      </filled>\r\n'
            f'    </field>\r\n'
            f'  </form>\r\n'
            '</vxml>\r\n'
        )

    def _build_vxml_survey(self, session_id: str, caller: str) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<vxml version="2.1" xmlns="http://www.w3.org/2001/vxml">\r\n'
            f'  <!-- Post-Call Survey - Session: {session_id} -->\r\n'
            f'  <form id="survey">\r\n'
            f'    <field name="satisfaction" type="digits?length=1">\r\n'
            f'      <prompt>\r\n'
            f'        On a scale of 1 to 5, with 5 being the highest,\r\n'
            f'        how would you rate your overall experience today?\r\n'
            f'      </prompt>\r\n'
            f'      <filled>\r\n'
            f'        <goto next="#survey_q2"/>\r\n'
            f'      </filled>\r\n'
            f'    </field>\r\n'
            f'  </form>\r\n'
            f'  <form id="survey_q2">\r\n'
            f'    <field name="recommend" type="digits?length=1">\r\n'
            f'      <prompt>\r\n'
            f'        How likely are you to recommend us to a friend?\r\n'
            f'        Press 1 for not likely, 5 for very likely.\r\n'
            f'      </prompt>\r\n'
            f'      <filled>\r\n'
            f'        <data name="survey_result"\r\n'
            f'              src="api://survey/submit"\r\n'
            f'              method="post"\r\n'
            f'              namelist="satisfaction recommend"/>\r\n'
            f'        <prompt>Thank you for your feedback. Goodbye.</prompt>\r\n'
            f'        <disconnect/>\r\n'
            f'      </filled>\r\n'
            f'    </field>\r\n'
            f'  </form>\r\n'
            '</vxml>\r\n'
        )

    def _build_vxml_balance_inquiry(self, session_id: str, caller: str) -> str:
        balance = f"{random.randint(10, 50000)}.{random.randint(0, 99):02d}"
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\r\n'
            '<vxml version="2.1" xmlns="http://www.w3.org/2001/vxml">\r\n'
            f'  <!-- Balance Inquiry - Session: {session_id} -->\r\n'
            f'  <form id="balance_inquiry">\r\n'
            f'    <field name="account_number" type="digits?length=10">\r\n'
            f'      <prompt>\r\n'
            f'        To check your balance, please enter your\r\n'
            f'        10-digit account number.\r\n'
            f'      </prompt>\r\n'
            f'      <filled>\r\n'
            f'        <data name="balance_result"\r\n'
            f'              src="api://account/balance"\r\n'
            f'              method="post"\r\n'
            f'              namelist="account_number"/>\r\n'
            f'        <prompt>\r\n'
            f'          Your current balance is\r\n'
            f'          <say-as interpret-as="currency">${balance}</say-as>.\r\n'
            f'        </prompt>\r\n'
            f'        <goto next="#post_balance_menu"/>\r\n'
            f'      </filled>\r\n'
            f'    </field>\r\n'
            f'  </form>\r\n'
            f'  <menu id="post_balance_menu">\r\n'
            f'    <prompt>Press 1 to make a payment. Press 2 to hear recent transactions. '
            f'Press 3 to speak with an agent.</prompt>\r\n'
            f'    <choice dtmf="1" next="#payment">Make a payment</choice>\r\n'
            f'    <choice dtmf="2" next="#transactions">Recent transactions</choice>\r\n'
            f'    <choice dtmf="3" next="#transfer">Speak with agent</choice>\r\n'
            f'  </menu>\r\n'
            '</vxml>\r\n'
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_xml_fields(self, xml_str: str) -> List[Dict]:
        """Build field metadata from an XML document."""
        xml_bytes = xml_str.encode("utf-8")
        fields = [{"name": "xml_document", "offset": 0, "length": len(xml_bytes)}]

        decl_end = xml_str.find("?>")
        if decl_end > 0:
            fields.append({"name": "xml_declaration", "offset": 0, "length": decl_end + 2})

        first_tag = xml_str.find("<", decl_end + 2 if decl_end > 0 else 0)
        if first_tag >= 0:
            tag_end = xml_str.find(">", first_tag)
            if tag_end > 0:
                fields.append({
                    "name": "root_element",
                    "offset": first_tag,
                    "length": tag_end - first_tag + 1,
                })

        return fields

    def _build_mrcp_fields(self, mrcp_str: str) -> List[Dict]:
        """Build field metadata from an MRCP message."""
        mrcp_bytes = mrcp_str.encode("utf-8")
        fields = [{"name": "mrcp_message", "offset": 0, "length": len(mrcp_bytes)}]

        # Find request line
        first_line_end = mrcp_str.find("\r\n")
        if first_line_end > 0:
            fields.append({"name": "request_line", "offset": 0, "length": first_line_end})

        # Find header/body separator
        body_sep = mrcp_str.find("\r\n\r\n")
        if body_sep > 0:
            fields.append({
                "name": "headers",
                "offset": first_line_end + 2,
                "length": body_sep - first_line_end - 2,
            })
            if body_sep + 4 < len(mrcp_bytes):
                fields.append({
                    "name": "body",
                    "offset": body_sep + 4,
                    "length": len(mrcp_bytes) - body_sep - 4,
                })

        return fields

    def _finalize_metadata(
        self, msg_type: str, message: bytes, meta: Dict, fields: list
    ) -> Dict:
        """Create the final metadata dict."""
        meta.update({
            "protocol": "ivr",
            "message_type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message),
            "fields": fields,
            "hash": hashlib.sha256(message).hexdigest(),
        })
        return meta

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of IVR documents."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = [
            ("vxml_form", self.generate_vxml_form, 0.20),
            ("vxml_menu", self.generate_vxml_menu, 0.20),
            ("mrcp_speak", self.generate_mrcp_speak, 0.15),
            ("mrcp_recognize", self.generate_mrcp_recognize, 0.15),
            ("ccxml_event", self.generate_ccxml_event, 0.15),
            ("payment_capture_flow", self.generate_payment_capture_flow, 0.15),
        ]

        dataset_metadata = {
            "protocol": "ivr",
            "version": "VoiceXML 2.1/MRCP v2/CCXML 1.0",
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

                bin_path = output_path / f"ivr_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = output_path / f"ivr_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate IVR dataset."""
    generator = IVRGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "ivr"

    print("Generating IVR dataset...")
    metadata = generator.generate_dataset(num_samples=1000, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} samples")
    print(f"Output directory: {output_dir}")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")


if __name__ == "__main__":
    main()
