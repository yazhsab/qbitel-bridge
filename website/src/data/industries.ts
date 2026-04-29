export interface Industry {
  id: string;
  title: string;
  tagline: string;
  description: string;
  protocols: string[];
  href: string;
  color: string;
  icon: string;
  impact?: string;
}

export const industries: Industry[] = [
  {
    id: 'bpo-call-center',
    title: 'BPO & Call Centers',
    tagline: 'Voice-channel fraud and compliance protection',
    description: 'Protect high-volume SIP/RTP, IVR, CRM, and recording workflows from toll fraud, payment leakage, voice phishing, and agent misuse while producing PCI/HIPAA-ready evidence for client audits.',
    protocols: ['SIP', 'RTP/SRTP', 'WebRTC', 'IVR logs', 'PCI voice capture', 'CRM APIs'],
    href: '/industries/bpo-call-center',
    color: 'from-teal-500 to-cyan-500',
    icon: 'M2.25 6.75c0 8.284 6.716 15 15 15h2.25a2.25 2.25 0 002.25-2.25v-1.372c0-.516-.351-.966-.852-1.091l-4.423-1.106a1.125 1.125 0 00-1.173.417l-.97 1.293a1.125 1.125 0 01-1.21.38 12.035 12.035 0 01-7.143-7.143 1.125 1.125 0 01.38-1.21l1.293-.97c.36-.27.526-.733.417-1.173L6.963 3.102A1.125 1.125 0 005.872 2.25H4.5A2.25 2.25 0 002.25 4.5v2.25z',
    impact: 'Fraud reduction and client audit evidence',
  },
  {
    id: 'banking',
    title: 'Banking & Financial Services',
    tagline: 'Quantum-safe financial infrastructure',
    description: 'Protect mainframe transaction systems running on undocumented protocols. Discover legacy payment flows, apply PQC optimized for SWIFT and SEPA latency constraints, and automate PCI-DSS/BASEL-III compliance continuously.',
    protocols: ['ISO 8583', 'SWIFT MT/MX', 'FIX 4.x/5.x', 'ACH/NACHA', 'SEPA', 'Fedwire'],
    href: '/industries/banking',
    color: 'from-blue-500 to-indigo-500',
    icon: 'M2.25 18.75a60.07 60.07 0 0115.797 2.101c.727.198 1.453-.342 1.453-1.096V18.75M3.75 4.5v.75A.75.75 0 013 6h-.75m0 0v-.375c0-.621.504-1.125 1.125-1.125H20.25M2.25 6v9m18-10.5v.75c0 .414.336.75.75.75h.75m-1.5-1.5h.375c.621 0 1.125.504 1.125 1.125v9.75c0 .621-.504 1.125-1.125 1.125h-.375m1.5-1.5H21a.75.75 0 00-.75.75v.75m0 0H3.75m0 0h-.375a1.125 1.125 0 01-1.125-1.125V15m1.5 1.5v-.75A.75.75 0 003 15h-.75M15 10.5a3 3 0 11-6 0 3 3 0 016 0zm3 0h.008v.008H18V10.5zm-12 0h.008v.008H6V10.5z',
    impact: 'Protocol discovery: $2-10M to $200K',
  },
  {
    id: 'iot',
    title: 'IoT & Connected Devices',
    tagline: 'Device trust for unmanaged fleets',
    description: 'Discover unmanaged devices, proprietary protocols, weak crypto, and firmware risk across fragmented IoT estates. Build device trust graphs and protect constrained devices through gateway-first controls.',
    protocols: ['MQTT', 'CoAP', 'BLE', 'Zigbee', 'LoRaWAN', 'Proprietary binary'],
    href: '/industries/iot',
    color: 'from-lime-500 to-emerald-500',
    icon: 'M8.25 3v1.5M4.5 8.25H3m1.5 7.5H3m5.25 4.5V21m7.5-18v1.5m3.75 3.75H21m-1.5 7.5H21m-5.25 4.5V21M7.5 7.5h9v9h-9v-9z',
    impact: 'Device inventory and crypto posture',
  },
  {
    id: 'defense',
    title: 'Defense & Mission Systems',
    tagline: 'Air-gapped modernization and CNSA 2.0 readiness',
    description: 'Run offline discovery and modernization for mission networks, tactical links, classified environments, and long-lived defense systems with strict approval gates and signed evidence packages.',
    protocols: ['Tactical data links', 'Serial links', 'Legacy IP', 'MIL-STD messages', 'Custom mission protocols'],
    href: '/industries/defense',
    color: 'from-slate-500 to-zinc-400',
    icon: 'M12 2.25c2.25 2.25 5.25 3 8.25 3v6.75c0 5.25-3.75 8.25-8.25 9.75-4.5-1.5-8.25-4.5-8.25-9.75V5.25c3 0 6-.75 8.25-3z',
    impact: 'CNSA 2.0 and ATO evidence',
  },
  {
    id: 'healthcare',
    title: 'Healthcare & Medical Devices',
    tagline: 'Non-invasive protection for patient safety',
    description: 'Secure medical devices and EHR systems without firmware modifications or downtime. Discover HL7/FHIR communications, apply lightweight PQC for power-constrained devices, and automate HIPAA compliance with cryptographic evidence.',
    protocols: ['HL7 v2/v3', 'FHIR R4', 'DICOM', 'X12 837/835', 'IEEE 11073'],
    href: '/industries/healthcare',
    color: 'from-emerald-500 to-green-500',
    icon: 'M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z',
    impact: 'Zero firmware changes required',
  },
  {
    id: 'critical-infrastructure',
    title: 'Critical Infrastructure & SCADA',
    tagline: 'Quantum-safe industrial control systems',
    description: 'Protect SCADA networks and power grid communications with quantum-safe encryption optimized for IEC 62351 timing constraints. Autonomous threat response with industrial safety interlocks prevents both cyber attacks and unsafe automated actions.',
    protocols: ['Modbus TCP/RTU', 'DNP3', 'IEC 61850', 'OPC UA', 'BACnet', 'Profinet'],
    href: '/industries/critical-infrastructure',
    color: 'from-amber-500 to-orange-500',
    icon: 'M3.75 21h16.5M4.5 3h15M5.25 3v18m13.5-18v18M9 6.75h1.5m-1.5 3h1.5m-1.5 3h1.5m3-6H15m-1.5 3H15m-1.5 3H15M9 21v-3.375c0-.621.504-1.125 1.125-1.125h3.75c.621 0 1.125.504 1.125 1.125V21',
    impact: 'NERC CIP automated compliance',
  },
  {
    id: 'automotive',
    title: 'Automotive & Connected Vehicles',
    tagline: 'Quantum-safe V2X for 20-year vehicle lifespans',
    description: 'Ultra-low latency PQC for vehicle-to-everything communications with crypto agility designed for 15-20 year automotive lifecycles. Protect connected vehicles from spoofing, replay, and man-in-the-middle attacks at wire speed.',
    protocols: ['V2X', 'IEEE 1609.2', 'CAN', 'LIN', 'FlexRay'],
    href: '/industries/automotive',
    color: 'from-red-500 to-rose-500',
    icon: 'M8.25 18.75a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h6m-9 0H3.375a1.125 1.125 0 01-1.125-1.125V14.25m17.25 4.5a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0H21M3.375 14.25h3.86a1.5 1.5 0 001.342-.83l1.065-2.13a1.5 1.5 0 011.342-.83h3.03a1.5 1.5 0 011.342.83l1.065 2.13a1.5 1.5 0 001.342.83h3.86',
    impact: 'Crypto agility for vehicle lifecycle',
  },
  {
    id: 'aviation',
    title: 'Aviation & Aerospace',
    tagline: 'Securing 50-year aviation lifecycles',
    description: 'Protect aircraft communications with quantum-resistant cryptography designed for multi-decade aviation equipment cycles. Prevent ADS-B spoofing, secure ACARS messaging, and protect air traffic management systems with certified PQC.',
    protocols: ['ADS-B', 'ACARS', 'CPDLC', 'ARINC 429', 'Mode S'],
    href: '/industries/aviation',
    color: 'from-sky-500 to-cyan-500',
    icon: 'M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5',
    impact: 'Anti-spoofing for ADS-B systems',
  },
  {
    id: 'telecommunications',
    title: 'Telecommunications',
    tagline: 'Quantum-safe 5G and core networks',
    description: 'Secure telecom core infrastructure with PQC-protected signaling across SS7, Diameter, and GTP. Zero-touch threat response for network operations with carrier-grade reliability and sub-millisecond PQC overhead.',
    protocols: ['Diameter', 'SS7/SIGTRAN', 'SIP', 'GTP', 'SMPP', 'RADIUS'],
    href: '/industries/telecommunications',
    color: 'from-violet-500 to-purple-500',
    icon: 'M8.288 15.038a5.25 5.25 0 017.424 0M5.106 11.856c3.807-3.808 9.98-3.808 13.788 0M1.924 8.674c5.565-5.565 14.587-5.565 20.152 0M12.53 18.22l-.53.53-.53-.53a.75.75 0 011.06 0z',
    impact: 'Carrier-grade PQC at <1ms',
  },
];
