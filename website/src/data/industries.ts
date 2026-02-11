export interface Industry {
  id: string;
  title: string;
  tagline: string;
  description: string;
  protocols: string[];
  href: string;
  color: string;
  icon: string;
}

export const industries: Industry[] = [
  {
    id: 'banking',
    title: 'Banking & Financial Services',
    tagline: 'Quantum-safe protection for financial infrastructure',
    description: 'Protect mainframe transaction systems, discover undocumented payment protocols, and ensure PCI-DSS/BASEL-III compliance with domain-optimized PQC for SWIFT and SEPA latency constraints.',
    protocols: ['ISO 8583', 'SWIFT MT/MX', 'FIX 4.x/5.x', 'ACH/NACHA', 'SEPA', 'Fedwire'],
    href: '/industries/banking',
    color: 'from-blue-500 to-indigo-500',
    icon: 'M2.25 18.75a60.07 60.07 0 0115.797 2.101c.727.198 1.453-.342 1.453-1.096V18.75M3.75 4.5v.75A.75.75 0 013 6h-.75m0 0v-.375c0-.621.504-1.125 1.125-1.125H20.25M2.25 6v9m18-10.5v.75c0 .414.336.75.75.75h.75m-1.5-1.5h.375c.621 0 1.125.504 1.125 1.125v9.75c0 .621-.504 1.125-1.125 1.125h-.375m1.5-1.5H21a.75.75 0 00-.75.75v.75m0 0H3.75m0 0h-.375a1.125 1.125 0 01-1.125-1.125V15m1.5 1.5v-.75A.75.75 0 003 15h-.75M15 10.5a3 3 0 11-6 0 3 3 0 016 0zm3 0h.008v.008H18V10.5zm-12 0h.008v.008H6V10.5z',
  },
  {
    id: 'healthcare',
    title: 'Healthcare & Medical Devices',
    tagline: 'Non-invasive protection for medical systems',
    description: 'Secure medical devices and EHR systems without firmware changes. Discover HL7/FHIR communications, apply PQC optimized for power-constrained devices, and automate HIPAA compliance.',
    protocols: ['HL7 v2/v3', 'FHIR', 'DICOM', 'X12 837/835', 'IEEE 11073'],
    href: '/industries/healthcare',
    color: 'from-emerald-500 to-green-500',
    icon: 'M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z',
  },
  {
    id: 'critical-infrastructure',
    title: 'Critical Infrastructure & SCADA',
    tagline: 'Quantum-safe industrial control systems',
    description: 'Protect SCADA networks and power grid communications with quantum-safe encryption optimized for IEC 62351. Autonomous threat response with safety constraints for industrial environments.',
    protocols: ['Modbus TCP/RTU', 'DNP3', 'IEC 61850', 'OPC UA', 'BACnet', 'Profinet'],
    href: '/industries/critical-infrastructure',
    color: 'from-amber-500 to-orange-500',
    icon: 'M3.75 21h16.5M4.5 3h15M5.25 3v18m13.5-18v18M9 6.75h1.5m-1.5 3h1.5m-1.5 3h1.5m3-6H15m-1.5 3H15m-1.5 3H15M9 21v-3.375c0-.621.504-1.125 1.125-1.125h3.75c.621 0 1.125.504 1.125 1.125V21',
  },
  {
    id: 'automotive',
    title: 'Automotive & Connected Vehicles',
    tagline: 'Quantum-safe V2X communications',
    description: 'Ultra-low latency PQC for vehicle-to-everything communications. Protect connected vehicles with crypto agility designed for 15-20 year vehicle lifespans.',
    protocols: ['V2X', 'IEEE 1609.2', 'CAN', 'LIN', 'FlexRay'],
    href: '/industries/automotive',
    color: 'from-red-500 to-rose-500',
    icon: 'M8.25 18.75a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h6m-9 0H3.375a1.125 1.125 0 01-1.125-1.125V14.25m17.25 4.5a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0H21M3.375 14.25h3.86a1.5 1.5 0 001.342-.83l1.065-2.13a1.5 1.5 0 011.342-.83h3.03a1.5 1.5 0 011.342.83l1.065 2.13a1.5 1.5 0 001.342.83h3.86',
  },
  {
    id: 'aviation',
    title: 'Aviation & Aerospace',
    tagline: 'Securing the skies for the quantum era',
    description: 'Protect aircraft communications with quantum-resistant cryptography designed for 50-year aviation lifecycles. Prevent ADS-B spoofing and secure air traffic management.',
    protocols: ['ADS-B', 'ACARS', 'CPDLC', 'ARINC 429', 'Mode S'],
    href: '/industries/aviation',
    color: 'from-sky-500 to-cyan-500',
    icon: 'M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5',
  },
  {
    id: 'telecommunications',
    title: 'Telecommunications',
    tagline: 'Quantum-safe 5G and network infrastructure',
    description: 'Secure telecom infrastructure with PQC-protected signaling, carrier authentication, and zero-touch threat response for network operations.',
    protocols: ['Diameter', 'SS7/SIGTRAN', 'SIP', 'GTP', 'SMPP', 'RADIUS'],
    href: '/industries/telecommunications',
    color: 'from-violet-500 to-purple-500',
    icon: 'M8.288 15.038a5.25 5.25 0 017.424 0M5.106 11.856c3.807-3.808 9.98-3.808 13.788 0M1.924 8.674c5.565-5.565 14.587-5.565 20.152 0M12.53 18.22l-.53.53-.53-.53a.75.75 0 011.06 0z',
  },
];
