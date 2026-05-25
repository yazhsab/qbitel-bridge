# QBITEL Industry Video Campaign

Date: April 30, 2026
Status: Enterprise-grade creator and publishing brief

## Campaign Goal

Create narrated, enterprise-grade animated explainer videos that make QBITEL immediately understandable by vertical.

Each video follows the same simple story:

1. The industry has critical systems it cannot easily replace.
2. Those systems hide protocol, fraud, cyber, crypto, and evidence risk.
3. QBITEL creates a governed path: Discover, Understand, Modernize, Protect, Prove.
4. The output is not just insight. It is usable evidence, modernization assets, and a controlled rollout path.

## Recommended Video Set

| Video | Solution Pack | Main Audience | Best Channel |
|---|---|---|---|
| BPO and Call Centers | QBITEL Voice Shield | CX leaders, BPO operators, CISOs | LinkedIn, YouTube, sales deck |
| IoT and Connected Devices | QBITEL Device Trust | IoT platform, product security, operations | LinkedIn, YouTube, partner outreach |
| Defense and Mission Systems | QBITEL Mission Bridge | Mission modernization, architecture, high-assurance teams | Private pitch, conference deck |
| Banking and Insurance | QBITEL Mainframe Shield | CIO, CISO, mainframe, payments | LinkedIn, investor deck, sales deck |
| OT and Critical Infrastructure | QBITEL Industrial Shield | OT security, plant operations, utilities | LinkedIn, conference, partner pitch |
| Healthcare and Medical Devices | QBITEL MedTech Bridge | Healthcare CISO, biomedical engineering, compliance | LinkedIn, webinar |
| Telecommunications | QBITEL Signaling Shield | Telecom security, network engineering, fraud teams | LinkedIn, operator pitch |
| Aviation and Automotive | QBITEL Mobility Shield | Mobility security, platform engineering, certification teams | LinkedIn, technical pitch |

## Rendered Output Folder

Rendered MP4s should be placed here:

`docs/marketing/videos/`

Expected filenames:

- `QBITEL_BPOVoiceShield.mp4`
- `QBITEL_IoTDeviceTrust.mp4`
- `QBITEL_DefenseMissionBridge.mp4`
- `QBITEL_BankingMainframeShield.mp4`
- `QBITEL_IndustrialShield.mp4`
- `QBITEL_HealthcareMedTechBridge.mp4`
- `QBITEL_TelecomSignalingShield.mp4`
- `QBITEL_MobilityShield.mp4`

## Source Project

The Remotion source lives here:

`docs/brochures/my-video/`

Key files:

- `src/IndustryUsecaseVideo.tsx`: reusable animated video template
- `src/industry-data.ts`: industry-specific copy, risks, protocols, workflow, and evidence
- `src/Root.tsx`: Remotion composition registration
- `scripts/render-industry-videos.mjs`: renders all industry videos
- `scripts/generate-industry-voiceovers.mjs`: generates ElevenLabs narration MP3 files

Render all videos:

```bash
cd docs/brochures/my-video
npm run render-industries
```

Generate voiceovers and render narrated videos:

```bash
cd docs/brochures/my-video
printf 'ELEVENLABS_API_KEY=your_key_here\nELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM\n' > .env
npm run generate-industry-voiceovers
npm run render-industries-voiced
```

Render a single video:

```bash
cd docs/brochures/my-video
npx remotion render BPOVoiceShield --codec h264 ../../marketing/videos/QBITEL_BPOVoiceShield.mp4
```

## Video Structure

Each video is approximately 50 seconds:

1. Executive hook and outcome
2. Buyer pressure: audit, fraud, uptime, crypto, and trust
3. Self-testable Demo Lab workflow with realistic sample input
4. Evidence pack and board-level outcomes
5. Investor and buyer call to action

Core CTA:

Launch Demo Lab. Test with your own samples. Generate executive evidence.

## Master Caption

Most enterprises cannot simply rip out the systems that run their most critical operations.

QBITEL Bridge helps teams discover, understand, modernize, protect, and prove control over those systems across BPO, IoT, defense, banking, OT, healthcare, telecom, and mobility environments.

The result: critical systems that are visible, protected, and provable.

## Short Captions

### BPO

Call-center risk is no longer only inside the agent desktop. It lives across SIP, RTP, IVR, recordings, PCI workflows, and client evidence. QBITEL Voice Shield makes those flows visible, protected, and provable.

### IoT

You cannot secure device fleets you cannot explain. QBITEL Device Trust discovers devices, protocols, firmware exposure, weak crypto, and evidence gaps across IoT estates.

### Defense

Mission systems need modernization without reckless replacement. QBITEL Mission Bridge creates a controlled path from unknown dependencies to governed, evidence-backed control.

### Banking

Mainframes are not the problem. Unknown payment dependencies are. QBITEL Mainframe Shield helps banks discover, modernize, protect, and prove control over critical financial flows.

### OT

Critical infrastructure needs visibility first and disruption last. QBITEL Industrial Shield supports protocol inventory, governed overlays, and evidence-led rollout for OT environments.

### Healthcare

Patient-care workflows depend on systems that must not fail silently. QBITEL MedTech Bridge helps secure legacy devices, EHR flows, and clinical integrations with evidence.

### Telecom

Telecom trust depends on protocol intelligence at network scale. QBITEL Signaling Shield helps make signaling, mediation, voice, and fraud-control flows explainable and provable.

### Mobility

Connected vehicles and aviation systems need long-lifecycle trust. QBITEL Mobility Shield supports protocol discovery, low-change protection, and certification-ready evidence.

## Posting Order

1. BPO and Call Centers
2. Banking and Insurance
3. IoT and Connected Devices
4. OT and Critical Infrastructure
5. Defense and Mission Systems
6. Healthcare and Medical Devices
7. Telecommunications
8. Aviation and Automotive

This order starts with the clearest commercial ROI and expands into strategic markets.

## Claims Guardrails

Avoid:

- fully autonomous security without humans
- guaranteed quantum-proof forever
- instant compliance
- replacement of all legacy systems
- production readiness for every classified or safety-critical environment without validation

Use:

- governed protection
- evidence-backed modernization
- protocol intelligence
- controlled rollout
- post-quantum migration path
- approval-led deployment
