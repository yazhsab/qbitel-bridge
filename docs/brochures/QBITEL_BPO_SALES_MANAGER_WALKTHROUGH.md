# QBITEL Bridge — Sales Manager Walkthrough
## Selling Into BPOs With Multiple Dialers, CRMs, and Ticketing Systems

> **Audience:** Sales Managers, Account Executives, Solution Consultants reaching out to BPOs and Contact Centers. Use this as your pre-call prep, mid-call reference, and post-call follow-up template.
>
> **Version 1.0 | February 2026 | Confidential — For Authorized Recipients Only**

---

## 1. How to Use This Walkthrough

This document is built for three moments in your sales cycle:

| Moment | What to use |
|---|---|
| **Pre-call (30 min before)** | Sections 2 (Buying Committee), 3 (Pitch tiers), and 4 (Discovery Questions) |
| **Mid-call** | Section 5 (Demo walk), Section 7 (ROI), and Section 8 (Objections) — kept short on purpose so you can skim them while listening |
| **Post-call (within 24 hrs)** | Section 10 (Next-Step Menu) — pick the right commitment level for the prospect and send the follow-up |

The defining peculiarity of selling to a BPO is that **the BPO isn't one customer — it's a portfolio of clients, each with their own dialer, CRM, and ticketing stack**. The conversation that wins is the one that demonstrates QBITEL covers all of them simultaneously, with isolation. Section 4 (Discovery) and Section 5 (Demo) are written specifically around that.

---

## 2. Who You're Selling To: The BPO Buying Committee

A BPO deal has three cohorts. You need at least one champion in each before close.

### 2.1 Business Buyers — *the people who control the spend*

**Roles:** CEO, CFO, COO, VP Operations, Contact Center Director.

**What they care about:**
- Margin compression from rising compliance and audit costs
- Toll-fraud losses showing up on carrier bills they didn't budget for
- Winning new client contracts that have stricter security clauses
- Avoiding the breach that ends the relationship with their biggest client

**Opening question (Business buyer):**
> *"How many of your client contracts in the last 18 months have come in with security clauses you couldn't have signed three years ago — and how is that changing your win rate on new RFPs?"*

**What makes them say yes:** A clear ROI story — toll fraud payback in 30–60 days, PCI audit cost reduction of $500K–$2M annually, and the ability to win new contracts faster because the security answer is already done.

### 2.2 Technical Buyers — *the people who decide whether it works*

**Roles:** CTO, IT Director, VP Engineering, Network Architect, Infrastructure Lead.

**What they care about:**
- "Does this actually work with Avaya, Cisco, Genesys, Asterisk, Salesforce, Zoho, HubSpot, ServiceNow, Freshdesk, and Zendesk *simultaneously*?"
- Performance impact on voice quality
- Whether they have to retrain their NOC
- Reversibility — "if we don't like it, can we pull it out?"

**Opening question (Technical buyer):**
> *"Take me through what your largest three clients run for dialers, CRMs, and ticketing — is it all on one stack, or do you operate three different ones in parallel? What's that like to keep secured?"*

**What makes them say yes:** Network-overlay deployment (no infrastructure replacement), <2ms voice latency, working integrations with every major dialer/CRM/ticketing system, and a clean revert path.

### 2.3 Security & Compliance Buyers — *the people who can veto*

**Roles:** CISO, Compliance Officer, Risk Lead, Legal, DPO.

**What they care about:**
- Per-client compliance isolation (PCI-DSS for one client, HIPAA for another, SOC 2 for a third — without leakage)
- Evidence-pack quality their QSAs and auditors will accept
- Post-quantum readiness (regulators are starting to ask)
- Audit trail integrity — tamper evidence, retention, chain of custody

**Opening question (Security buyer):**
> *"When your QSA does the PCI scoping walk this year, how much of your environment is in scope — and how much of that is driven by recordings, agent desktops, and CRM integrations that touch cardholder data?"*

**What makes them say yes:** Per-tenant key isolation, 9 compliance frameworks automated, blockchain-backed audit trails, NIST Level 5 PQC, on-premise LLM option (no customer data leaving their network).

---

## 3. The 30-Second, 3-Minute, and 30-Minute Pitches

### 3.1 The 30-Second Pitch *(elevator / cold open)*

> *"BPOs are uniquely exposed: every one of your enterprise clients mandates a different dialer, CRM, and ticketing system, so you're running Avaya for the bank, Genesys for the healthcare payer, Asterisk for the retailer — and each one needs its own PCI, HIPAA, and SOC 2 evidence. QBITEL Bridge sits at the network layer, discovers every protocol across every client stack in 2 to 4 hours, wraps them in quantum-safe encryption, and produces per-client compliance reports in under 10 minutes. Deploy in 4 to 6 hours, no PBX replacement, no agent retraining. Toll fraud alone usually pays for it in 30 to 60 days."*

### 3.2 The 3-Minute Pitch *(first qualified meeting)*

Three converging problems your CTO and CISO are already worrying about:

1. **Quantum harvest is happening now.** Adversaries are recording your encrypted call traffic today — including the calls from this morning — to decrypt in 5–10 years when quantum computers arrive. Your SOX/HIPAA-mandated 7-year recording retention means everything you record this week is still sitting there when quantum decryption becomes viable.

2. **Toll fraud is bleeding $10B+ a year from this industry.** A single compromised PBX trunk can rack up $50K in fraudulent premium-rate calls over one weekend. Most BPOs find out at the next invoice — 72 hours too late.

3. **Multi-client compliance is eating your margin.** Every enterprise client demands their own audit. Every audit demands its own evidence. Three clients means three audits per year, each one consuming engineering time.

**QBITEL Bridge solves all three at once.** It's a network-overlay platform — no PBX replacement — that uses AI to discover every protocol your client stacks use (any dialer, any CRM, any ticketing system), wraps them in NIST Level 5 post-quantum encryption, detects toll fraud in under a second, blocks data exfiltration at the agent desktop, and generates per-client compliance evidence packs in under 10 minutes.

Deployment is 4–6 hours, zero downtime, nothing replaced. Toll fraud savings alone usually fund the platform in 30 to 60 days. We have a free 2-hour Discovery Assessment that produces a real protocol map of your environment — would that be valuable as a starting point?

### 3.3 The 30-Minute Pitch *(full deck / first technical meeting)*

Use the [QBITEL_BRIDGE_BPO_MARKETING_PITCH.md](QBITEL_BRIDGE_BPO_MARKETING_PITCH.md) as the structural spine. The flow:

1. **Opening hook** (4 min) — three converging threats (quantum harvest, toll fraud, remote-workforce risk).
2. **The BPO portfolio problem** (4 min) — multiple clients = multiple dialers + CRMs + ticketing + compliance frameworks. Make them describe their own. *(This is the section where they sell themselves on the problem.)*
3. **QBITEL in one diagram** (3 min) — network overlay, 5 modules, discover → understand → modernize → protect → prove.
4. **The Acme Outsourcing walk** (8 min) — Section 5 below. Concrete, vivid, multi-stack.
5. **ROI math** (5 min) — Section 7 below.
6. **Compliance + per-tenant isolation** (3 min) — they're going to ask, get ahead of it.
7. **Next step** (3 min) — close on the free 2-hour Discovery Assessment.

---

## 4. Discovery Questions — Mapping Their Client Portfolio

This is your differentiator. Most security vendors ask discovery questions about *the BPO's* infrastructure. The right questions for a BPO are about *each of their clients' infrastructure*. Asking these correctly signals you understand the business.

### 4.1 Portfolio Shape

- How many enterprise clients are you running operations for right now?
- What's the largest in seat count? Smallest?
- Of those, how many are PCI-driven? HIPAA-driven? SOX-driven? GDPR-driven?
- How many separate annual audits does your compliance team prep for across all clients?

### 4.2 Dialer / Telephony Diversity

- Walk me through the top three clients. What dialer or contact-center platform does each one mandate?
- Do you run Avaya, Cisco, Genesys, Asterisk — or a mix? *(Probe: "When a new client comes in mandating their preferred stack, what does the onboarding look like for you?")*
- Any legacy PBX still in production? Any mainframe terminal sessions (TN3270e/TN5250) the agents touch?

### 4.3 CRM Diversity

- Does each client bring their own CRM (Salesforce, Zoho, HubSpot, Zendesk, Dynamics, custom)?
- How does the agent desktop swap between them — federated SSO, separate logins, virtual desktops?
- Where does cardholder data, PHI, or PII actually live during a call — in the CRM record, in the recording, in the agent's clipboard?

### 4.4 Ticketing & ITSM Diversity

- ServiceNow for the enterprise client? Freshdesk for the SMB client? Zendesk for the e-commerce client?
- Who currently maintains the integrations between dialer → CRM → ticketing per client?

### 4.5 Pain Probes

- Have you had a toll-fraud incident in the last 24 months? How much was the realized loss?
- What's your annual external audit + QSA spend across all clients?
- What percentage of your agents are working from home? Are they on consumer-grade home networks?
- Have any of your clients started asking about post-quantum cryptography in their RFPs or DPAs yet? *(This is the question that often gets you in the door with the CISO.)*
- When was the last time IT discovered a protocol or integration in production they didn't know was there?

### 4.6 The Closing Discovery Move

After they've answered the portfolio shape questions, say:

> *"So if I'm hearing you right, you have NorthBank on Avaya + Salesforce + ServiceNow with PCI and SOX scope, CareFirst on Genesys + Zoho + Freshdesk with HIPAA scope, and ShopRight on Asterisk + HubSpot + Zendesk with PCI and GDPR scope. Three different stacks. Three different compliance regimes. One shared agent floor and a remote pool. Is that fair?"*

If they say yes, you've earned the right to walk them through the Acme Outsourcing demo in Section 5 — because *they just told you the demo is their environment*.

---

## 5. The Acme Outsourcing Demo Walk

This is the demo script. It's the same composite scenario the product owner walkthrough uses, but here it's narrated for a sales call. Acme Outsourcing has 3,000 seats plus 600 WFH agents, serving three clients:

| Client | Stack | Compliance |
|---|---|---|
| **NorthBank** (banking) | Avaya Aura CM + Cisco UCCE + Salesforce + ServiceNow | PCI-DSS, SOX, FCA |
| **CareFirst Health** (healthcare payer) | Genesys Cloud + Zoho + Freshdesk | HIPAA, HITECH |
| **ShopRight Retail** (e-commerce) | Asterisk/FreePBX + HubSpot + Zendesk | PCI-DSS, GDPR, TCPA |

### Demo Beat 1 — Discovery in Real Time *(2 min)*

> *"This is what happened in Acme's environment 4 hours after we put a tap on the network. The AI engine found everything across all three client stacks — without anyone configuring it."*

Show the protocol graph: SIP+RTP on the Avaya/Cisco side for NorthBank, Genesys Cloud WebRTC for CareFirst, Asterisk AMI/ARI for ShopRight, plus the Salesforce API streams, Zoho REST, HubSpot REST, ServiceNow/Freshdesk/Zendesk traffic, and the TN3270e mainframe sessions NorthBank uses.

**What to say:** *"Notice the legend on the left — every protocol is color-coded by tenant. Each client is fully isolated even at the discovery layer."*

### Demo Beat 2 — Three Tenants, One Console *(2 min)*

Switch to the multi-tenant policy view. Show three tenant cards side-by-side:

- **NorthBank** — PCI-DSS 4.0 policy bundle, SOX policy bundle, FCA recording policy. Separate ML-KEM encryption keys.
- **CareFirst** — HIPAA policy bundle, HITECH policy bundle. Separate ML-KEM encryption keys.
- **ShopRight** — PCI-DSS policy bundle, GDPR policy bundle, TCPA outbound policy. Separate ML-KEM encryption keys.

**What to say:** *"One platform, one console, three completely separate cryptographic and compliance worlds. The platform structurally cannot leak data across these tenants — even if an operator wanted to."*

### Demo Beat 3 — The Toll Fraud Incident *(2 min)*

Trigger (or show a replay of) an IRSF pattern on ShopRight's Asterisk trunk: three calls in a row to high-risk premium-rate country codes.

Watch QBITEL block the trunk in <1 second, generate an LLM-written NOC narrative (*"Outbound trunk T-12 quarantined: three sequential calls to premium-rate destinations in country code +XYZ within 47 seconds. Pattern matches IRSF profile. Estimated exposure if continued: $42,000 over 48 hours. Trunk isolated; forensic capture preserved; ShopRight notification sent."*), and log the event.

**What to say:** *"That's the difference between finding out Monday morning at the carrier invoice and finding out 47 seconds in. This single feature usually pays for the platform in the first 30 to 60 days."*

### Demo Beat 4 — PCI Evidence Pack for NorthBank, In Under 10 Minutes *(2 min)*

From the NorthBank tenant view, click *Generate Compliance Evidence Pack → PCI-DSS 4.0*. Watch the timer count up. The report appears with full evidence: DTMF-masked call inventory, per-agent screen-mask events, recording encryption keys' rotation history, scope-boundary attestation.

**What to say:** *"That pack contains only NorthBank's data. Now watch."*

Switch to the CareFirst tenant view, generate a HIPAA evidence pack. Different content, different scope, different audit trail — generated in parallel, in <10 minutes.

**What to say:** *"NorthBank's QSA sees only NorthBank. CareFirst's auditor sees only CareFirst. ShopRight's auditor sees only ShopRight. No cross-contamination, no manual evidence-gathering, no engineering time. This single capability typically retires an entire evidence-collection team."*

### Demo Beat 5 — The Remote Agent *(1 min)*

Show a WFH agent's endpoint dashboard: VPN-less PQC tunnel up, posture checks green (WPA3 home WiFi, disk encryption on, AV current, geo within policy), session watermark active.

**What to say:** *"This agent is handling a CareFirst call from their kitchen. There's no VPN, no client-installed CRM agent, no PHI on their disk — and HIPAA evidence is being captured automatically. When CareFirst's HIPAA audit comes, this agent's session is already in the pack."*

### Demo Beat 6 — The 4th Client Hypothetical *(1 min)*

**What to say:** *"What happens when you sign your next client and they mandate, say, Mitel + Dynamics 365 + ServiceNow with FedRAMP requirements? The marketplace already has Mitel and Dynamics adapters. The AI discovery handles anything the marketplace doesn't. Onboarding a new client onto QBITEL is a policy exercise — not a re-architecture. That means you can quote security-sensitive RFPs faster, and you can win them."*

---

## 6. Tailoring the Pitch by BPO Size

QBITEL Bridge is priced per concurrent agent seat in three tiers (per the [QBITEL_BPO_PITCH_QA_GUIDE.md](QBITEL_BPO_PITCH_QA_GUIDE.md)). Match the pitch to the tier:

### Contact Center Tier — Up to 500 seats
- **Hook:** *"You're a small enough operation that one toll-fraud incident or one failed PCI audit could be existential. QBITEL is the insurance that pays for itself."*
- **Focus on:** Toll fraud prevention, PCI scope reduction, fast deployment.
- **Avoid:** Heavy multi-tenant emphasis (they may have one or two clients, not ten).

### Enterprise BPO Tier — 500–5,000 seats *(the sweet spot)*
- **Hook:** *"You're running the portfolio problem at scale — multiple clients, multiple stacks, multiple audits. QBITEL is the only platform that handles the portfolio rather than the line item."*
- **Focus on:** Multi-tenant isolation, per-client compliance automation, time-to-onboard new clients, total annual audit savings.
- **This is where most of your sales effort lives.**

### Global BPO Tier — 5,000+ seats, unlimited tenants
- **Hook:** *"You're a competitive battleground. RFP win rates increasingly come down to who can answer the security questionnaire fastest and most credibly. QBITEL turns 'we'll need 6 weeks to scope that' into 'we already have a compliance pack ready for that.'"*
- **Focus on:** Strategic win-rate impact, M&A integration speed (new acquisition's stack discoverable in hours), executive dashboards, dedicated CSM and SLAs.
- **Bring:** Customer references at similar scale, breach cost avoidance framing ($4.8M IBM 2024 average), board-level briefing artifacts.

**A note on pricing on the call:** Don't quote ranges. Anchor on value first — "we're priced per concurrent seat, with tiers that match the scale you just described, and the toll-fraud line item alone usually covers it in 30 to 60 days. Let me get you a tailored quote based on the seat count and tenant count you walked me through." Then route to `enterprise@qbitel.com` for the formal proposal.

---

## 7. The ROI Conversation

The four ROI sources (numbers consistent with the [Q&A guide](QBITEL_BPO_PITCH_QA_GUIDE.md)):

### 7.1 Toll Fraud Prevention — 30–60 Day Payback

- Average BPO loses **$200K–$2M annually** to SIP toll fraud.
- A single prevented weekend attack (typical $30K–$80K) often covers the annual license cost.
- QBITEL detects patterns within 3 calls, blocks in <1 second.

**Sales line:** *"You're already paying for QBITEL today — you're just paying it to the fraudsters. We redirect that line item to a platform that gives you eight other capabilities on top."*

### 7.2 PCI-DSS Audit Scope Reduction — $500K–$2M Annual Savings

- DTMF masking + recording controls reduce PCI audit scope by up to **80%**.
- Annual audit and compliance cost reduction: **$500K–$2M+** depending on seat count and QSA.
- Multiply that by the number of PCI-scoped clients in your portfolio.

**Sales line for multi-client BPOs:** *"You're not saving $500K — you're saving $500K per client with PCI scope. NorthBank, ShopRight, that's already $1M. Add a fourth client and you've doubled it."*

### 7.3 SOC Team Efficiency — 78% Autonomous

- 78% of routine security events handled with no human intervention.
- For a 5-person SOC, that recovers **2–3 analyst hours per day** — redeployed to actual incidents and new-client onboarding.

**Sales line:** *"Your SOC analysts are expensive and hard to hire. QBITEL doesn't replace them — it stops them from drowning in alert fatigue so they can do the work you actually hired them for."*

### 7.4 Breach Cost Avoidance — $4.8M (IBM 2024)

- Average BPO data breach cost: **$4.8M** (IBM 2024 Cost of a Data Breach Report).
- BPO breaches typically come from protocol-layer attacks and insider exfiltration — both addressed directly by QBITEL.
- For BPOs, a breach also costs the relationship with the affected client — often worth multiples of the immediate breach cost.

**Sales line:** *"The math on the breach you don't have isn't on the proposal. But you and I both know one CareFirst-class incident ends the CareFirst contract. QBITEL is asymmetric — you spend a known number to remove an unknown but catastrophic one."*

### 7.5 The Per-Client Multiplier Frame

For every ROI number above, apply this lens:

> *"Take that ROI number. Now multiply by the number of clients you operate. That's the actual case for QBITEL in a BPO."*

This is the line that converts the conversation from "interesting product" to "this is how we run our business."

---

## 8. Objection Handling — Top 10 Pushbacks

Each one is a short script. *Hear the objection → say the response → pivot to the next concrete step.*

### O1. *"We can't deploy anything that touches our clients' CRMs without their approval."*
**Response:** *"You won't have to. QBITEL is a network-layer overlay — it observes and protects, but it doesn't touch the CRM itself. Most CRM vendors have already approved the integration pattern; we can walk through it for Salesforce, Zoho, HubSpot, or any of the ones in your portfolio in 15 minutes. And for any client that asks, the per-tenant isolation and evidence pack is usually what they want to see — it makes you look better to them."*
**Pivot:** *"Which client would you want to walk through first?"*

### O2. *"Our clients dictate our security stack. We can't add anything they haven't approved."*
**Response:** *"That's exactly the right concern. The interesting thing is that QBITEL doesn't *replace* anything in their stack — it sits underneath and produces better evidence for them. Most of our BPO customers find their clients end up *requesting* QBITEL once they see the per-tenant compliance reports. We can give you an objection-handling brief for your client conversations."*
**Pivot:** *"Want us to send you the client-facing one-pager so you can float it past one client first?"*

### O3. *"Each client demands their own SOC tools and SIEM."*
**Response:** *"QBITEL plays nicely with that — we feed per-tenant events into whatever SIEM each client mandates (Splunk, QRadar, Sentinel, Chronicle, etc.). The platform sits below their tools, not in place of them. You're augmenting their stack, not competing with it."*

### O4. *"We just finished a major PBX investment. We're not ripping it out."*
**Response:** *"Good — you shouldn't have to. QBITEL deploys *on top of* your existing PBX without replacing a single component. The fact that you've invested in a modern PBX makes the QBITEL deployment faster, not slower. We'd be adding the quantum-safe layer and the compliance automation on top of your investment, not competing with it."*

### O5. *"Our voice latency is already tight. We can't add anything that affects call quality."*
**Response:** *"PQC overhead on the voice path is under 2 milliseconds — within the ITU-T G.114 budget, which means it's structurally inaudible. We measure MOS scores before and after deployment as part of every POC. If you can hear a difference, we don't go live."*

### O6. *"We don't have budget this year — security is committed."*
**Response:** *"QBITEL isn't an additional cost — it's cost displacement. Toll fraud is hitting your carrier line item every month already; QBITEL redirects that spend. PCI audit prep is consuming engineering time already; QBITEL automates that. The 2-hour Discovery Assessment is free and tells you exactly what you're already losing. That conversation usually unlocks the budget conversation."*

### O7. *"Quantum is 10 years away. We don't need quantum-safe today."*
**Response:** *"Two things. First, the harvest-now-decrypt-later threat is happening today — your recordings from this week sit in retention for 7 years under SOX/HIPAA, so they're already exposed. Second, QBITEL is not just PQC. The toll-fraud, DTMF masking, agent DLP, and per-client compliance automation pay back the platform regardless of quantum. The PQC is the bonus."*

### O8. *"This sounds too good to be true — surely there's a catch."*
**Response:** *"Fair question. The catch is it's a network-overlay model, so you do need SPAN-port access on your core switches and a few hours of change-window. Beyond that, the platform is designed not to replace anything, so the 'rip-out cost' you'd normally have is zero. We'd rather you do the 2-hour Discovery Assessment and see for yourself than take our word for it."*

### O9. *"How do I know this isn't going to be 18 months of professional services?"*
**Response:** *"Deployment is 4–6 hours, end-to-end, with zero downtime. Per-client onboarding after deployment is 2–5 days because the major dialers/CRMs/ticketing systems are pre-built in the marketplace. We can show you the deployment checklist line-by-line — it's not a slide, it's an operational document."*

### O10. *"We tried something like this — it didn't work for our environment."*
**Response:** *"Genuinely curious — what didn't work? Most 'similar' tools are either inline appliances (latency + single point of failure) or cloud-only (off-shore data residency issues) or single-protocol (don't actually cover the mix). QBITEL is none of those — it's a passive-tap network overlay with on-prem AI option that handles every dialer, CRM, and ticketing system. The Discovery Assessment will tell us in 2 hours whether your specific environment is one we can serve."*

---

## 9. Competitive Positioning Cheatsheet

When the prospect mentions a competitor, here's what to say:

| If they mention... | Say this |
|---|---|
| **CrowdStrike / SentinelOne / EDR** | *"Great for endpoint malware — not built for protocol-layer voice/SIP/RTP/CTI security or DTMF masking. They live on the endpoint; QBITEL lives on the wire. Customers usually run both."* |
| **Palo Alto / Fortinet / firewalls** | *"Excellent perimeter and SASE. They don't discover undocumented BPO protocols, don't do DTMF masking, don't do toll-fraud pattern matching, don't do per-tenant compliance automation. QBITEL handles the inside-the-perimeter contact-center-specific layer."* |
| **Claroty / Dragos / OT security** | *"OT-only — they don't speak SIP, CTI, or contact-center protocols. Different vertical."* |
| **Generic SIEM (Splunk / Sentinel / QRadar)** | *"We feed them. We're the sensor and the enforcement layer; they're the analytics layer. Most of our deployments fan events out into the customer's SIEM of choice — including the client's mandated SIEM per tenant."* |
| **NICE / Verint compliance recording tools** | *"They record. They don't prevent toll fraud, don't do DTMF masking in real time, don't generate per-tenant PQC-encrypted evidence packs. We complement them — QBITEL encrypts the recordings they produce."* |
| **A managed SOC service** | *"A managed SOC can watch your dashboards — but they can't reach into your PBX, your CRMs, and your ticketing systems to enforce policy at wire speed. QBITEL is the platform that gives the SOC something useful to watch."* |

---

## 10. The Close & Next-Step Menu

End every meaningful conversation with one of three concrete next steps. Pick the one matched to where they are in the cycle.

### Commitment Level A — Free Discovery Assessment *(2 hours, no commitment)*

**What you ask for:** SPAN-port access on one floor for 2 hours and an hour of an architect's time to interpret the output.
**What they get:** A real protocol map of their environment showing every dialer/CRM/ticketing/legacy protocol in production, plus a current-state risk and compliance-scope snapshot.
**Use when:** First serious meeting; they need internal evidence to justify deeper engagement.
**Follow-up email subject line:** *"2-hour Discovery Assessment — what we'd find at [Acme]"*

### Commitment Level B — Single-Client Pilot *(30 days)*

**What you ask for:** Approval to deploy against one client's stack on one floor for 30 days. Typically pick the most painful client (most PCI scope or most toll-fraud exposure).
**What they get:** Full QBITEL deployment against production traffic for one tenant; toll-fraud monitoring live; one compliance evidence pack generated and walked through with their QSA/auditor; ROI measured against a baseline.
**Use when:** Champion exists, but procurement is uneasy committing to multi-client rollout.
**Follow-up artifact:** Pilot scoping doc + deployment checklist (the existing [BPO deployment checklist](QBITEL_BPO_DEPLOYMENT_CHECKLIST.md) maps cleanly).

### Commitment Level C — Full Multi-Client Rollout *(production engagement)*

**What you ask for:** Annual license commitment sized to seat + tenant count; access to all client environments per a phased rollout plan; named Customer Success Manager engagement.
**What they get:** Full production deployment in 4–6 hours; per-client onboarding rolled in over 2–4 weeks; ongoing CSM support, SLA guarantees, and quarterly business reviews.
**Use when:** They've already seen a pilot or are operating at scale and need the strategic answer this quarter.
**Follow-up artifact:** Tailored proposal from `enterprise@qbitel.com`, deployment timeline, and reference customer call.

### The Email Template After Every Meeting

> *"Thanks for the conversation today — what stood out to me was [reflect their #1 pain back]. To make this real, the most useful next step is [commitment level A/B/C]. Specifically, [one concrete deliverable] in [one concrete timeframe], with [one concrete success criterion you both agreed to]. Are we good to put [date/time] on the calendar?"*

Three sentences. One commitment. One date. Always.

---

## Appendix A — Related Sales Resources

| Resource | When to use |
|---|---|
| [QBITEL_BRIDGE_BPO_MARKETING_PITCH.md](QBITEL_BRIDGE_BPO_MARKETING_PITCH.md) | Full pitch narrative; source of opening hook |
| [QBITEL_BPO_PITCH_QA_GUIDE.md](QBITEL_BPO_PITCH_QA_GUIDE.md) | Deep Q&A by buyer persona — RFP responses, hard objections |
| [QBITEL_BPO_DEPLOYMENT_CHECKLIST.md](QBITEL_BPO_DEPLOYMENT_CHECKLIST.md) | Operational checklist — share with technical champions |
| [10_BPO_CALL_CENTERS.md](10_BPO_CALL_CENTERS.md) | One-page vertical brochure — pre-meeting leave-behind |
| [QBITEL_BPO_PRODUCT_OWNER_WALKTHROUGH.md](QBITEL_BPO_PRODUCT_OWNER_WALKTHROUGH.md) | Hand to their product owner / solution architect after the first qualified meeting |
| Audio walkthrough: *Quantum-Safe Security for Legacy Call Centers* | Pre-call education for prospects who prefer listening |
| Infographic: `qbitel-bridge-BPO-infographics.png` | One-slide visual for cold outreach |

---

## Appendix B — The Multi-Stack Quick Reference

Carry this in your head. When a prospect names any of these systems, you can immediately confirm QBITEL covers it:

| Category | Systems QBITEL Integrates With |
|---|---|
| **Dialers / Telephony** | Avaya Aura CM, Cisco UCCE / CUCM, Genesys Cloud, Asterisk / FreePBX, Mitel, BroadWorks, RingCentral, plus AI discovery for anything custom |
| **CRMs** | Salesforce, Zoho, HubSpot, Zendesk, Dynamics 365, ServiceNow CSM, Freshdesk (when used as CRM), plus REST/SOAP for custom CRMs |
| **Ticketing / ITSM** | ServiceNow, Freshdesk, Zendesk Support, Jira Service Management, plus REST adapters |
| **WFM / Quality** | NICE WFM, Verint, Aspect, Calabrio, Genesys WFM |
| **Recording** | NICE Engage, NICE CXone, Verint WEM, Avaya WFO |
| **Mainframe / Legacy** | TN3270e, TN5250, custom CICS / IMS protocols via Translation Studio |
| **IVR** | Nuance, Cisco CVP, Avaya IR, Genesys Dialog Engine, Amazon Connect |
| **Compliance Frameworks** | PCI-DSS 4.0, TCPA, HIPAA, HITECH, SOC 2 Type II, GDPR, SOX, GLBA, FCA/MiFID II, NIST PQC |

When in doubt: *"It's a network-overlay platform with AI discovery and a marketplace of 1,000+ pre-built protocols. If it's on the wire, we cover it."*

---

*QBITEL Bridge — Sales Manager Walkthrough for BPO & Call Center Sales Cycles.*
*Confidential — For Authorized Recipients Only — © 2026 QBITEL.*
*enterprise@qbitel.com | bridge.qbitel.com*
