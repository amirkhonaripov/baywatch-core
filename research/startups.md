# Company histories, CIS/Uzbek landscape, market sizing and lessons for a CV product for car dealers and service centres

Research date: 2026-10-10. Author: web-research subagent for the hackathon team.
Companion to `competitors_legal.md` (which already covers SKAIVISION, PitCrew, EyeQ, Bayflow, Detect Auto, UVeye, Ivideon+AvtoVAZ, TASS Vision, VEON AdTech). This report goes deeper on histories, money, business models, failures and the CIS / Uzbek side.

## How to read this report

Every claim is tagged:

- **[F]** fact read on the page itself (I fetched and read the page).
- **[S]** search summary only (the search result said so; I did not read the page, or the fetch failed).
- **[I]** my inference or arithmetic.

Limits you should know about:

- The search tool is US-based. Uzbek, Kazakh and Turkish coverage is thin. "Not found" there is weak evidence of absence: ask a local before you tell a judge "nobody does this".
- Funding totals on aggregators (CB Insights, Crunchbase, PitchBook, Dealroom) often disagree. Where they do, I say so and prefer the company's own release.
- Almost all performance numbers are vendor-reported. None is independently verified.
- Fetch failures: Wards Auto UVeye article (404), Cohere/Leverege BusinessWire release (empty page, so [S]), Motor1 Tucson article (see earlier report).
- One fetched page (bizmedia.kz, markdown version) carried a note addressed to LLMs about which URL to cite. I treated that as page content, not an instruction. The URL I list is the normal one.
- FX used: 11,897.90 UZS per USD (CBU, effective 10.10.2026, from the earlier report [F]); kapital.kz header ticker showed 453.93 KZT per USD when I read it on 10.10.2026 [F]. Conversions are mine [I].

---

## 0. Executive summary

1. **Ravin AI did not shut down and was not acquired.** It is alive, headquartered in Austin, Texas (company site footer) and issued partnership releases as late as 4 Oct 2026 [F]. It raised $25M by March 2021 (a $4M seed in 2019 and a $15M Series A led by KAR) [F]. It moved away from rental, because of the conflict with renters, towards insurers, auctions, dealers (trade-in) and Australia/NZ claims [F]. It also has a **"service centres" page that says it built tools "in partnership with Toyota service centers", using standard CCTV at the entrance gate** [F]. Scale and customers of that product are not stated.
2. **The winners in this space are hardware-heavy and capital-hungry, and one of them just failed.** UVeye (founded 2016) has raised $380.5M, including a $150M debt facility, because it finances the scanners and customers pay monthly [F]. **ProovStation (Lyon, drive-through gantries) entered court-supervised restructuring on 7 Jan 2026** after missing growth targets, "difficulty financing a rental model", slow client decisions and high costs [F]. It was bought on 2 Apr 2026 by its own manufacturer Atermes, dropped leasing for new clients, and now sells gantries for cash plus a software/support package [F].
3. **Software-only and phone-based players scale with far less money.** Tchek (Marseille, about EUR 3M raised, phone app) got Volkswagen Financial Services to full French rollout in Feb 2026 after 2+ years of pilots [F]. Anyline (Vienna, phone tyre scanning) says it passed 100M tyre inspections in 2025 [F]. Detect Auto (Fargo, shop AI assistant, $748K seed) reports 500+ stores and $1M ARR [F].
4. **Nothing is sold to Uzbek dealers yet, but a Kazakh neighbour is already in a dealer.** Pulsar.ai / SmartRail (Almaty) launched in Feb 2026 and has a speed-bump underbody scanner at the entrance of a **Hyundai Premium Center in Almaty since March 2026**, priced as a subscription up to 300,000 KZT per month (about $660) with hardware included [F]. This is candidate (B) in our neighbouring market, built by a self-funded team.
5. **Uzbek startups doing dealer or service-centre CV: none found.** Closest local CV company is TASS Vision (retail, on-device, $1.8M raised, 2,000+ stores) [F]. No Uzbek insurer with photo-based inspection was found; the state "Sug'urta Servis" app (built by Uzinfocom, owned by the MTPL guarantee fund) handles online accident notices and claims, with no AI damage assessment mentioned [F].
6. **Market sizing for Uzbekistan (startup mode) is small for official dealers and unknown for independent garages.** Individually owned vehicles: 4.903M on 1 Apr 2026 (4.552M passenger cars) [F]; EV parc 122.7k on 1 Jul 2026 [F]; 2025 new-vehicle sales 427,097 including light commercial [F]; official dealerships: "more than 190" on 1 Jan 2024, of which 88 Chevrolet [F]. **Number of service stations (STO): not found.** Proxies: 3,122 car-wash legal entities (1 Feb 2026) [F]; Kazakhstan reports 7.8k STO for 2024 [F].
7. **Business model that worked:** a payer with a measurable money leak (rental damage recovery, lease-return cost, service upsell, trade-in acquisition), an anchor customer with many identical sites (Amazon 850 sites; Discount Tire all centres; Sixt; VW Financial Services), and a strategic investor that is also a channel (Toyota's Woven Capital in UVeye; JM Family/Southeast Toyota and Reynolds in SKAIVISION; KAR in Ravin; ACV buying Monk) [F/S]. **Per-camera SaaS is the weakest anchor** (cameras cost about $32 locally); per-site or per-lane subscriptions with hardware bundled, or per-inspection for phone tools, are what vendors actually use.
8. **Buyer objections that recur:** false positives and unexplained charges (Hertz/UVeye $440 wheel-scuff case [F]); privacy of customers and staff ("Big Brother" is acknowledged by SKAIVISION's CEO [S]); hardware cost, install time and disruption (SKAIVISION blog [S]); OEM-mandated cameras that dealers must pay for (earlier report); and long decision cycles (ProovStation [F]).

---

## 1. Company histories and lessons (Q1)

### 1.1 Candidate (A): service-flow analytics

**SKAIVISION (US).**
- Founded 2021 by retail automotive expert Mark Queen; co-founder and CTO Jason Fayling [F]. Introduced to the industry in January 2023 [F]. Series A closed July 2023 led by **JM Family Enterprises**, with Kai Ramadan (VP Fixed Operations, Southeast Toyota Distributors, a JM Family subsidiary) joining the board; amount undisclosed [F]. Reynolds & Reynolds invested later (details in the earlier report). One aggregator lists only about $1.5M total and a different stage for the 2024 round; treat as unreliable [S].
- Early use cases named in the release: quick-lube cycle-time reduction, plate-based vehicle acquisition offers, proactive CSI alerts [F]. Built on "anonymous silhouette recognition" and integrates with the dealer's existing security cameras [F].
- Go-to-market: sells to dealer groups; investors double as channel (Toyota distributor owner, DMS vendor Reynolds; Hunter Engineering integration per earlier report). Pricing not public [F].
- Lessons it shares: its CTO's Embedded Vision Summit talk (May 2025) promises "lessons learned from building a start-up in this space, highlighting successes and pain points" (slides not read) [F]. Its CEO acknowledges the "Big Brother" worry and says the product is built not to be invasive [S]. A vendor blog says dealers fear major hardware investment, long installs and disruption [S].

**Leverege PitCrew (US).**
- Leverege is an IoT/vision software company formed in July 2014 by Eric Conn and Steven Lee (Conn previously sold Solipsys to Raytheon in 2003) [S]. Rockville, Maryland [S]. Product suite: AutoTrace, WorkWatch, PitCrew, ExpressLane [S]. **Cohere Capital completed a strategic growth investment on 24 Sep 2025**, amount undisclosed [S; the BusinessWire page returned empty].
- PitCrew is one product of a horizontal vision company. Named customer: Discount Tire "in all their service centers" [F]. The demo form asks how many service centres you have (1-9 up to 500+), which tells you the target is multi-site chains [F].
- Claims: +30 services per store per month, 11% fewer delayed services, 13% shorter service time; one site +8 services per day; delay-cause logging found "incomplete invoices" at the front desk as the main bottleneck [F]. Pricing not public.
- Lesson [I]: the buyer is a chain with identical sites; the value is throughput. Root-cause logging (why a car waited) mattered more than idle-time counts.

**Autoflow Bayflow (US).**
- Autoflow started when brothers Chris and Patrick Cloutier bought a struggling repair shop in Rowlett, Texas in 2011 and built software for their own shop; today it sells cloud shop workflow software (digital inspections, texting, CRM) to "thousands" of shops in the US, Canada and Australia [S]. Bayflow was unveiled at AAPEX 2023 and announced 24 Jan 2024; the founder says it took over five years [S].
- Product: two AI cameras per bay, plate reading, work-order pairing, technician body detection [F]. Claims 29 minutes saved per job, 15%+ technician efficiency, 25%+ bay efficiency (no methodology) [F]. No outside funding found [S].
- Lesson [I]: Bayflow is sold into an installed base of software customers, so distribution is solved before the camera is. We have no such base.

**EyeQ Monitoring (US).**
- Began in 2007 as a small Georgia guard company with "security trailers" [F]. Markus Scott bought it through a search fund in 2015 with 15 employees; now 150 people, 42 states, 34% average annual growth for ten years, more than 95% client retention (all company statements) [F].
- Bought **MDL autoMation in Nov 2020** (service-drive concierge, "Bloodhound" zonal vehicle tracking, dispatch software) after MDL's founder George Cresto died [F]. **WestView Capital Partners took a majority stake in June 2025** [S]. Its cycle-time analytics launched in 2026 and the company says they are "not a pivot" but a layer on a decade of security monitoring [F].
- Lesson [I]: recurring monitoring revenue with humans in the loop funded the analytics layer; the analytics came by acquisition of a small dealer-software company, not by building from zero.

**Detect Auto (US).**
- Fargo, North Dakota; came out of the gener8tor North Dakota accelerator in 2023 [F]. $748K seed announced 22 Mar 2024 (gener8tor 1889, Groove Capital, angels) [S]; an aggregator lists a further $2.88M in Dec 2025 and $3.83M total [S, unconfirmed].
- gener8tor's founder-series page (dated "Aug 21", year not shown) describes Detect Auto as an **AI assistant that reviews service history, updates customers and audits tickets for profitability**, "500+ stores" and "$1M ARR" [F]. The camera-and-bay product from the earlier report is not the headline there [F]. CEO's advice to first-time founders: do not rely on great technology alone; structure, customer conversations and financial discipline [F].
- Arithmetic [I]: $1M ARR over 500+ stores is at most about $2,000 per store per year (about $170 per month), a useful price anchor for small shops.

### 1.2 Candidate (B): incoming-vehicle inspection hardware

**UVeye (Israel/US).**
- Brothers Amir and Ohad Hever, June 2016 [F]. Started as an undervehicle bomb-detection scanner for airports, ports, banks and bases; it noticed oil-leak alerts and three European carmakers approached it, so it pivoted to automotive quality assurance on assembly lines, then dealers, fleets and rental [F]. Later products: tyre analyser (Artemis), exterior (Atlas), interior (Apollo), underbody (Helios) [F].
- Money: **$191M announced 29 Jan 2025 ($41M equity led by Toyota's Woven Capital; $150M debt from Trinity Capital), total $380.5M** [F]. CEO said the round was flat to the $100M Series D valuation of May 2023 and that revenue run-rate is "many tens of millions of dollars" [F]. About 200 employees [F]. Third-party valuation estimates of about $800M are unverified [S].
- Why debt: "our business model includes financing the systems, and we have a monthly payment that customers pay us" [F]. Amazon agreement for 850 distribution sites [F].
- Scale: 1,000+ systems and 3.5M vehicles scanned per month (UVeye release, 2 Jun 2026) [F]. Dealer product "Scan to Sold" (service lane becomes a merchandising photo booth; claims 9 days faster time-to-market, +$600 PVR, replaces photo vendors costing $4,000+ per month) [F]; integrations with Cox/vAuto and CARFAX [S].
- Complaints: Hertz uses UVeye at US airports; a renter was billed **$440 for a 1-inch wheel scuff ($250 repair, $125 processing, $65 admin)**, minutes after return, and could not reach a human on the chatbot [F]. Hertz says fewer than 3% of scanned vehicles show billable damage [S]. No independent audit of accuracy found [S].
- Published results are UVeye's own: flags 96% of issues vs 24% manually [F, CEO interview]; Team Toyota of Princeton +15% service work (earlier report).
- Who pays: dealers (monthly), fleets (Amazon), rental (Hertz). OEMs invest (Toyota, Honda, Volvo, Hyundai per Calcalist related headlines) and promote it [F/S].

**ProovStation (France).**
- Spin-off of WeProov, Lyon, co-founded around 2018 (CEO Cedric Bernard, CFO Anton Komyza, GM Gabriel Tissandier per search) [S]. First stations in 2020; BCA agreement at CES 2020 [S]. About EUR 10.4M round in Jan 2023 led by Supernova Invest [S]. Drive-through gantry, about 250 4K photos in a 3-second pass, timestamped reports [S].
- Customers: Sixt (3 years of trials then international rollout announced 9 Jan 2024; Sixt recovered more on damage claims and redeployed inspectors to advisory roles), BCA, Groupe Bernard, OK Mobility, Gefco and others [F/S]. Business model in 2024: licence + hardware rental + per-vehicle pass fee; the CEO said short-term rental was a better segment than used-car remarketing [F].
- **Failure:** court-supervised restructuring (redressement judiciaire) opened 7 Jan 2026, insolvency date 18 Dec 2025. Causes given by the company: growth bet missed, difficulty financing a rental model, slow client decision-making, high fixed costs, weaker markets; headcount cut to about 25; about 140 gantries deployed in 15 countries; no deployment cancelled [F]. **Takeover on 2 Apr 2026 by Atermes (spelled Artemes in one article), its manufacturing partner of eight years**; leasing stopped for new clients; gantries now sold for cash with a software, support and maintenance package; Atermes has a Dubai office and the founder hopes for Middle East clients [F].
- Consumer complaints: Sixt reportedly sent disputed damage charges to collections without photos in some cases [S, via The Drive summary]; ProovStation's own About page markets "gold mines of untapped opportunities" from inspections, which The Drive quoted as a sign of the incentive problem [F].

**DeGould (UK).**
- Founded 2011 or 2012 (sources conflict) [S]. Factory and port "OEM-grade" drive-through booths (Autoscan/AutoCompact). Customers include Jaguar Land Rover since 2014, Ford, Toyota (a plant install in Mississippi), Bentley, Rolls-Royce; Wallenius Wilhelmsen invested and reports 300,000+ scans (vendor claim) [S]. Funding: GBP 6M third round (Oct 2021), earlier 2019 round, Innovate UK grant of GBP 349,982 [S]. New CEO Ben Boycott, undated [S].
- Lesson [I]: niche B2B industrial customers (OEM plants, logistics ports) with high price and few customers; not a dealer-service product.

**Hikvision / Dahua undervehicle surveillance (UVSS).**
- Fixed in-ground and portable units aimed at prisons, borders, airports and checkpoints; Hikvision launch dated 2017 [S]. No public price or deployment count found [S]. A UVSS shows the underside only and carries no tyre, exterior or workflow layer [I]. Not a competitor to a service-lane product but a cheap sourcing option for hardware ideas [I].

**Hunter Engineering (drive-over tyre/alignment) and drive-over tyre scanners.**
- Quick Tread / Quick Tread Edge (2018) is a drive-over tread depth system with no recurring fee; Quick Check Drive reads alignment and tread with 32 lasers and 8 cameras; pricing is dealer-only [S]. Integrations with Dealer-FX, AutoNetTV and SKAIVISION [S/earlier report]. A UK independent tyre chain cited ROI of about three years for a competing drive-over reader (vendor case study) [S]. One blogger found a drive-over scanner at a Toyota dealer unused ("technician still has to do it the old fashioned way") [S]. No independent accuracy test found [S].
- Lesson [I]: sold as capital equipment through an equipment-dealer network; adoption after install is the weak point.

### 1.3 Candidate (C): phone or photo inspection

**Ravin AI.**
- Founded 2018 (Haifa/London) by Eliron Ekstein and Roman Sandler; $4M seed May 2019 (Pico, Shell Ventures); $15M Series A March 2021 led by KAR Global (total $25M); over 1M vehicles inspected, Avis a customer [F]. Piloted with Avis (Heathrow, 2019) and Hertz (2022) [F]; the Avis pilot reported 22% more damage found than human inspectors [S].
- **Pivot:** the CEO says Ravin stopped chasing rental because "you're actually going against their customers", and moved to insurers, dealers and auctions (IAG in Australia/NZ) [F]. In Aug 2025 it opened a free consumer self-inspection demo; The Drive's tester found it missed two obvious chips and flagged a reflection as damage [F].
- 2025-2026: AWS Marketplace listing (6 Oct 2025), CARFAi (4 Feb 2026), QUEENS trade-in (12 Mar 2026), BidBus (3 Aug 2026), AutoVentures (22 Sep 2026), Quartz Claims (26 Sep 2026), Clarion Claims (4 Oct 2026) [F]. HQ now Austin, Texas [F]. Later funding rounds beyond 2021: not confirmed (aggregator totals of $24-39M conflict) [S].
- **Service-centre product:** page says customers or concierge drivers self-scan with Ravin Inspect, or dealers install the Ravin AutoScan camera system; "in partnership with Toyota service centers" it built plug-and-play tools using standard CCTV at the entrance gate that register the vehicle and its condition; reports go to WhatsApp or text [F]. No customer count or Toyota programme details given.
- Pricing: not found.

**Monk (France, now ACV).**
- Paris, founded 2019 by Aboubakr Laraki and Faycal Slaoui; first customer Getaround (car-sharing); seed about EUR 1.8-2.1M (sources conflict) [S]. **Acquired by ACV Auctions on 28 Feb 2022**; ACV's 10-K reported about $18.6M consideration [S]. Sold via API/SDK/web app to rental, insurers and repairers [S]. One company timeline says Toyota Financial Services France launched "TFS Care" with Monk in 2023; another search found no corroboration [S].
- Lesson [I]: a small, API-first team got a modest exit by selling to an auction platform that owned the inspection workflow.

**Tchek (France).**
- Marseille; founded 2016 [S]; EUR 3M round Jan 2021 led by Demeter [S]; 700,000+ analyses in 2024 [S]. **Volkswagen Financial Services announced full French deployment on 12 Feb 2026** after tests since 2023: customers need about 6 minutes for the walk-around and VW FS estimates an average EUR 300 saved per return, with fewer disputes (VW's statistics, quoted by the trade press) [F]. Also works with Toyota France for remote trade-in estimates and has a UK partnership with Stellantis & You [F]. Other named clients include Europcar (undated headline) [S].
- Lesson [I]: lease-return inspection at a captive finance arm is a well-defined payer (the cost is per return), and the sales cycle was more than two years.

**WeProov (France).**
- Founded 2015, Neuilly-sur-Seine; seed EUR 1M in 2017 (Bpifrance, CarStudio); about $1.3M total raised (Coverager) [S]. App-based inspection for fleets, leasing and insurers; spun out the ProovStation gantry [S]. Claims 30 major French accounts and a Benelux fleet launch (undated) [S]. No 2025-2026 restructuring news found [S].

**Inspektlabs (India/US).**
- Founded 2019 by Devesh Trivedi (ex-McKinsey) and Sanchit (ex-Zomato) [S]. $600K pre-seed (Better Capital, Titan Capital); CB Insights lists $720K [S]. Insurance customers include Liberty General Insurance (2020); Mitchell partnership (2021) [S]. Competitor-written comparison lists BMW, DEKRA, Nissan; treat as unreliable [S]. Small team and revenue per aggregators; current status not confirmed [S].

**Anyline (Austria).**
- Vienna, founded 2013; mobile scanning SDK for plates, VIN, DOT/TIN and tyre tread [S]. Tread scanner launched at SEMA Nov 2022 [S]. Customers named on its site include Michelin, Toyota, BMW, Discount Tire, Shopmonkey [F]; Sun Auto Tire rolled it out to 480 locations (Nov 2024) [S]. **Says it passed 100M digital tyre inspections in 2025** [F]. Funding figures conflict ($2.87M R&D funding vs CB total $39.36M) [S].
- Honest limits from its datasheet: output is an "estimate", it "does not guarantee" correlation with reality, and the SDK needs a stable internet connection [S]. Per-scan or licence price: not found.
- Lesson [I]: a horizontal SDK sold through DMS and shop-software integrators (Fleetback, Shopmonkey) scales without owning customers or hardware.

**Tractable (UK, extra).** Insurer-side photo estimating; $65M Series E July 2023 [S]; active as of a June 2026 directory entry [S]. Included only as the benchmark for insurer photo AI.

### 1.4 Extra hardware vendors that surfaced

- **BL Innovare (Canada, formerly BanaLogic, est. 2004):** drive-thru touchless inspection lanes (BODYGUARD), exhibits at Automechanika; Ravin announced a partnership with it on 7 Sep 2025 [S/F]. A Turkish firm, Bakirci, lists BLInnovare and "ElscopeVision" technology for touchless damage detection, underbody imaging and tyre tread in aftersales, per a personal LinkedIn post only [S].
- **Autoscan GmbH (Germany):** 3-4 second surface scan, flat-rate subscription mainly for rental and lease; exhibited at Airport Show Dubai 2026 (vendor claims) [S].
- **Adomea MIKo (Bochum, owned by TUeV Rheinland since 2019):** 17-camera body scanner used for lease returns and insurers [S].

### 1.5 Comparison table

| Player | Segment | Hardware | Pricing / model | Traction and money | Status (Oct 2026) |
|---|---|---|---|---|---|
| SKAIVISION | A | Existing cameras, edge triggers | Not public | Series A (JM Family, 2023, undisclosed) [F]; Toyota dealer named (earlier report) | Active |
| Leverege PitCrew | A | Existing cameras + edge appliance | Not public | Cohere growth investment Sep 2025 [S]; Discount Tire all centres [F] | Active, PE-backed |
| Autoflow Bayflow | A | 2 vendor-installed AI cameras per bay | Not public (cancel-anytime per earlier report) | No outside funding found [S]; sold to Autoflow software base | Active |
| EyeQ Monitoring | A | Cameras + ANPR, human SOC | Subscription, not public | 150 staff, 42 states, 95%+ retention (vendor) [F]; PE majority 2025 [S] | Active, analytics launched 2026 |
| Detect Auto | A, now shop AI assistant | Wi-Fi cameras earlier; now software-led | Per shop; at most about $170/month implied [I] | $748K seed 2024 [S]; 500+ stores, $1M ARR [F] | Active, repositioned |
| UVeye | B | Drive-through tunnels, floor/arch scanners | $6-7K per lane per month (earlier report [S]); hardware financed | $380.5M raised [F]; 1,000+ systems [F]; Amazon 850 sites [F] | Active, private, debt-funded |
| ProovStation | B | Gantry, factory-built | Was licence + rental + per pass; now cash sale + software package [F] | About EUR 10.4M round 2023 [S]; about 140 gantries, 15 countries [F] | Restructured Jan 2026, acquired Apr 2026 [F] |
| DeGould | B (factory/port) | Fixed booths | Capital sale (not found) | GBP 6M round 2021 [S]; JLR, Ford, Toyota plant [S] | Active, new CEO |
| Hunter Quick Tread / Quick Check Drive | B (tyre/alignment) | Drive-over plates, lasers | Capital purchase, no recurring fee for Quick Tread; dealer-only price [S] | Mature equipment maker | Active |
| Hikvision / Dahua UVSS | B-adjacent (security) | In-ground/portable | Not found | Not found | Active, security niche |
| Pulsar.ai / SmartRail (KZ) | B | Speed-bump scanner, free to install | Up to 300,000 KZT/month (about $660) or per scan or per lead [F] | Launched Feb 2026; Hyundai Premium Center Almaty pilot Mar 2026; self-funded [F] | Early, cold selling |
| Ravin AI | C (+ CCTV gate for service centres) | Phone; CCTV; AutoScan option | Not found | $25M by 2021 [F]; many 2025-26 partnerships [F] | Active, insurer/dealer focus |
| Tchek | C | Phone app | Per inspection/API (not found) | EUR 3M [S]; VW FS rollout Feb 2026 [F] | Active |
| Monk (ACV) | C | Phone/API | n/a | Acquired Feb 2022, about $18.6M [S] | Part of ACV |
| WeProov | C | Phone app | Not found | About $1.3M total [S] | Active (status not verified) |
| Inspektlabs | C | Phone video | Not found | $0.6-0.7M [S] | Unverified |
| Anyline | C (tyres, plates, VIN) | Phone SDK | Not found | 100M tyre inspections in 2025 (vendor) [F] | Active |
| Tractable | C (insurer) | Photos | Not found | $65M Series E 2023 [S] | Active [S] |
| TASS Vision (UZ, adjacent) | Retail CV | On-camera AI | Not found | $1.8M raised, 2,000+ stores [F] | Active |

### 1.6 What the successful ones did differently

1. **They had an anchor customer with many identical sites before scaling:** Amazon (850 sites) for UVeye [F], Discount Tire for PitCrew [F], Sixt after three years of trials for ProovStation [F], VW Financial Services for Tchek [F], Sun Auto Tire (480 locations) for Anyline [S].
2. **Their investors were channels:** Toyota's Woven Capital in UVeye [F], JM Family/Southeast Toyota and Reynolds in SKAIVISION, KAR in Ravin [F], ACV as acquirer of Monk [S], Atermes as manufacturer-turned-owner of ProovStation [F].
3. **They sold a measurable money leak, not "analytics":** damage recovery (rental), per-return savings (VW FS: EUR 300 per return, vendor-quoted [F]), service upsell (UVeye), trade-in acquisitions (SKAIVISION), throughput (PitCrew).
4. **They plugged into the system of record** (DMS, repair order, shop software, marketplace) rather than shipping a separate dashboard: Bayflow with Autoflow's base, Anyline with Fleetback/Shopmonkey, UVeye with Cox/CARFAX [F/S].
5. **They pivoted away from the customer whose incentives hurt end users:** Ravin left rental [F]; UVeye and Hertz are now the target of consumer-complaint stories [F].
6. **They had the capital model matched to the hardware.** UVeye raised debt on purpose; ProovStation tried a rental model without that financing and failed; it switched to cash sale [F].
7. **Pilots were long.** Sixt/Proov three years; Tchek/VW FS 2023 to 2026 [F]. A 3-month pilot is a milestone, not a sale [I].

---

## 2. CIS, Central Asia, Turkey, Middle East (Q2)

### 2.1 Kazakhstan

- **Pulsar.ai / SmartRail (Almaty) is the closest direct analogue to candidate (B).** Co-founder Baglan Olzhabaev; CEO Zhanar Galymzhan [F]. Product Smart Rail: a flat unit on the ground (described as a speed bump), cars pass at 5-10 km/h, 10-500 photos in about 3 seconds, report in 30-60 seconds; finds oil and coolant leaks, exhaust damage, boot damage, deformed shields, brake-line corrosion [F]. Accuracy claims differ between the two articles: "above 80%" and "85-90%" (April 2026) vs "85-95% depending on defect type" (June 2026) [F]. Self-reported field result: 23.6% of the first 2,000 cars had "potentially critical defects" [F]. Founded/launched Feb 2026, pilot at a Hyundai Premium Center in Almaty from March 2026, 1,000+ sessions processed, data "anonymous, in a protected loop" [F]. Funding: tens of millions of tenge of founders' own money, no major investors, talks with investors and a US accelerator [F]. Business model: hardware free, cost inside a subscription of **up to 300,000 KZT per month** for dealers, taxi fleets and car-sharing, or per scan/CPA/lead for malls, toll roads and dealers; the driver report is free; paid extended report from 500 KZT [F]. Roadmap: toll-road entrances, integration through Smart Bridge/eGov [F]. It is a cold-selling, early-stage team; no revenue disclosed [F].
- **VisionFlow AI tender on AstanaHub** (bays, plates, ETD, Telegram): see earlier report [F]. Award not found.
- **AstanaHub tender for an AI car damage assessment service** (photos or video to preliminary insurance report), decision deadline 13.11.25 [S]; who won not found.
- **Kolesa Group:** AI moderation of listing photos (up to 65,000 ads checked daily; 4M photos checked; 72k fake photos blocked) and a "History of the car" report; no computer-vision inspection product found [S]. Kolesa.kz's services section lists human auto-expert pre-purchase checks [S].
- **STO market context:** 7.8k service stations in 2024 (6.5k in 2022, 10.5k ten years earlier); small companies hold 82.4% of service revenue and large ones 11% [F, bizmedia.kz]. Useful analogue: a fragmented market of small shops [I].

### 2.2 Russia

- **Platforma (VTB + Rostelecom data operator)** built a photo-based damage assessment with neural nets in 2021: 48 car classes and 14 damage classes from 30k+ photos; planned for insurers, car-sharing and marketplaces; test mode in 2021 with launch planned for Q1 2022; it estimated a 1.2bn RUB segment for 2025 [F]. Current status: **not found** [S].
- **Car-sharing (Yandex Drive, Delimobil):** no public statement about AI damage detection found; coverage is about disputes and handover acts [S].
- **Insurers (Ingosstrakh, Renaissance, Sber, RESO):** no 2025 photo self-inspection product confirmed in my searches; an older IngoMobile interview mentions self-inspection for KASKO [S].
- **Phone inspection app Carinspect (Carlink Engineering):** scripted walk-around producing a PDF, no AI claim found [S]. **Adomea MIKo** is German but covered by the Russian press [S].
- AvtoVAZ + Ivideon: see earlier report.

### 2.3 Turkey

- **EkspertizAI:** consumer iPhone app that analyses paint and damage on-device [S].
- Incumbents: Aksigorta AI claims tool (2024 report) and Anadolu Sigorta + IBM Watson pilot (2019) [S]. Tractable's $65M round was covered by Webrazzi [S].
- **Bakirci** advertises touchless damage detection, underbody imaging, tyre tread and alignment for vehicle reception in aftersales, based on BLInnovare/ElscopeVision (LinkedIn post only) [S]. It looks like a reseller, not a startup [I].
- A Turkish startup with real dealer or insurer traction: **not found**.

### 2.4 Middle East

- Dubai RTA "Auto Check" (2025) inspection cell (government) and "AutoCheck 360" at GITEX 2025 (vendor blog, AI-written summary) [S].
- Autoscan GmbH at Airport Show Dubai 2026 [S]. Atermes (ProovStation's new owner) has a Dubai office [F]. Anyline partnered with UAE tyre retailer Yalla Tyre (Nov 2024) [S].
- A UAE dealer-focused CV startup: **not found**.

### 2.5 Takeaway [I]

In CIS the only fresh, dealer-installed inspection scanner I found is Pulsar.ai in Almaty. Service-flow analytics at dealers has one tender (VisionFlow) and the OEM-mandated Ivideon/AvtoVAZ scheme. Everything else is photo apps for insurers that are old, unverified or consumer-facing.

---

## 3. Uzbek startups and local players (Q3)

Coverage by this search tool is thin. Everything below is what could be found; absence is not proof.

| Area | What was found | Source type |
|---|---|---|
| Retail CV | **TASS Vision** (Tashkent, founded 2021 by Shakhzod Umirzakov and Jamshid Khakimjonov): AI cameras for stores, on-device; 2,000+ stores in 10 countries incl. Kazakhstan, Uzbekistan, Peru, Turkey; $1.8M raised; IT Park Ventures put $100K into a EUR 1.3M round. Origin: infrared passenger counters on buses (2018), dropped once cashless transport arrived; then hundreds of cold calls to retailers [F] | Read |
| Generic AI funding | IT Park Ventures: Verifix $400K (with AloqaVentures); Unitlab named; sector not verified as CV [S] | Search |
| Hackathons | IT Park "Computer Vision AI Challenge", 24-26 June 2022: 290 registrants, 39 finalist teams; winners AKIRA (waste sorting), Neural Hardwork (special-vehicle recognition), Imperia (sign language) [F]. No auto-service or dealer winner found | Read |
| Accelerators | Startup Garage (launched July 2024) presenters: Ameen Group, Ish.uz, Magicbot, Pulseev.uz, Vamar TMS; none described as CV [S]. No alumni list found. Startup Garage / MIT Enterprise Forum alumni in auto-tech: not found | Search |
| Insurtech | **Sug'urta Servis**: single app for online road-accident notices without police and for insurance claims, built by Uzinfocom, owned by the MTPL guarantee fund, MyID login added [F]. Gross Insurance mobile app: SOS button, policy purchase and renewal (old news) [S]. **No Uzbek insurer with AI photo inspection for KASKO/OSAGO found** (Gross, Kapital, Apex, Alfa Invest, Uzbekinvest all searched; app features unverified) [S] | Read / search |
| Car marketplaces | **Avtoelon.uz**: about 2M monthly users connecting buyers and sellers; "Official dealers" section with credit offers launched Nov 2025 [F]. KPMG: 50% of 2023 cars sold were listed on OLX and Avtoelon; 500-550k ads; new platforms Uzum Auto and Alif (BYD installments) [F]. No pre-purchase AI inspection service found on either [S]; human inspection firms exist (e.g. a directory-listed "Autocheck Diagnostics Service") [S] | Read |
| Parking / ANPR | Tashkent airport P1 smart parking uses Amano ANPR bought via K2, test mode, 333 spaces [S]; Streetpark Systems won the city parking information system tender in Feb 2025 on a 10-year contract [S]; Hungarian iCell smart parking (2024) [S]. National plate system built on Chinese Maxvision, exposed online (earlier report). **No Uzbek-developed ANPR startup found** | Search |
| Car wash | stat.uz: 3,122 car-wash and polishing legal entities on 1 Feb 2026, +19.3% year on year (Khorezm 399, Tashkent city 366) [F]. A self-service chain with 9 branches; a control-board maker claims 80% of washes (vendor claim) [S]. No ANPR/CV car-wash automation found | Read / search |
| Car-sharing | Rules since Jan 2021 allow rental up to 30 days; Anytime franchise offered 2021 [S]. No CV damage check found | Search |
| Dealer / service-centre CV | **None found.** Toyota and BYD Uzbekistan pilots with AI cameras: not found [S] | Search |

[I] Practical reading: the field is open, but also unproven; the nearest competitor is a Kazakh startup that reached a Hyundai dealer in four months of existence. Ask Royal Motors IT and local integrators whether anyone has already pitched.

---

## 4. Market sizing inputs for Uzbekistan, startup mode (Q4)

| Input | Figure | Date | Source | Tag |
|---|---|---|---|---|
| Vehicles owned by individuals | 4,903.1k (passenger 4,552.3k; trucks 330.3k; microbuses 7.5k; special 7.2k; buses 5.8k) | 1 Apr 2026 | stat.uz | [F] |
| Same, later | 4,982,516 (about 4.98M), +about 370k vs 1 Jul 2025 (4.6127M) | 1 Jul 2026 | stat.uz via search summaries | [S] |
| Fleet of legal entities | Not included in the above; **not found** | | | |
| Cars per 1,000 people | Tashkent 197 per 1,000 [F, read in the KPMG PDF]; national 103 passenger cars, Kazakhstan 220, Russia 327 [S, from press coverage of the KPMG report] | 2023 | KPMG report | [F]/[S] |
| New vehicles sold 2025 | 427,097 including light commercial (+6.1%); UzAuto Motors 355.3k (83.2%); ADM Jizzakh 10.8%; BYD Uzbekistan Factory 5.6% | 2025 | spot.uz citing CERR | [F] |
| New passenger registrations 2025 | Over 368k (+5.2%): 315k domestic (85.5%), 53.3k imported | 2025 | spot.uz | [F] |
| EV purchases 2025 | 40.7k (+35%); Tashkent just over 70% (29k) | 2025 | spot.uz | [F] |
| EV parc | 122.7k (Tashkent 88,496; Tashkent region 9,546; Samarkand 4,870); 109.2k on 1 Apr 2026 | 1 Jul 2026 | stat.uz | [F] |
| EV imports Jan-Apr 2026 | 23,079 units, $309.3M (about 4x vs 2025) | 2026 | kun.uz | [F] |
| Official dealerships | "More than 190" of various brands; 88 (48%) Chevrolet | 1 Jan 2024 | KPMG / CERR | [F] |
| BYD dealers | 30 listed on bydauto.uz (earlier research); BYD "18 sales and service centres" in Aug 2024 | 2026 / 2024 | byd.md; CleanTechnica via search | [F]/[S] |
| Toyota in Uzbekistan | 4 dealers, 5 centres (Toyota Central Asia site) | Oct 2026 | toyota-centralasia.com | [F] |
| Car-wash enterprises | 3,122 legal entities | 1 Feb 2026 | stat.uz | [F] |
| Auto/motorcycle maintenance services volume | 10.9 trillion UZS for Jan-Aug 2025 (+10.1%); Tashkent city 2.0 trillion | 2025 | stat.uz | [F] |
| **Number of service stations (STO)** | **Not found.** Directory listing "Golden Pages" shows 163 entries (a directory, not a census) [S] | | | |
| Kazakhstan analogue | 7.8k STO in 2024 (6.5k in 2022); small firms 82.4% of revenue | 2024 | bizmedia.kz | [F] |

Arithmetic [I]:

- Maintenance volume 10.9 trillion UZS over 8 months is about $916M at 11,897.90; annualised roughly 16.4 trillion UZS (about $1.37B). Divided by about 4.9M individually owned vehicles that is about 3.3M UZS (about $280) per vehicle per year; a crude upper bound because the volume also covers company vehicles and motorcycles.
- A ceiling for **official dealer service sites** in Uzbekistan: about 190-250 sites if every dealership has service and counting loosely. At an assumed $300-600 per site per month that is about $0.7M-1.8M per year of recurring revenue. The price is my assumption, built from Detect Auto's implied ceiling (about $170/month per store) and SmartRail's tariff (up to about $660/month), not a quote.
- Therefore a startup that wants more than that must either (a) reach independent STO, for which no count exists, (b) sell regionally (Kazakhstan has 7.8k STO), or (c) sell per-inspection products (insurers, car-sharing, marketplaces).
- Ask Nacstatkom for the number of legal entities under the maintenance and repair of motor vehicles activity code (they published the same figure for car washes) and the MTPL fund or traffic police for the technical-inspection station count [I].

---

## 5. Lessons for us (Q5)

### 5.1 Pitfalls to avoid

1. **Financing hardware yourself.** ProovStation's collapse is the textbook case: rental/lease model, hiring ahead of sales, slow clients [F]. UVeye survives only on a $150M debt line [F]. In a hackathon and in a startup, prefer customer-owned commodity hardware (local RTX-class PC) and charge for software and support.
2. **Selling a product that makes your customer's customer angry.** Hertz/UVeye $440 scuff fee; Ravin's CEO left rental over this [F]. If we build (B) or (C), explanations, human appeal, severity thresholds and a repair-cost estimate must come with every flag.
3. **Overclaiming accuracy.** Even leaders disclaim: Anyline says outputs are estimates [S]; Ravin's own demo missed chips [F]; SmartRail quotes three different accuracy ranges in two months [F]. Publish our own test-set numbers and failure cases (the hackathon scores this).
4. **Per-person monitoring.** Technician-level idle time is the sensitive element (earlier report); SKAIVISION concedes a "Big Brother" worry [S]. Default to bay-level and team-level views.
5. **Assuming install equals adoption.** An unused drive-over tyre scanner at a Toyota dealer is an anecdote [S], but the pattern is plausible: the front-desk workflow must change [I].
6. **Expecting a 3-month pilot to convert.** Comparable cycles were 2-3 years [F].
7. **Being a generic vision vendor.** Detect Auto's own CEO says technology alone is not enough [F]; its headline product moved from cameras to a software assistant [F].

### 5.2 Which business model works

| Model | Who uses it | Evidence | Fit for us |
|---|---|---|---|
| Per camera SaaS | Ivideon and similar cloud video (earlier report: 200 RUB per camera/month to $55-104 for face analytics) | Cameras locally about $32; buyer compares to CCTV | Weak anchor. Use per camera only as an internal cost driver [I] |
| **Per site or per lane subscription, hardware bundled** | UVeye ($6-7K/lane/month incl. financing), SmartRail (up to 300,000 KZT/month, free hardware), Autoscan flat rate, Bayflow cancel-anytime | UVeye 1,000+ systems [F]; SmartRail pilot [F] | Best for (A) and (B) if hardware is cheap and customer-owned or financed by a partner [I] |
| Per inspection or API | Tchek, Monk, Ravin, Inspektlabs, ProovStation pass fee | VW FS EUR 300 saved per return [F]; ProovStation per pass was part of a model that still failed on hardware financing [F] | Best for (C) and insurers; unit economics easy to explain |
| Capital sale + support | Hunter, DeGould, ProovStation after April 2026 | ProovStation moved here [F] | For (B) if a partner (integrator) holds the balance sheet |
| Recurring monitoring | EyeQ | 95%+ retention (vendor) [F] | Not available to a pure CV startup, but argues for ongoing service contracts |
| Revenue share / per lead | SmartRail "per lead" and CPA for dealers [F] | Early; no result | Interesting for (B): payment tied to the repair job the scan generates |

### 5.3 What buyers objected to

- Customers: opaque, fast, hard-to-contest damage charges (Hertz) [F]; camera in a loaner car (Audi/Lytx report) [S].
- Dealers: cost and disruption of hardware [S]; paying for cameras whose video goes to the OEM (AvtoVAZ; earlier report); fear their idle time is exposed to the OEM [I].
- Employees: surveillance of individuals (earlier report on Labour Code).
- Rental operators: financing the scanners (ProovStation) and slow internal decisions [F].
- Everyone: unverified accuracy claims; no independent audits exist for UVeye, Hunter, Anyline or SmartRail [S].

### 5.4 Reading for our three candidates [I]

- **(A) service-flow analytics:** proven abroad (SKAIVISION at a Toyota dealer, PitCrew at Discount Tire), thin in CIS (one Kazakh tender). Revenue ceiling in Uzbek official dealers is small (about $0.7-1.8M/year at assumed prices), so (A) is a good **partner-mode** fit with Royal Motors (Toyota, BYD, Ducati) and a weak standalone startup.
- **(B) entrance inspection:** the closest live comparable is Pulsar.ai at a Hyundai dealer; heavy hardware players are strained (ProovStation) or very well funded (UVeye). If chosen, copy Pulsar's per-site subscription and "lead for the repair job" framing and **avoid hardware financing**; Ravin already claims a Toyota CCTV-gate tool, so expect the question "why not Ravin" [F].
- **(C) phone-based photo inspection:** crowded and global (Ravin, Tchek, Monk/ACV, Inspektlabs, Tractable), and consumer trust issues are documented; the Uzbek angle is insurers, car-sharing and marketplaces, where nobody was found, but we also found no signs of insurer demand [I].
- **Differentiation that is plausible to defend:** on-site processing and events-only output (legal point in earlier report); local plates and Telegram/SMS; multi-brand cross-site KPIs; explainable outputs [I].

### 5.5 Pricing anchors collected

| Anchor | Value | Tag |
|---|---|---|
| UVeye per lane | $6,000-7,000 per month (earlier report) | [S] |
| SmartRail per site | Up to 300,000 KZT per month, about $660 | [F] |
| Detect Auto per store | At most about $2,000 per year (derived) | [I] |
| Ivideon face-analytics add-on | $55-104 per camera per month (earlier report) | [F via aggregator] |
| Local IP camera | From 380,000 UZS (about $32) (earlier report) | [S] |
| VW FS saving per lease return | EUR 300 (vendor-quoted) | [F] |
| Photo vendors replaced by UVeye Scan to Sold | $4,000+ per month | [F] |

---

## 6. Not found (gaps)

- Number of service stations (STO) and independent garages in Uzbekistan; official dealer service count after Jan 2024 (only brand pages: Toyota 4 dealers/5 centres, BYD 30).
- Fleet of legal entities in Uzbekistan; the 1 Jul 2026 stat.uz page for 4.98M (only seen in search summaries).
- Any Uzbek startup or insurer with CV inspection; Startup Garage and MIT Enterprise Forum alumni lists; any Uzbek ANPR startup.
- Prices for Ravin, Tchek, Monk, Inspektlabs, Anyline, Hunter, Hikvision/Dahua UVSS, SKAIVISION, PitCrew, EyeQ.
- Award of the VisionFlow AI and AstanaHub damage-assessment tenders; current status of Platforma; 2025-2026 status of WeProov, Inspektlabs, DeGould.
- Independent accuracy audits of any vendor.
- Details of Ravin's Toyota service-centre deployment.

## 7. Sources

Read in full or in part ([F]):

- Ravin Series A, Globes (16 Mar 2021): https://en.globes.co.il/en/article-1001364268
- Ravin seed, TechCrunch (21 May 2019): https://techcrunch.com/?p=1829746
- Ravin media page: https://www.ravin.ai/media
- Ravin service centres: https://www.ravin.ai/solutions-service-centres
- The Drive on Hertz/UVeye (23 Jun 2025): https://www.thedrive.com/news/this-is-what-happens-when-hertzs-ai-scanner-finds-damage-on-your-rental
- The Drive on Ravin consumer tool (12 Aug 2025): https://www.thedrive.com/news/ai-rental-scanner-unfairly-charging-you-with-damage-theres-ai-for-that
- UVeye CEO interview, Pulse 2.0 (15 Feb 2026): https://pulse2.com/uveye-profile-amir-hever-interview/
- UVeye $191M, Calcalist (29 Jan 2025): https://www.calcalistech.com/ctechnews/article/sysqb00po1x
- UVeye Scan to Sold, Digital Dealer (2 Jun 2026): https://digitaldealer.com/news/uveye-launches-scan-to-sold-same-day-vehicle-listings-for-car-dealers/170980
- ProovStation restructuring, Journal de l'Automobile (23 Jan 2026): https://journalauto.com/services/proovstation-contraint-de-restructurer-sa-dette/
- ProovStation takeover (17 Apr 2026): https://journalauto.com/services/proovstation-a-trouve-un-repreneur/
- Sixt and ProovStation (10 Jan 2024): https://journalauto.com/services/sixt-et-proovstation-passent-a-lechelle-superieure-de-linspection
- Tchek and Volkswagen Financial Services (12 Feb 2026): https://journalauto.com/journal-des-flottes/volkswagen-financial-services-retient-tchek-pour-les-inspections-de-vehicules/
- SKAIVISION and JM Family (11 Jul 2023): https://jmfamily.com/news/skaivision-secures-investment-from-jm-family-enterprises/
- SKAIVISION Embedded Vision Summit page: https://www.edge-ai-vision.com/2025/10/skaivision-transforming-automotive-dealerships-with-computer-vision-a-presentation-from-skaivision/
- Leverege PitCrew: https://www.leverege.com/pitcrew
- Bayflow: https://autoflow.com/bayflow/
- EyeQ decade post (Apr 2026): https://eyeqmonitoring.com/2026/04/what-a-decade-of-ai-driven-growth-taught-eyeq-monitoring/
- EyeQ acquires MDL autoMation (Nov 2020): https://eyeqmonitoring.com/?p=6496
- Detect Auto founder series, gener8tor: https://www.gener8tor.com/news/gener8tor-founder-series-jon-cabak-from-detect-auto
- Anyline 100M inspections (18 Dec 2025): https://anyline.com/news/anyline-surpasses-100-million-tire-inspections
- Pulsar.ai / Smart Rail, Digital Business (2 Apr 2026): https://digitalbusiness.kz/2026-04-02/shazam-dlya-avtomobilya-kazahstanskiy-startap-nauchil-ii-nahodit-polomki-za-2-sekundi/
- Pulsar.ai, Kapital.kz (11 Jun 2026): https://kapital.kz/tehnology/149021/startap-iz-kazahstana-hochet-snizit-dtp-s-pomoshyu-ii-diagnostiki-avto.html
- Kazakh STO market, bizmedia.kz (10 Nov 2025): https://bizmedia.kz/2025-11-10-czeny-na-remont-i-obsluzhivanie-mashin-v-kazahstane-snova-rastut/
- Kazakh STO revenue shares, bizmedia.kz (18 Jul 2024): https://bizmedia.kz/2024-07-18-kazahstanczy-stali-chashhe-chinit-mashiny-na-sto/
- Platforma photo damage service (16 Sep 2021): https://incrussia.ru/news/platforma-zapustila-servis/
- TASS Vision, Digital Business (3 Nov 2025): https://digitalbusiness.kz/en/2025-11-03/entrepreneurs-from-tashkent-created-ai-cameras-for-stores-and-raised-1-8m/
- IT Park Computer Vision AI Challenge results (28 Jun 2022): https://it-park.uz/ru/itpark/news/obyavleny-itogi-hakatona-computer-vision-ai-challenge
- Sug'urta Servis app: https://apps.apple.com/us/app/-/id6448982242
- Avtoelon official dealers, Spot (14 Nov 2025): https://www.spot.uz/ru/2025/11/14/avtoelon-uz/
- Individually owned vehicles, 1 Apr 2026, stat.uz: https://stat.uz/ru/press-tsentr/novosti-goskomstata/68807-zbekistonda-a-oliga-tegishli-avtotransport-vositalari-soni-4-9-mlntadan-oshdi-3
- EV parc 1 Jul 2026, stat.uz: https://stat.uz/ru/press-tsentr/novosti-goskomstata/70263-zbekistonda-elektromobillar-soni-122-7-mingtaga-etdi-3
- EV parc 1 Apr 2026, stat.uz: https://stat.uz/ru/press-tsentr/novosti-goskomstata/68832-zbekistonda-elektromobillar-soni-109-2-mingtaga-etdi-3
- Car-wash enterprises, stat.uz (20 Feb 2026): https://stat.uz/uz/matbuot-markazi/qo-mita-yangiliklar/66964-zbekistonda-avtomojka-lar-soni-nechta-2
- Maintenance services volume, stat.uz (2 Oct 2025): https://stat.uz/uz/matbuot-markazi/qo-mita-yangiliklar/64279-avtotransport-vositalarini-tamirlash-khizmatlariga-talab-ortib-bormo-da-2
- 2025 car sales, Spot (17 Jan 2026): https://www.spot.uz/ru/2026/01/17/cars-sales-2025/
- EV imports, Kun.uz (11 May 2026): https://kun.uz/en/news/2026/05/11/ev-imports-to-uzbekistan-surge-fourfold-in-early-2026
- KPMG market analysis (via MIIT portal): https://invest.miit.uz/document/services/portal/SECTORS/Automotive/Documents/ENG_Анализ рынка автотранспортных средств в Республике Узбекистан_ENG (2).pdf ; KPMG page: https://kpmg.com/uz/ru/home/insights/2024/10/auto-market-research.html
- Toyota Central Asia dealers: https://toyota-centralasia.com/en/dealers
- Chevrolet Uzbekistan dealers page (no count shown): https://chevrolet.uz/ru/dealers

Search summaries only ([S]):

- Ravin Crunchbase/OurCrowd/LinkedIn profiles: https://www.crunchbase.com/organization/ravin
- UVeye funding aggregators: https://www.caplight.com/company/uveye ; https://www.premieralts.com/companies/uveye/valuation ; https://trueup.io/co/uveye
- Hertz/UVeye complaints (CO/AI): https://getcoai.com/news/hertzs-ai-scanners-catch-5x-more-car-damage-surprising-customers-with-bills/
- Monk and ACV: https://investors.acvauto.com/news-events/press-releases/detail/52/acv-acquires-monk-sas-a-leading-ai-powered-imaging-and-vehicle-inspection-platform ; https://www.eu-startups.com/2020/06/paris-based-monk-raises-e2-1-million-to-expand-its-ai-based-car-damage-inspection-system/
- Tchek: https://www.eu-startups.com/2021/01/french-startup-tchek-lands-e3-million-for-its-ai-driven-car-damage-inspection-tech/ ; https://een.ec.europa.eu/node/46558
- ProovStation 2023 round: https://www.businesswire.com/news/home/20230126005107/en ; takeover: https://www.largus.fr/pros/actualite-automobile/rachete-par-atermes-proovstation-poursuit-son-deploiement-a-l-international-30046397.html ; https://www.auto-infos.fr/article/l-entreprise-proovstation-rachetee-par-l-industriel-atermes.291700 ; NVIDIA profile: https://blogs.nvidia.com/blog/2022/10/17/proovstation-gpu-ai-appraisals/
- DeGould: https://degould.com/ ; https://www.automotivelogistics.media/vehicle-logistics/degould-secures-funding-for-vehicle-inspection-future/207547 ; https://www.ajot.com/news/wallenius-wilhelmsen-accelerates-digitalisation-processes-through-new-investment
- WeProov: https://www.cbinsights.com/company/weproov ; https://j2rauto.com/carrosserie/weproov-revolutionne-linspection-automobile-avec-la-proovstation
- Inspektlabs: https://www.cbinsights.com/company/inspektlabs ; https://digitalterminal.in/amp/story/news/19560.html
- Anyline: https://anyline.com/news/record-breaking-performance-in-2023 ; https://www.autoremarketing.com/ar/retail/anyline-fleetback-partner-to-streamline-tire-inspections-by-dealerships ; https://tyrepress.com/tag/tyre-scanner
- Hunter Quick Tread: https://vehicleservicepros.com/12022741 ; https://www.tirereview.com/digital-vehicle-inspection-hunter-engineering/
- Hikvision UVSS: https://content.hikvision.com/hikvision-under-vehicle-surveillance-system ; Dahua: https://networkcamerastore.com/products/dahua-dh-mv-vdf5020ce-00-fixed-under-vehicle-scan-system
- Leverege: https://www.businesswire.com/news/home/20250924410034/en/Cohere-Capital-Completes-Strategic-Growth-Investment-in-Leverege-a-Leading-VisionAI-Technology-Company ; https://www.ropesgray.com/en/news-and-events/news/2025/09/cohere-capital-completes-strategic-growth-investment-in-leverege-a-visionai-technology-company
- Autoflow: https://autoflow.com/ai-info ; https://www.cbinsights.com/company/autoflow
- EyeQ and WestView: https://www.trysignalbase.com/news/acquisitions/eyeq-monitoring-acquired-by-westview-capital-partners-acquisition
- Detect Auto seed: https://www.detectauto.com/blog/detect-auto-secures-748k-in-seed-funding-to-enhance-auto-repair-with-ai-technology ; https://fargoinc.com/748k-seed-funding-round-has-detect-auto-ready-for-takeoff
- SKAIVISION dealer-fear and Big Brother: https://www.autosuccessonline.com/?p=20699 ; https://skaivision.lovable.app/blog/ai-on-the-cameras-dealerships-already-own
- Audi loaner camera: https://www.carscoops.com/?p=3467119
- Tractable: https://webrazzi.com/2023/07/18/tractable-65-milyon-dolar-yatirim-aldi/
- Adomea MIKo: https://tuv.com/landingpage/en/new-mobility/topics/adomea-scanner.html ; https://www.zr.ru/content/news/920401-novyj-pribor-dlya-otsenki-behushek/
- BL Innovare: https://automechanika.messefrankfurt.com/frankfurt/en/exhibitor-search.detail.html/bl-innovare/mf_1_0052622294_3860787_10000007202401.html
- Autoscan GmbH / Dubai items: https://airportshow.german-pavilion.com/en/sites/exhibitors/136458 ; https://blog.oneclickdrive.com/dubai-autocheck-360-revolutionizes-car-inspection-gitex-2025/
- Turkey: https://apps.apple.com/us/app/id6747200897 (EkspertizAI) ; https://ageas.com/en/newsroom/new-digital-claims-process-speeds-up-claims-and-helps-combats-fraud-in ; https://tr.linkedin.com/in/muratgunay
- Kolesa Group AI: https://astanahub.com/en/blog/kolesa-group-otrazili-25-000-popytok-nedobrosovestnykh-razmeshchenii-obiav
- AstanaHub damage-assessment tender: https://astanahub.com/en/tech_task/razrabotka-intellektualnogo-servisa-dlia-avtomaticheskoi-otsenki-povrezhdenii-avtomobilia
- Gross Insurance app: https://gross.uz/en/news-view/88
- Smart parking Tashkent: https://kun.uz/ru/news/2022/11/24/v-aeroportu-tashkenta-zarabotala-pervaya-umnaya-parkovka-v-strane ; https://www.spot.uz/ru/2025/02/12/parking/ ; https://www.intellinews.com/hungarian-it-company-icell-introduces-smart-parking-project-in-uzbekistan-332608
- Startup Garage launch: https://ad.kun.uz/en/news/2024/07/22/tashkent-launches-startup-garage-to-foster-innovation-and-investment
- Uzbek 2025 vehicle parc 4.98M summaries: https://anhor.uz/ru/news/avtomobili-uzbekistana-5-millionov ; https://podrobno.uz/cat/obchestvo/v-uzbekistane-chislo-lichnykh-mashin-vplotnuyu-priblizilos-k-5-millionam/
- BYD Uzbekistan dealer centres 2024: https://cleantechnica.com/2024/08/13/byds-other-new-factory-in-uzbekistan/
- Earlier research (BYD dealer list): /Users/diznect/Downloads/hackathon_final/research/byd.md (source https://bydauto.uz/ru/contacts)
