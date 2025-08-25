# Core strategy & discovery (why/for whom)

* For any adult over 18 that has a house, car, valuables that they need to insure and are looking to renew their insurance
* [hypothesis] Current insurance renewal process is broken - it's manual, time consuming and opaque
* [hypothesis] Most customers are not satisfied with how much they are paying for their insurance
 * [hypothesis .1] Most customers do not understand the subcomponents to their insurance and how they apply to their situations
 * [hypothesis .2] Prices for insurance are opaque, shopping process is so cumbersome that most people give-up after their first few data points
   * Lack of standardized process or centralized price place to cross shop
   * There is a mesh of companies and their agents, each entity has a unique way of receiving information - either through forms OR phone / email
   * A lot of dealings are done in human-native ways which have traditionally been hard to automate

# Product definition (what)
* Take the existing insurance coverage either as a pdf of the coverage document OR a screenshot 
* Parse the document to ...
  * Understand the current coverage, price point and payment frequency 
  * Understand the coverage parameters, e.g. bodily liability for auto is (150/300, 250/500 etc.)
  * Understand area of residence (zip code should be fine)
* Shop for coverage from insurance companies that service the area of residence
* Get price ranges from different providers and provide to the customer
* Help with selection, recommend an option to the customer. Draw comparison to other options
* Recommend modifications to the policy if any 


# Engineering & data (how)
* Use OCR OR LLM to assess the existing insurance coverage
  * Setup two way dialog with user to confirm all points before proceeding
  * Consolidate frequently asked information so it can be filled through forms / voiced over
* Use a research agent to understand various attributes of the asset
  * Key tasks - Build fact base about assets to be insured. e.g. type of roof on the home, house layout, year built, type of floor etc.
  * Tools needed -
  *   Internet Access: To look at asset pictures, bring up window sticker against car vin, or sale listing of the home
  *   Call APIs of redfin, NMLS etc. to understand sale listing and get details about the home/car etc
* Use an insurance research agent to research companies and agents serving the area of residence
  * Key tasks - Locate companies, use information on assets and prior insurance to get coverage quotes
    * Locate forms to fill out and begin filling them out, using previously collected information
    * Use foundational models to understand assets and make logical extrapolations to fill in gaps -- will want this to be flagged so it can be taken to the user for confirmation/amends
  * Tools needed -
    * Internet Access: To locate companies in specific areas, gather contact details
    * Phone Access: To place calls to agents to assess offerings, have conversation around pricing etc.
    * Messaging/Email Access: For communication similar to above
* Present compiled options to user and update the list as the quotes come in
  * Breakdown coverage by the different components - like home, auto etc
  * Understand the subcomponents selected in the insurance - e.g. is there flood insurance, is there earthquake insurance etc


# Delivery & ops (when/risk)


# Key Documents

## Core Strategy
Problem One-Pager — crisp articulation of the user/job, constraints, and success.
Vision & North-Star — what great looks like in 12–18 months; success metrics.
Jobs/Personas & Journey Map — pains, triggers, moments of truth.
Research Brief & Insights Log — what we’ll learn, from whom, by when.

## Product definition (what)
Lean PRD — outcomes, users, use cases, non-functionals, guardrails, metrics.
MVP Scope & Release Plan — v0/v1/v2 cuts, feature flags, “won’t haves.”
Experiment Plan — hypotheses, test design, power, decision rules.
Analytics Spec (Event Taxonomy) — events, props, IDs, governance.

## Engineering & data (how)
Tech Spec — system context, sequence diagrams, key algorithms, tradeoffs.
API Contracts — request/response, error model, versioning.
Architecture Decision Records (ADRs) — one page per irreversible choice.
Security & Privacy Pack — data flow, retention, DPIA/PIA, threat model.
SLOs & Observability — golden signals, alerts, runbooks.

## Delivery & ops (when/risk)
Roadmap & RACI — who decides/builds/reviews.
RAID Log — risks, assumptions, issues, dependencies.
Launch Checklist & GTM Brief — readiness gates, comms, support.
Postmortem Template — blameless, action-items with owners/dates.

