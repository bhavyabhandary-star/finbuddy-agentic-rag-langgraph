"""Labeled training examples for the Intent Router.

Replaces the 20-example placeholder seed set (see git history) with a properly
sized, realistic dataset per build_prompt.md's MLOps "Data & Feature" step.

Honesty note: these are realistic *constructed* examples, not live user logs —
no live traffic exists pre-launch, so there is no real query log to draw from
yet. "Policy" examples are deliberately grounded in the actual topics covered
by the 5 real RBI/DPDP PDFs ingested in milestone step 2/3 (data retention,
Consent Managers, Account Aggregator eligibility, Default Loss Guarantee,
cooling-off periods, DPDP rights/obligations) plus a handful of general
coaching-style questions — the router's job is only to pick a *path*
(retrieval vs credit tools vs escalate), not to predict corpus coverage; the
RAG sufficiency check (tools/rag_tools.py) is the layer responsible for
catching a policy-shaped question the corpus doesn't actually cover.

Once this project has real traffic, replace/augment this file with an actual
labeled query log — re-run mlops/intent_router/train.py, and only promote a
new model if it beats the current one on eval/evaluate_agent.py (see the
Intent Router's governance gate in build_prompt.md — a worse model must never
reach production, as already demonstrated once during scaffolding).
"""
from __future__ import annotations

ROUTES = ("policy", "credit_assessment", "off_topic")

# --- policy: data retention / storage location (RBI Storage of Payment System
# Data 2018; DPDP Rules 2025 retention provisions) ---------------------------
_POLICY_DATA_RETENTION = [
    "how long do you keep my transaction data",
    "where is my payment data actually stored",
    "is my data stored only in India",
    "what happens to my data after I close my account",
    "do you delete my data if I stop using the app",
    "how long are logs of my UPI transactions kept",
    "can my payment data be stored outside the country",
    "what's the retention period for my financial records",
    "is there a time limit on how long you can hold my data",
    "does RBI require my payment data to stay in India",
    "why does my data need to be stored locally",
    "what's your data storage policy",
]

# --- policy: Consent Manager role and obligations (DPDP Act 2023 / Rules 2025) ---
_POLICY_CONSENT_MANAGER = [
    "what is a Consent Manager",
    "who is responsible for managing my consent",
    "can I withdraw my consent through a Consent Manager",
    "what are a Consent Manager's obligations to me",
    "is the Consent Manager accountable to me or to the company",
    "how do I revoke consent I gave earlier",
    "what happens if I refuse consent",
    "does the Consent Manager share my data with anyone else",
    "how is my consent recorded and tracked",
    "can I see a history of consents I've given",
]

# --- policy: Account Aggregator eligibility / registration (RBI AA Master
# Direction 2016) -------------------------------------------------------------
_POLICY_ACCOUNT_AGGREGATOR = [
    "what is an Account Aggregator",
    "who can register as an Account Aggregator NBFC",
    "what's the minimum net owned fund for an Account Aggregator",
    "does an Account Aggregator ever see my raw financial data",
    "is my consent required before an Account Aggregator shares my data",
    "how is an Account Aggregator different from a bank",
    "can an Account Aggregator sell my financial information",
    "what license does a company need to operate as an Account Aggregator",
    "how does the Account Aggregator framework protect my data",
    "why do I need to go through an Account Aggregator at all",
]

# --- policy: Default Loss Guarantee (RBI Digital Lending Directions 2025) ---
_POLICY_DLG = [
    "what is a Default Loss Guarantee",
    "what's the cap on Default Loss Guarantee arrangements",
    "how does DLG protect the lender",
    "is DLG the same as FLDG",
    "can DLG be used for credit card lending",
    "who provides the Default Loss Guarantee in a digital loan",
    "what happens if the DLG provider can't honour the guarantee",
    "how is Default Loss Guarantee different from insurance",
]

# --- policy: cooling-off period / digital lending disclosures (RBI Digital
# Lending Directions 2025) -----------------------------------------------------
_POLICY_DIGITAL_LENDING_DISCLOSURES = [
    "what is the cooling-off period for a digital loan",
    "can I cancel a loan within the cooling-off window",
    "what disclosures must a lending app show me before I borrow",
    "what is a Key Fact Statement",
    "do I get a full breakdown of interest and fees before agreeing",
    "what happens if a lending app doesn't disclose all charges upfront",
    "are multi-lender arrangements disclosed to me",
    "how do I know if my loan involves more than one lender",
]

# --- policy: DPDP rights, obligations, grievance, cross-border transfer -----
_POLICY_DPDP_GENERAL = [
    "what rights do I have over my personal data",
    "how do I file a complaint about misuse of my data",
    "what is a Data Fiduciary's responsibility to me",
    "can my data be transferred outside India",
    "what happens if there's a data breach",
    "will I be notified if my data is breached",
    "does the data protection law apply to children's data differently",
    "what is the Data Protection Board",
    "can I ask a company to erase my personal data",
    "what counts as personal data under the law",
    "do I have to consent again if the purpose of data use changes",
    "what are the penalties for a company that misuses my data",
]

# --- policy: general coaching-style informational questions (may or may not
# be covered by the ingested corpus — that's the sufficiency check's job to
# determine, not the router's) -------------------------------------------------
_POLICY_GENERAL_COACHING = [
    "why was my credit limit reduced",
    "how do I improve my repayment score",
    "why do you need my UPI transaction data",
    "is my data shared with third parties",
    "what does consent mean for me on this app",
    "why did my application get flagged",
    "how is my income regularity calculated",
    "what counts as a missed payment",
]

POLICY_EXAMPLES = [
    (t, "policy")
    for t in (
        _POLICY_DATA_RETENTION
        + _POLICY_CONSENT_MANAGER
        + _POLICY_ACCOUNT_AGGREGATOR
        + _POLICY_DLG
        + _POLICY_DIGITAL_LENDING_DISCLOSURES
        + _POLICY_DPDP_GENERAL
        + _POLICY_GENERAL_COACHING
    )
]

# --- credit_assessment: score checks, approval, risk trend, borrowing -------
_CREDIT_SCORE_CHECK = [
    "can you check my credit score",
    "what's my current credit score",
    "score my UPI profile",
    "run a credit check on my account",
    "I want to know my credit rating",
    "give me my score based on my transactions",
    "how good is my credit profile right now",
    "what score would I get today",
]

_CREDIT_APPROVAL = [
    "am I approved for a loan",
    "will I get approved if I apply now",
    "what's my chance of loan approval",
    "can you tell me if I qualify for credit",
    "am I eligible for a loan based on my income",
    "would I pass the approval criteria",
]

_CREDIT_RISK_TREND = [
    "is my financial trend improving or getting worse",
    "what's my risk trend this month",
    "am I trending toward higher risk",
    "is my financial situation getting better or worse",
    "how has my repayment trend changed recently",
    "is my spending pattern flagged as risky",
]

_CREDIT_BORROWING_CAPACITY = [
    "how much can I borrow based on my transactions",
    "what's the maximum loan amount I qualify for",
    "how much credit can I access right now",
    "what loan amount matches my income level",
    "can I get a bigger loan than last time",
]

_CREDIT_MISC = [
    "evaluate my creditworthiness",
    "why did my score change since last month",
    "is my account flagged as anomalous",
    "does my transaction pattern look unusual to you",
    "what factors are hurting my score the most",
    "what's dragging my score down",
    "assess my UPI-based credit profile",
    "check if I'm creditworthy for a small loan",
]

_CREDIT_MORE_VARIANTS = [
    "rate my financial profile",
    "how creditworthy am I",
    "pull up my score",
    "I need my credit assessment",
    "generate my credit report",
    "what's my repayment probability",
    "how likely am I to repay on time",
    "calculate my credit score from my transactions",
    "show me my income-based credit rating",
    "what does my calibrated score look like",
    "give me a breakdown of my top score factors",
    "explain the top reasons behind my score",
    "is my income regularity hurting my score",
    "how does my transaction diversity affect my score",
    "what's my approval probability for a bigger loan",
    "recheck my score with updated transactions",
    "did my recent transactions improve my score",
    "will a higher income raise my score",
    "what income band am I in",
    "am I in the low or high risk category",
    "how does my b2b transaction ratio affect my score",
    "can you re-run my assessment",
    "is my profile considered high risk right now",
    "what would happen to my score if my income dropped",
    "check whether I'd qualify for a top-up loan",
    "give me my updated risk classification",
    "how many months of tenure do I need for a better score",
    "does merchant diversity improve my credit score",
]

CREDIT_ASSESSMENT_EXAMPLES = [
    (t, "credit_assessment")
    for t in (
        _CREDIT_SCORE_CHECK
        + _CREDIT_APPROVAL
        + _CREDIT_RISK_TREND
        + _CREDIT_BORROWING_CAPACITY
        + _CREDIT_MISC
        + _CREDIT_MORE_VARIANTS
    )
]

# --- off_topic: clearly outside FinBuddy's scope ----------------------------
_OFF_TOPIC_EXAMPLES_RAW = [
    "what's the weather like today",
    "what stock should I invest in this week",
    "tell me a joke",
    "who won the cricket match",
    "what's a good recipe for dinner tonight",
    "recommend me a movie to watch",
    "what's the capital of France",
    "how do I fix my wifi router",
    "what time zone is Tokyo in",
    "can you write me a poem",
    "what's trending on social media today",
    "help me plan a trip to Goa",
    "what's the best programming language to learn",
    "how do I train for a marathon",
    "what's the news today",
    "should I buy crypto right now",
    "who is the prime minister of India",
    "what's a good gift for my mom's birthday",
    "how do I make my phone battery last longer",
    "what's the meaning of life",
    "translate this sentence to French",
    "what's a fun fact about space",
    "can you help me with my homework",
    "what's the best diet for weight loss",
    "how do airplanes fly",
    "tell me about the history of Rome",
    "what's a good workout routine",
    "how do I learn to cook Italian food",
    "what's happening in the stock market today",
    "recommend a good book to read",
    "good morning, how are you",
    "what's your favorite color",
    "can you sing me a song",
    "what's the best phone to buy this year",
    "how do I get better at chess",
    "what's a good name for my new puppy",
    "explain how black holes work",
    "what's the tallest mountain in the world",
    "how do I learn guitar",
    "what's a good password manager",
    "give me tips for public speaking",
    "how does the stock market work in general",
    "what's the plot of that new movie",
    "how do I remove a stain from my shirt",
    "what's a good podcast to listen to",
    "how do I set up a home wifi network",
    "what's the fastest way to learn a new language",
    "tell me about ancient Egypt",
    "what's a good exercise for back pain",
    "how do I bake a chocolate cake",
    "what's the score of last night's match",
    "can you help me write a resume",
    "what's a good vacation spot in December",
    "how do solar panels work",
    "what's the best way to study for exams",
    "recommend a TV show to binge watch",
    "how do I improve my sleep schedule",
    "what's a fun weekend activity",
    "how do I fix a flat tire",
    "what's the difference between a crocodile and an alligator",
]

OFF_TOPIC_EXAMPLES = [(t, "off_topic") for t in _OFF_TOPIC_EXAMPLES_RAW]

ALL_EXAMPLES: list[tuple[str, str]] = POLICY_EXAMPLES + CREDIT_ASSESSMENT_EXAMPLES + OFF_TOPIC_EXAMPLES


def load_training_examples() -> tuple[list[str], list[str]]:
    texts = [t for t, _ in ALL_EXAMPLES]
    labels = [label for _, label in ALL_EXAMPLES]
    return texts, labels
