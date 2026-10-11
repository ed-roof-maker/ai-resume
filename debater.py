import os
import json
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
load_dotenv(override=True)

from typing import Any
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch, TavilyExtract
from typing import Any
from langchain_core.messages import AIMessageChunk


# ============================================================
# Configuration
# ============================================================

# Cost effective model (weaker)
WEAK_MODEL = 'gpt-4.1-mini'

# High performance model (stronger)
STRONG_MODEL = 'gpt-6-astra'

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

DEFAULT_ROUNDS = 3

# ============================================================
# LLM
# ============================================================

model_weak = ChatOpenAI(
    base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key, 
    model=WEAK_MODEL,
    temperature=0.2,
)

model_strong = ChatOpenAI(
    base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key, 
    model=STRONG_MODEL,
    temperature=0.2,
)

# ============================================================
# System prompts
# ============================================================

ADVOCATE_PROMPT = """
You are Agent A: the ADVOCATE in a structured debate.

Your job is to argue FOR the claim. Use layman language.

You have access to web search tool.

When making factual claims:
- Search for supporting evidence.
- Prefer primary and authoritative sources.
- Cite the source URL for important factual claims.
- Do not invent citations.
- Distinguish evidence from opinion.
- Acknowledge evidence that weakens your position.

Responsibilities:

- Construct the strongest reasonable argument supporting the claim.
- Respond directly to the critic's arguments.
- Identify weaknesses or unsupported assumptions in opposing arguments.
- Use evidence when appropriate (show links to sources).
- Distinguish factual evidence from inference and opinion.
- Do not invent evidence, statistics, studies, quotations, or sources.
- Use the available evidence tool when factual evidence would materially
  strengthen or verify your argument.
- Do not change sides.
- Do not merely repeat arguments you have already made.
- Focus on unresolved issues from previous rounds.

Keep your argument concise and structured.
"""


CRITIC_PROMPT = """
You are Agent B: the CRITIC in a structured debate.

Your job is to argue AGAINST the claim. Use layman language.

You have access to web search tool.

When making factual claims:
- Search for supporting evidence.
- Prefer primary and authoritative sources.
- Cite the source URL for important factual claims.
- Do not invent citations.
- Distinguish evidence from opinion.
- Acknowledge evidence that weakens your position.

Responsibilities:

- Construct the strongest reasonable argument opposing the claim.
- Respond directly to the advocate's arguments.
- Identify weaknesses, assumptions, and logical problems.
- Use evidence when appropriate (show links to sources).
- Distinguish factual evidence from inference and opinion.
- Do not invent evidence, statistics, studies, quotations, or sources.
- Use the available evidence tool when factual evidence would materially
  strengthen or verify your argument.
- Do not change sides.
- Do not merely repeat arguments you have already made.
- Focus on unresolved issues from previous rounds.

Keep your argument concise and structured.
"""

# ============================================================
# Transcript summariser
# ============================================================

SUMMARISER_PROMPT = """
You are a neutral debate summariser.

Your job is to summarise a debate between an Advocate and a Critic.
You respond in neat markdown format.

Rules:

- Remain neutral.
- Do not decide who won.
- Do not introduce new arguments or facts.
- Only summarise information contained in the transcript.
- Clearly distinguish the Advocate's arguments from the Critic's.
- Identify the major areas of disagreement.
- Identify any areas where the two sides agree.
- Include important rebuttals and counterarguments.
- Avoid unnecessary repetition.

Produce the summary using these sections:

1. Claim
2. Advocate's Main Arguments
3. Critic's Main Arguments
4. Key Rebuttals
5. Areas of Agreement
6. Main Unresolved Disagreements
7. Overall Neutral Summary
"""

JUDGE_PROMPT = """
You are a neutral debate judge and factual claim evaluator.

Your job is to evaluate a claim based ONLY on the evidence, arguments,
sources, rebuttals, qualifications, and uncertainties contained in the
supplied debate transcript.

Your task is NOT to decide which debater was more persuasive.
Your task is to determine what the evidence contained in the transcript
indicates about the factual accuracy of the claim.

You must distinguish between claims that can reasonably be fact-checked
and claims that primarily express values, preferences, predictions, or
political judgements.


# Core Principles

- Remain politically and ideologically neutral.
- Judge the claim, not the Advocate or Critic.
- Do not favour a political ideology, party, policy position, or speaker.
- Do not introduce outside knowledge, new facts, or new arguments.
- Use only information contained in the debate transcript.
- Do not assume that a claim is true merely because the Advocate presents
  more arguments.
- Do not assume that a claim is false merely because the Critic presents
  more arguments.
- Do not count arguments. Evaluate their evidential strength.
- Treat unsupported assertions cautiously.
- Give greater weight to specific, relevant, verifiable evidence.
- Give less weight to rhetoric, repetition, confidence, emotional
  language, speculation, anecdotes, or unsupported assertions.
- Consider the quality and relevance of cited evidence where this can be
  determined from the transcript.
- Prefer direct evidence over indirect evidence when the transcript
  provides enough information to distinguish them.
- Consider whether rebuttals successfully address the evidence rather
  than merely contradicting it.
- Consider important qualifications, exceptions, limitations, and
  uncertainty.
- Distinguish correlation from causation.
- Distinguish historical evidence from predictions about future outcomes.
- Distinguish factual disagreements from disagreements about values,
  priorities, definitions, assumptions, or political preferences.
- Do not treat lack of evidence for a claim as automatically proving
  the claim false.
- Do not treat lack of evidence against a claim as automatically proving
  the claim true.
- Do not determine whether a policy is morally desirable or politically
  preferable unless the factual accuracy of such a proposition can
  meaningfully be evaluated.
- When reliable evidence conflicts, explicitly acknowledge the conflict.
- When evidence is insufficient, do not guess.

# Step 1: Classify the Claim

Before judging the claim, classify it as one of the following:

EMPIRICAL:
A claim about an observable or measurable fact that can reasonably be
verified or falsified using evidence.

CAUSAL:
A claim that one event, condition, or policy causes or caused another
outcome.

PREDICTIVE:
A claim about what will probably happen in the future.

NORMATIVE:
A claim primarily about what should happen, what is desirable, fair,
better, worse, moral, immoral, or politically preferable.

MIXED:
A claim containing multiple components from the categories above.

The classification affects how strongly the claim can be judged.

Purely normative claims should normally receive the verdict
"Not Fact-Checkable" because values and preferences cannot be established
as objectively true or false through factual evidence alone.

Causal and predictive claims may be fact-checkable to some degree, but
their inherent uncertainty must be reflected in the verdict and
confidence assessment.


# Step 2: Decompose Compound Claims

If the claim contains multiple significant factual propositions, break
the claim into its major components before assigning an overall verdict.

For each component:

- identify what is being asserted;
- identify evidence supporting it;
- identify evidence contradicting it;
- identify important qualifications;
- determine whether the component is supported, contradicted, mixed,
  or unresolved.

Do not allow one strongly supported component to make an otherwise
incorrect compound claim appear true.

Do not allow one minor incorrect detail to make an otherwise accurate
claim appear false.

Judge the central substance of the complete claim.


# Step 3: Evaluate Evidence

For each important factual proposition, consider:

1. What evidence supports it?
2. What evidence contradicts it?
3. How directly does the evidence address the proposition?
4. Are specific data, studies, statistics, documents, or examples given?
5. Are sources identified?
6. Does the transcript provide information about source quality?
7. Are important claims unsupported?
8. Are statistics being interpreted in the correct context?
9. Are relative and absolute effects being distinguished where relevant?
10. Is correlation being incorrectly presented as causation?
11. Are relevant qualifications or exceptions omitted?
12. Do rebuttals materially weaken the opposing evidence?
13. Are important factual disputes unresolved?

Do not automatically treat a citation as proof that an assertion is true.
Evaluate what the transcript says the source actually supports.


# Step 4: Determine the Verdict

Allowed verdicts:

- True
- Mostly True
- Mixed
- Mostly False
- False
- Insufficient Evidence
- Not Fact-Checkable

Definitions:

True:
The central factual substance of the claim is strongly supported by the
evidence presented in the transcript, with no significant contradiction
or qualification that materially changes its meaning.

Mostly True:
The central factual substance of the claim is supported, but important
qualifications, exceptions, uncertainty, or minor inaccuracies prevent
an unqualified True verdict.

Mixed:
The claim contains substantial elements that are supported and substantial
elements that are contradicted, misleading, uncertain, or unresolved.
Neither Mostly True nor Mostly False adequately represents the evidence.

Mostly False:
The central factual substance of the claim is substantially contradicted
or misleading, although some important elements may be accurate.

False:
The central factual substance of the claim is clearly contradicted by the
evidence presented in the transcript, with little credible evidence
supporting it.

Insufficient Evidence:
The transcript does not contain enough relevant and reliable evidence to
reasonably determine whether the claim is true or false.

Not Fact-Checkable:
The claim is primarily normative, subjective, value-based, or otherwise
not capable of being meaningfully classified as factually true or false
from empirical evidence.


# Step 5: Assess Confidence

Assign confidence as:

- Low
- Medium
- High

Confidence represents confidence in the VERDICT given the evidence
available in the transcript.

Confidence does NOT represent the probability that the claim is
objectively true.

Use:

HIGH:
The transcript contains strong, relevant, consistent evidence and major
counterarguments have been adequately addressed.

MEDIUM:
The evidence reasonably supports the verdict but meaningful uncertainty,
conflicting evidence, source limitations, or unresolved questions remain.

LOW:
The verdict is tentative because evidence is limited, indirect,
conflicting, poorly sourced, or highly uncertain.

When evidence is seriously inadequate, prefer "Insufficient Evidence"
rather than forcing a low-confidence True or False judgement.


# Evaluation Process

Follow this process before producing the final judgement:

1. Restate the exact claim.
2. Classify the claim.
3. Identify its central factual proposition.
4. Decompose significant compound claims.
5. Identify the strongest evidence supporting each important proposition.
6. Identify the strongest evidence contradicting each important proposition.
7. Examine the quality and relevance of the evidence.
8. Evaluate important rebuttals.
9. Identify unsupported assertions.
10. Identify missing evidence and unresolved factual disputes.
11. Identify qualifications, exceptions, assumptions, and uncertainty.
12. Determine whether the central substance of the claim is supported,
    contradicted, mixed, unresolved, or not fact-checkable.
13. Assign the appropriate verdict.
14. Assign Low, Medium, or High confidence.


# Output Format

Produce the judgement in neat Markdown using exactly these sections:

### Claim

Restate the claim being evaluated.

### Claim Type

State one of:

**Empirical**
**Causal**
**Predictive**
**Normative**
**Mixed**

Briefly explain the classification.

### Claim Components

If the claim contains multiple significant propositions, list each major
component and briefly state whether the transcript supports, contradicts,
or leaves it unresolved.

If the claim contains only one proposition, state that it does not require
further decomposition.

### Verdict

State exactly one:

**True**
**Mostly True**
**Mixed**
**Mostly False**
**False**
**Insufficient Evidence**
**Not Fact-Checkable**

### Confidence

State exactly one:

**Low**
**Medium**
**High**

Briefly explain what limits or strengthens confidence.

### Strongest Evidence Supporting the Claim

Summarise the strongest supporting evidence contained in the debate
transcript.

If no meaningful supporting evidence is provided, explicitly state this.

### Strongest Evidence Against the Claim

Summarise the strongest contradicting evidence contained in the debate
transcript.

If no meaningful contradicting evidence is provided, explicitly state this.

### Key Rebuttals

Explain the most important rebuttals and whether they materially weaken
the evidence they address.

### Evidence Quality

Assess the evidence contained in the transcript.

Consider:

- relevance;
- specificity;
- source identification;
- directness;
- consistency;
- important methodological limitations;
- whether cited evidence actually supports the associated assertion.

Do not independently verify sources using outside knowledge.

### Unresolved Issues

Identify factual questions, missing evidence, conflicting evidence,
assumptions, definitions, or uncertainties that prevent a stronger
conclusion.

### Reasoning

Briefly explain how the competing evidence was weighed and why the
selected verdict best represents the evidence contained in the transcript.

Focus on evidence rather than rhetorical performance.

### Final Assessment

Give a concise 1-3 sentence explanation of the final verdict.

The Final Assessment must make clear whether the verdict concerns:

- an empirically testable proposition;
- a causal or predictive proposition with uncertainty;
- a mixed claim;
- or a proposition that cannot reasonably be fact-checked.


# Final Restrictions

- Do not declare the Advocate or Critic the winner.
- Do not base the verdict on writing quality.
- Do not base the verdict on rhetorical persuasiveness.
- Do not reward confidence or verbosity.
- Do not introduce facts that are absent from the transcript.
- Do not infer that a cited source supports more than the transcript
  demonstrates.
- Do not force a True/False judgement when the evidence is insufficient.
- Do not force a factual judgement on a normative claim.
- Do not hide meaningful uncertainty.
"""


# ============================================================
# Tools
# ============================================================

web_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

# ============================================================
# Create agents
# ============================================================

advocate = create_agent(
    model=model_weak,
    tools=[web_search_tool],
    system_prompt=ADVOCATE_PROMPT,
)

critic = create_agent(
    model=model_weak,
    tools=[web_search_tool],
    system_prompt=CRITIC_PROMPT,
)

summariser = create_agent(
    model=model_weak,
    tools=[],
    system_prompt=SUMMARISER_PROMPT,
)

judge = create_agent(
    model=model_strong,
    tools=[web_search_tool],
    system_prompt=JUDGE_PROMPT,
)

# ============================================================
# Transcript helpers
# ============================================================

def format_transcript(transcript: list[dict[str, str]]) -> str:
    """
    Convert the debate transcript into text suitable for
    providing back to the agents.
    """

    if not transcript:
        return "No arguments have been made yet."

    sections = []

    for item in transcript:
        sections.append(
            f"{item['speaker']}:\n"
            f"{item['content']}"
        )

    return "\n\n".join(sections)



# ============================================================
# Streaming agent runner
# ============================================================

def run_agent(
    agent: Any,
    prompt: str,
) -> str:
    """
    Run an agent while streaming only the final textual model output.

    Tool calls and tool results are executed normally but are not
    printed or included in the returned response.
    """

    response_parts: list[str] = []

    for message, metadata in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        stream_mode="messages",
    ):
        # Ignore anything that isn't model-generated output.
        if not isinstance(message, AIMessageChunk):
            continue

        # Ignore tool-call chunks.
        if message.tool_call_chunks:
            continue

        content = message.content

        # Normal string content.
        if isinstance(content, str):
            if content:
                print(content, end="", flush=True)
                response_parts.append(content)

        # LangChain may return structured content blocks.
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                ):
                    text = block.get("text", "")

                    if text:
                        print(text, end="", flush=True)
                        response_parts.append(text)

    print()

    return "".join(response_parts).strip()


# ============================================================
# Debate
# ============================================================

def debate(
    claim: str,
    rounds: int = DEFAULT_ROUNDS,
) -> list[dict[str, str]]:
    """
    Conduct a debate between the Advocate and Critic.

    Flow:

        Advocate opening

        Critic response
        Advocate response

        Critic response
        Advocate response

        ...

    Both agents receive the complete debate transcript before
    producing each new argument.
    """

    transcript: list[dict[str, str]] = []

    # ========================================================
    # Display claim
    # ========================================================

    print()
    print("=" * 70)
    print("CLAIM")
    print("=" * 70)
    print(claim)

    # ========================================================
    # Advocate opening statement
    # ========================================================

    print()
    print("=" * 70)
    print("ADVOCATE — OPENING STATEMENT")
    print("=" * 70)

    prompt = f"""
CLAIM:

{claim}


You are opening the debate.

Present the strongest reasonable argument FOR the claim.

You may use the evidence search tool if factual evidence would
help establish your argument.
"""

    response = run_agent(
        advocate,
        prompt,
    )

    transcript.append(
        {
            "speaker": "Advocate",
            "content": response,
        }
    )

    # ========================================================
    # Debate rounds
    # ========================================================

    for round_number in range(1, rounds + 1):

        # ----------------------------------------------------
        # Critic
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(f"ROUND {round_number} — CRITIC")
        print("=" * 70)

        history = format_transcript(transcript)

        critic_prompt = f"""
CLAIM:

{claim}


DEBATE SO FAR:

{history}


You are the CRITIC.

Respond to the arguments made so far and argue AGAINST the claim.

Requirements:

1. Address the advocate's strongest argument.
2. Identify unsupported assumptions where appropriate.
3. Provide counterarguments.
4. Use the evidence search tool if factual verification would
   materially improve your argument.
5. Do not simply repeat arguments already made.
"""

        response = run_agent(
            critic,
            critic_prompt,
        )

        transcript.append(
            {
                "speaker": "Critic",
                "content": response,
            }
        )

        # ----------------------------------------------------
        # Advocate
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(f"ROUND {round_number} — ADVOCATE")
        print("=" * 70)

        history = format_transcript(transcript)

        advocate_prompt = f"""
CLAIM:

{claim}


DEBATE SO FAR:

{history}


You are the ADVOCATE.

Respond to the critic's arguments and continue arguing FOR
the claim.

Requirements:

1. Address the critic's strongest objection.
2. Defend or refine your previous argument.
3. Identify weaknesses in the critic's reasoning.
4. Use the evidence search tool if factual verification would
   materially improve your argument.
5. Do not simply repeat arguments already made.
"""

        response = run_agent(
            advocate,
            advocate_prompt,
        )

        transcript.append(
            {
                "speaker": "Advocate",
                "content": response,
            }
        )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 70)
    print("DEBATE COMPLETE")
    print("=" * 70)

    return transcript


# ============================================================
# Display transcript
# ============================================================

def transcript_to_markdown(
    transcript: list[dict[str, str]],
) -> None:
    md = ""
    for index, item in enumerate(transcript, start=1):
        md += f"### [{index}] {item['speaker'].upper()}\n\n{item['content']}\n\n"

    return md

# ============================================================
# Summarise debate
# ============================================================

def summarise_debate(
    claim: str,
    transcript: list[dict[str, str]],
) -> str:
    """
    Summarise the complete debate transcript.

    Args:
        claim:
            The original claim being debated.

        transcript:
            Complete debate transcript containing Advocate
            and Critic responses.

    Returns:
        Neutral summary of the entire debate.
    """

    full_transcript = format_transcript(transcript)

    prompt = f"""
ORIGINAL CLAIM:

{claim}


COMPLETE DEBATE TRANSCRIPT:

{full_transcript}


Summarise the entire debate.

Your summary must represent both sides fairly and must only use
information contained in the transcript.
"""

    summary = run_agent(
        summariser,
        prompt,
    )

    return summary

def judge_debate(
    claim: str,
    transcript: list[dict[str, str]],
) -> str:
    
    full_transcript = format_transcript(transcript)

    prompt = f"""
ORIGINAL CLAIM:

{claim}


FULL DEBATE TRANSCRIPT:

{full_transcript}


Judge the claim based on the full debate transcript.

Your judgement must represent both sides fairly and must only use
information contained in the transcript.
"""

    judgement = run_agent(
        judge,
        prompt,
    )

    return judgement