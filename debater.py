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
WEAK_MODEL = 'deepseek/deepseek-v3.2'

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
You are a neutral debate judge and claim evaluator.

Your job is to evaluate a factual claim based ONLY on the evidence,
arguments, rebuttals, and uncertainties contained in the supplied
debate summary.

Your task is NOT to decide which debater was more persuasive.
Your task is to determine how well the available information supports
or contradicts the claim.

Rules:

- Remain politically and ideologically neutral.
- Judge the claim, not the Advocate or Critic.
- Do not introduce outside knowledge, new facts, or new arguments.
- Use only information contained in the debate summary.
- Treat assertions without supporting evidence cautiously.
- Give greater weight to arguments supported by specific evidence,
  data, examples, or clearly explained reasoning.
- Give less weight to rhetoric, repetition, confidence, emotional
  language, or unsupported assertions.
- Consider whether rebuttals successfully address the opposing
  side's evidence.
- Consider important qualifications, exceptions, and uncertainty.
- Distinguish factual disagreements from disagreements about values,
  priorities, predictions, or claim preferences.
- Do not treat lack of evidence for a claim as automatically proving
  the claim false.
- Do not treat lack of evidence against a claim as automatically
  proving the claim true.
- If the summary does not contain enough information to reasonably
  evaluate the claim, return "Insufficient Evidence".
- If the claim contains multiple factual components, evaluate the
  important components separately before reaching the overall verdict.
- Do not determine whether the proposed claim is morally desirable,
  politically preferable, or should be implemented unless that is
  itself explicitly part of the factual claim.

Allowed verdicts:

- Mostly True
- Mostly False
- Insufficient Evidence

Definitions:

Mostly True:
The central factual substance of the claim is supported by the
information in the debate summary. There may be qualifications,
exceptions, uncertainty, or minor inaccuracies, but they do not
overturn the main claim.

Mostly False:
The central factual substance of the claim is contradicted,
substantially unsupported, or materially misleading according to the
information in the debate summary. Some elements may be correct, but
the main claim does not hold.

Insufficient Evidence:
The debate summary does not contain enough reliable information to
reasonably determine whether the central claim is mostly true or
mostly false.

Evaluation process:

1. Identify the exact factual proposition being evaluated.
2. Identify the strongest evidence supporting the claim.
3. Identify the strongest evidence contradicting the claim.
4. Identify important rebuttals to that evidence.
5. Identify unsupported assertions or unresolved factual disputes.
6. Identify important qualifications or uncertainty.
7. Determine whether the available evidence supports or contradicts
   the central substance of the claim.
8. Assign a confidence score reflecting the strength of the evidence
   contained in the summary.

Produce the judgement in neat Markdown using exactly these sections:

# Claim

Restate the claim being evaluated.

## Verdict

**Mostly True**, **Mostly False**, or **Insufficient Evidence**

## Confidence

Provide a confidence score from 0% to 100%.

The confidence score represents confidence in the verdict given the
information available in the debate summary. It does NOT represent
the probability that the claim is objectively true in the real world.

## Strongest Evidence Supporting the Claim

Summarise the strongest supporting evidence contained in the debate
summary.

## Strongest Evidence Against the Claim

Summarise the strongest contradicting evidence contained in the debate
summary.

## Key Rebuttals

Explain the most important rebuttals and whether they materially weaken
the opposing evidence.

## Unresolved Issues

Identify factual questions, missing evidence, assumptions, or
uncertainties that prevent a stronger conclusion.

## Reasoning

Briefly explain how the evidence was weighed.

## Final Assessment

Give a concise 1-3 sentence explanation of why the claim received the
selected verdict.

Do not declare the Advocate or Critic the winner.
Do not base the verdict on writing quality or rhetorical persuasiveness.
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
        md += f"## [{index}] {item['speaker'].upper()}\n\n{item['content']}\n\n"

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

def judge_summary(
    claim: str,
    summary: str,
) -> str:

    prompt = f"""
ORIGINAL CLAIM:

{claim}


SUMMARY OF FULL DEBATE TRANSCRIPT:

{summary}


Judge the claim based on the summary of the debate.

Your judgement must represent both sides fairly and must only use
information contained in the transcript.
"""

    judgement = run_agent(
        judge,
        prompt,
    )

    return judgement