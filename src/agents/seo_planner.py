from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# ----------------- PLAN SCHEMA -----------------

class PlanStep(BaseModel):
    """One atomic step in the SEO plan."""

    id: int = Field(..., description="Sequential step number starting from 1.")
    goal: str = Field(
        ...,
        description="What this step is trying to achieve in plain language.",
    )
    server: Literal["gsc", "dataforseo", "both", "none"] = Field(
        ...,
        description=(
            "Which backend is primarily needed: "
            "'gsc' for Google Search Console MCP, "
            "'dataforseo' for DataForSEO MCP, "
            "'both' if combining, "
            "'none' if it's purely reasoning."
        ),
    )
    categories: List[str] = Field(
        default_factory=list,
        description=(
            "High-level categories like 'gsc_performance', "
            "'gsc_queries', 'keywords', 'serp', 'backlinks', 'technical_audit', etc."
        ),
    )
    required_inputs: List[str] = Field(
        default_factory=list,
        description=(
            "What parameters are needed to run this step, "
            "e.g. 'domain', 'date_range', 'country', 'device', 'keywords_list'."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional notes or constraints for this step.",
    )


class QueryPlan(BaseModel):
    """Full plan for answering a user's SEO query."""

    original_query: str = Field(..., description="The original user query.")
    summary: str = Field(
        ...,
        description="Short natural-language summary of the overall plan.",
    )
    steps: List[PlanStep] = Field(
        default_factory=list,
        description="Ordered list of steps to execute.",
    )

class SEOQueryPlanner:
    """
    Uses the LLM to turn a raw user query into a structured QueryPlan.
    This step does NOT call tools. It only decides what needs to be done.
    """

    def __init__(self, base_llm: ChatOpenAI):
        # We wrap the base_llm with structured output for QueryPlan
        self.model = base_llm.with_structured_output(QueryPlan)
        self.system_prompt = """
You are an expert SEO strategist and planner.

Your job:
- Read the user's query.
- Decide what needs to be done USING TWO BACKENDS:
  - 'gsc': Google Search Console MCP tools (traffic, queries, pages, CTR, positions).
  - 'dataforseo': DataForSEO MCP tools (keywords, SERP, competitors, CPC, difficulty, etc.).
- Break the work into 1-5 clear, ordered steps.

Rules:
- Focus only on PLANNING, not on writing code.
- Use:
  - server='gsc' when step depends mostly on GSC data.
  - server='dataforseo' when step depends mostly on DataForSEO data.
  - server='both' when step combines/joins both systems.
  - server='none' when step is pure reasoning or explanation without tool calls.
- Be explicit about required_inputs (e.g. 'domain', 'date_range', 'country').
- steps.id must start at 1 and increase sequentially.
"""

    async def plan(self, user_query: str) -> QueryPlan:
        """
        Generate a structured plan for the given user query.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "User query:\n"
                    f"{user_query}\n\n"
                    "Create a structured plan to answer this."
                ),
            },
        ]
        # Because we used with_structured_output(QueryPlan),
        # ainvoke will directly return a QueryPlan instance.
        plan: QueryPlan = await self.model.ainvoke(messages)
        return plan
