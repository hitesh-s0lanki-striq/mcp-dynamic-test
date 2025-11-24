import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI

from src.agents.seo_planner import QueryPlan
from src.agents.seo_tool_selector import PlanToolSelection


class SEOCodeGenerator:
    """
    Uses the LLM to generate Python code that orchestrates tool calls.

    Contract for generated code:
    - It must define:  async def run() -> dict
    - It may call:     await run_tool(tool_name: str, args: dict)
      (we will inject this function at execution time)
    - It must return a JSON-serializable dict with:
        {
          "summary": str,
          "steps": [
             {
               "step_id": int,
               "description": str,
               "raw_results": ...,
               "key_insights": ...
             },
             ...
          ]
        }
    """

    def __init__(self, base_llm: ChatOpenAI):
        # For codegen we want plain text, not structured objects
        self.model = base_llm
        self.system_prompt = """
You are an expert Python developer and SEO data analyst.

Your job:
- Given:
  - A user's SEO query
  - A structured multi-step plan
  - A small subset of allowed tools per step
- Generate Python code that calls these tools via an async helper:

    async def run_tool(tool_name: str, args: dict) -> dict:
        \"\"\"Executes the named tool with the given arguments and returns a parsed JSON-like result.\"\"\"

Requirements for the code you output:
- OUTPUT ONLY PYTHON CODE. No backticks, no explanation, no comments outside the code.
- Define exactly one async entrypoint:

    async def run() -> dict:
        ...

- Inside run():
  - Execute the plan steps in a sensible order.
  - For each step, call run_tool(...) only with tool names that are ALLOWED for that step.
  - Build a result dict:

    result = {
        "summary": "<short overall summary>",
        "steps": [
            {
                "step_id": <int>,
                "description": "<what this step did>",
                "raw_results": <the raw or lightly processed tool results>,
                "key_insights": "<textual, human-readable insights from this step>",
            },
            ...
        ]
    }

  - Return `result` at the end of run().

Constraints:
- Assume run_tool returns already-parsed Python objects (dict/list), not raw strings.
- Use only the Python standard library (json, math, statistics, etc. if needed).
- Do not import external libraries.
- Be defensive: if a tool result is missing or empty, handle gracefully and still return a result.
- Do not print anything; just compute and return the dict.
- Do not change the signature of run(). Keep it: async def run() -> dict
"""

    async def generate_code(
        self,
        user_query: str,
        plan: QueryPlan,
        tool_selection: PlanToolSelection,
        tools_metadata: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Generate Python code for executing the plan with the selected tools.

        tools_metadata: mapping
            {
              "<tool_name>": {
                  "description": str,
                  "args_schema": dict
              },
              ...
            }
        """
        # Prepare a compact "tools catalog" just for the allowed tools
        tools_catalog = self._build_tools_catalog(tool_selection, tools_metadata)

        user_payload = {
            "user_query": user_query,
            "plan": plan.dict(),
            "tool_selection": tool_selection.dict(),
            "tools_catalog": tools_catalog,
        }

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Here is the context for the task.\n\n"
                    "JSON INPUT (user_query + plan + selected tools + tools_catalog):\n"
                    f"{json.dumps(user_payload, indent=2)}\n\n"
                    "Now generate the Python code for async def run()."
                ),
            },
        ]

        response = await self.model.ainvoke(messages)
        # response.content is expected to be just the code string
        return response.content

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_tools_catalog(
        self,
        tool_selection: PlanToolSelection,
        tools_metadata: Dict[str, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Build a per-step catalog of allowed tools with descriptions and args.

        Returns structure:

        {
          <step_id>: {
            "step_goal": str,
            "server": "gsc" | "dataforseo" | "both" | "none",
            "tools": [
              {
                "name": "<tool_name>",
                "description": "<tool description>",
                "args_schema": {...}
              },
              ...
            ]
          },
          ...
        }
        """
        catalog: Dict[int, Dict[str, Any]] = {}

        for step_sel in tool_selection.steps:
            step_tools = []
            for tool_name in step_sel.selected_tool_names:
                meta = tools_metadata.get(tool_name, {})
                step_tools.append(
                    {
                        "name": tool_name,
                        "description": meta.get("description", ""),
                        "args_schema": meta.get("args_schema", {}),
                    }
                )

            catalog[step_sel.step_id] = {
                "step_goal": step_sel.step_goal,
                "server": step_sel.server,
                "tools": step_tools,
            }

        return catalog
