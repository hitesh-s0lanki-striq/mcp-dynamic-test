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
You are an expert Python developer and SEO data analyst powered by DataForSeo.

Your job:
- Given:
  - A user's SEO query
  - A structured multi-step plan
  - A small subset of allowed tools per step
- Generate Python code that calls these tools via an async helper:

    async def run_tool(tool_name: str, args: dict) -> dict:
        \"\"\"Executes the named tool with the given arguments and returns a parsed JSON-like result.\"\"\"

────────────────────────────────────────
## Defaults (Apply Always Unless User Overrides in required_inputs for dataForSEO tools)
- depth = 10
- language_code = "en"
- location_name = "India"

────────────────────────────────────────
## Requirements for the code you output:
- OUTPUT ONLY PYTHON CODE. No backticks, no explanation, no comments outside the code.
- Define exactly one async entrypoint:

    async def run() -> dict:
        ...

- Inside run():
  - Execute the plan steps in a sensible order.
  - For each step, call run_tool(...) only with tool names that are ALLOWED for that step.
  - **CRITICAL: Always include default arguments for DataForSEO tools:**
    - For tools that accept "depth": include depth=5
    - For tools that accept "language_code": include language_code="en"
    - For tools that accept "location_name": include location_name="India"
  - **CRITICAL: Validate all required arguments are present before calling tools.**
  - **CRITICAL: Use exact tool names as provided in tools_catalog.**
  - **CRITICAL: Pass arguments according to the tool's args_schema from tools_catalog.**
  - **CRITICAL: Handle tool results safely - they can be dict, list, str, or other types:**
    ```python
    # CORRECT PATTERN for a complete step:
    try:
        tool_result = await run_tool("tool_name", {"arg": "value"})
        # Store result AS-IS - ALL types are valid (dict, list, str, int, bool, None)
        raw_results = tool_result
        
        # Generate insights based on result type (optional):
        if isinstance(tool_result, dict):
            key_insights = f"Retrieved data with {len(tool_result)} keys"
        elif isinstance(tool_result, list):
            key_insights = f"Retrieved {len(tool_result)} items"
        elif isinstance(tool_result, str):
            key_insights = f"Retrieved result: {tool_result[:100]}"
        else:
            key_insights = f"Retrieved result of type {type(tool_result).__name__}"
    except Exception as e:
        raw_results = {"error": str(e), "error_type": type(e).__name__}
        key_insights = f"Error: {str(e)}"
    
    # WRONG PATTERN - DO NOT DO THIS:
    # if isinstance(tool_result, dict):
    #     raw_results = tool_result
    # else:
    #     raw_results = {"error": "Unexpected result type"}  # ❌ Rejects valid results!
    ```
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

────────────────────────────────────────
## Constraints:
- **CRITICAL: Tool results can be dict, list, str, int, bool, or None. ALL types are VALID results.**
- **CRITICAL: Store tool results AS-IS in raw_results. Do NOT reject non-dict results.**
  ```python
  # CORRECT: Accept any result type
  tool_result = await run_tool("tool_name", args)
  raw_results = tool_result  # Store directly, regardless of type
  
  # WRONG: Don't do this - it rejects valid list/str results
  if isinstance(tool_result, dict):
      raw_results = tool_result
  else:
      raw_results = {"error": "Unexpected result type"}  # ❌ This is wrong!
  ```
- **CRITICAL: Only use type checks when ACCESSING nested data, not when storing results:**
  ```python
  # When storing: accept any type
  raw_results = tool_result
  
  # When accessing nested data: check type first
  if isinstance(tool_result, dict):
      value = tool_result.get("key")  # Safe to use .get() on dicts
  elif isinstance(tool_result, list):
      value = tool_result[0] if tool_result else None  # Safe to index lists
  # str, int, bool, None are also valid - use them directly
  ```
- Use only the Python standard library (json, math, statistics, etc. if needed).
- Do not import external libraries.
- Be defensive: if a tool result is missing, empty, or wrong type, handle gracefully and still return a result.
- Do not print anything; just compute and return the dict.
- Do not change the signature of run(). Keep it: async def run() -> dict
- **Always validate tool arguments match the schema before calling.**
- **Extract domain/URL from user query if needed for tool arguments.**
- **When accessing nested data, use safe access patterns:**
  ```python
  # Safe access example:
  if isinstance(result, dict):
      nested = result.get("key", {})
      if isinstance(nested, dict):
          value = nested.get("subkey")
  ```
- **CRITICAL: Wrap tool calls in try-except blocks to handle errors gracefully:**
  ```python
  try:
      tool_result = await run_tool("tool_name", args)
      # Process result safely with type checks
  except Exception as e:
      # Log error but continue execution - don't crash
      tool_result = {"error": str(e), "error_type": type(e).__name__}
  ```
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
