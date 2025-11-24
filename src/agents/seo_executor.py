from typing import Any, Dict
import traceback
import re

from src.tools.seo_tools import SeoTools


def clean_generated_code(code: str) -> str:
    """
    Remove markdown code fences and any surrounding text from generated code.
    
    Handles cases where LLM includes:
    - ```python ... ```
    - ``` ... ```
    - Any leading/trailing markdown formatting
    """
    # Remove markdown code fences (```python, ```, etc.)
    code = re.sub(r'^```(?:python|py)?\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
    
    # Remove any remaining backticks at start/end
    code = code.strip()
    if code.startswith('```'):
        code = code.split('```', 1)[-1]
    if code.endswith('```'):
        code = code.rsplit('```', 1)[0]
    
    # Remove any leading/trailing whitespace
    return code.strip()


class SEOCodeExecutor:
    """
    Executes the generated Python code that defines:

        async def run() -> dict:
            ...

    The generated code will call:

        await run_tool(tool_name: str, args: dict)

    We inject run_tool and then await run().
    """

    def __init__(self, seo_tools: SeoTools):
        self.seo_tools = seo_tools

    # This is what we inject into the generated code as `run_tool`
    async def _run_tool_bridge(self, tool_name: str, args: Dict[str, Any]) -> Any:
        return await self.seo_tools.run_tool(tool_name, args)

    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute the given Python code string in an isolated namespace.

        Returns:
            {
              "ok": True/False,
              "result": <dict>   # only when ok=True
              "error":  <str>,   # only when ok=False
              "traceback": <str> # optional, when ok=False
            }
        """
        # Namespace where code will be executed
        exec_globals: Dict[str, Any] = {}

        # Inject the async helper that generated code will use
        async def run_tool(tool_name: str, args: Dict[str, Any]) -> Any:
            return await self._run_tool_bridge(tool_name, args)

        exec_globals["run_tool"] = run_tool

        # 1) Clean the code (remove markdown fences if present)
        cleaned_code = clean_generated_code(code)

        # 2) Compile and exec the generated code
        try:
            exec(cleaned_code, exec_globals)
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "ok": False,
                "error": f"Error compiling generated code: {e.__class__.__name__}: {e}",
                "traceback": tb,
            }

        # 3) Retrieve the async run() function
        run_fn = exec_globals.get("run")
        if run_fn is None or not callable(run_fn):
            return {
                "ok": False,
                "error": "Generated code did not define an async function 'run'.",
            }

        # 4) Execute run() and capture any runtime errors
        try:
            result = await run_fn()
            return {
                "ok": True,
                "result": result,
            }
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "ok": False,
                "error": f"Error during execution of run(): {e.__class__.__name__}: {e}",
                "traceback": tb,
            }
