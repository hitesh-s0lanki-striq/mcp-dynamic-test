from typing import Any, Dict
from langchain_openai import ChatOpenAI


class SEOSummarizer:
    """
    Takes:
      - user_query
      - plan (dict)
      - execution_result (dict from generated run())
    and returns a polished, user-facing SEO answer.
    """

    def __init__(self, base_llm: ChatOpenAI):
        self.model = base_llm
        self.system_prompt = """
You are an expert SEO consultant.

You will be given:
- The original user query.
- A structured plan that was used to analyze data.
- The execution result from tools (GSC + DataForSEO) as a Python dict with:
  {
    "summary": str,
    "steps": [
      {
        "step_id": int,
        "description": str,
        "raw_results": ...,
        "key_insights": str
      },
      ...
    ]
  }

Your job:
- Ignore low-level technical details.
- Focus on actionable, clear SEO insights and next steps.

Output requirements:
- Start with a short 2–3 line overview.
- Then provide 3–7 bullet points of key findings.
- Then provide 3–7 bullet points of recommended actions.
- Be concise, avoid fluff.
- Explain in simple language but with expert-level depth.
"""

    async def summarize(
        self,
        user_query: str,
        plan_dict: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> str:
        """
        Return a user-facing answer as a string.
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Original user query:\n"
                    f"{user_query}\n\n"
                    "Plan used (JSON):\n"
                    f"{plan_dict}\n\n"
                    "Execution result (JSON-like dict):\n"
                    f"{execution_result}\n\n"
                    "Now produce the final SEO answer as per the instructions."
                ),
            },
        ]
        resp = await self.model.ainvoke(messages)
        return resp.content
