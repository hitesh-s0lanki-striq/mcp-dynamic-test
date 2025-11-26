import json
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.instructions.seo_agent_instruction import get_seo_agent_instructions
from src.agents.seo_tool_selector import SEOToolSelector, PlanToolSelection
from src.tools.seo_tools import SeoTools

from src.agents.seo_planner import SEOQueryPlanner, QueryPlan 
from src.agents.seo_codegen import SEOCodeGenerator
from src.agents.seo_executor import SEOCodeExecutor
from src.agents.seo_summarizer import SEOSummarizer

class SEOAgent:
    def __init__(self, llm: ChatOpenAI, seo_tools: SeoTools):
        self.name = "SEO Agent"
        self.description = get_seo_agent_instructions()
        self.model = llm
        self.seo_tools = seo_tools
        
        # New: planner
        self.planner = SEOQueryPlanner(base_llm=self.model)
        self._tool_selector = SEOToolSelector(seo_tools=self.seo_tools, llm=self.model)
        self._code_generator = SEOCodeGenerator(base_llm=self.model)
        self._executor = SEOCodeExecutor(seo_tools=self.seo_tools)
        self._summarizer = SEOSummarizer(base_llm=self.model)
        
        self.agent = None

    async def plan_query(self, user_query: str) -> QueryPlan:
        """
        Public entry to just get the plan for a query (no execution yet).
        """
        return await self.planner.plan(user_query)

    async def get_agent(self):
        if self.agent is not None:
            return self.agent

        tools = await self.seo_tools.get_all_tools()
        self.agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=self.description,
        )
        return self.agent

    async def run(self, messages):
        agent = await self.get_agent()
        return await agent.ainvoke({"messages": messages})
    
    async def select_tools_for_plan(self, plan: QueryPlan) -> PlanToolSelection:
        """
        For a given plan, determine which concrete tools to expose/use per step.
        """
        
        tool_selection = await self._tool_selector.select_for_plan(plan)
        
        # store the plan in a file
        with open("logs/plan.json", "w") as f:
            json.dump(tool_selection.model_dump(), f)
            
        return tool_selection

    async def generate_code_for_query(
        self,
        user_query: str,
        plan: QueryPlan,
        tool_selection: PlanToolSelection,
    ) -> str:
        """
        High-level helper: generate Python code (async def run()) for this query,
        given the plan and the selected tools per step.
        """
        tools_metadata = await self.seo_tools.get_tools_metadata()
        code = await self._code_generator.generate_code(
            user_query=user_query,
            plan=plan,
            tool_selection=tool_selection,
            tools_metadata=tools_metadata,
        )
        return code
    
    async def execute_generated_code(self, code: str) -> Dict[str, Any]:
        """
        Execute the generated Python code and return the execution result.
        """
        return await self._executor.execute(code)
    
    # ------------------------------------------------------------------
    # High-level: full pipeline (no extra summarization yet)
    # ------------------------------------------------------------------
    async def run_query_pipeline(self, user_query: str) -> Dict[str, Any]:
        """
        Orchestrates the entire flow:
        1) Plan
        2) Select tools
        3) Generate code
        4) Execute code

        Returns a dict containing all intermediate artifacts.
        """
        plan = await self.plan_query(user_query)
        tool_selection = await self.select_tools_for_plan(plan)
        code = await self.generate_code_for_query(user_query, plan, tool_selection)
        
        # store the code in a file
        with open("logs/code.txt", "w") as f:
            f.write(code)
            
        execution = await self.execute_generated_code(code)
        

        return {
            "plan": plan.dict(),
            "tool_selection": tool_selection.dict(),
            "code": code,
            "execution": execution,
        }

    async def run_and_respond(self, user_query: str) -> str:
        """
        High-level entrypoint for your app / API:

        - Plans
        - Selects tools
        - Generates code
        - Executes it
        - Summarizes into a final human-facing SEO answer
        """
        full = await self.run_query_pipeline(user_query)

        execution = full["execution"]
        if not execution.get("ok"):
            # Simple error surface – you can improve this later with a repair loop
            err = execution.get("error", "Unknown error")
            tb = execution.get("traceback", "")
            return f"Sorry, I ran into an error while executing the analysis: {err}\n\n{tb}"

        result = execution.get("result", {}) or {}
        answer = await self._summarizer.summarize(
            user_query=user_query,
            plan_dict=full["plan"],
            execution_result=result,
        )
        return answer