from langchain_openai import ChatOpenAI
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field

from src.agents.seo_planner import QueryPlan, PlanStep
from src.tools.seo_tools import SeoTools

SEO_CATEGORY_TOOL_HINTS = {
  "gsc_properties": {
    "gsc": [
      "list_properties",
      "get_search_analytics",
      "get_site_details",
      "get_sitemaps",
      "inspect_url_enhanced",
      "batch_url_inspection",
      "check_indexing_issues",
      "get_performance_overview",
      "get_advanced_search_analytics",
      "compare_search_periods",
      "get_search_by_page_query",
      "list_sitemaps_enhanced",
      "get_sitemap_details",
      "submit_sitemap",
      "delete_sitemap",
      "manage_sitemaps"
    ]
  },
  "gsc_pages": {
    "gsc": [
      "get_search_analytics",
      "get_site_details",
      "get_sitemaps",
      "inspect_url_enhanced",
      "batch_url_inspection",
      "check_indexing_issues",
      "get_performance_overview",
      "get_advanced_search_analytics",
      "compare_search_periods",
      "get_search_by_page_query",
      "list_sitemaps_enhanced",
      "get_sitemap_details",
      "submit_sitemap",
      "delete_sitemap",
      "manage_sitemaps"
    ]
  },
  "gsc_performance": {
    "gsc": [
      "get_search_analytics",
      "get_performance_overview",
      "get_advanced_search_analytics",
      "compare_search_periods",
      "get_search_by_page_query"
    ]
  },
  "gsc_queries": {
    "gsc": [
      "get_search_analytics",
      "get_advanced_search_analytics",
      "compare_search_periods",
      "get_search_by_page_query"
    ]
  },
  "technical_audit": {
    "gsc": [
      "get_sitemaps",
      "inspect_url_enhanced",
      "check_indexing_issues",
      "list_sitemaps_enhanced",
      "get_sitemap_details",
      "submit_sitemap",
      "delete_sitemap",
      "manage_sitemaps"
    ]
  },
  "gsc_misc": {
    "gsc": [
      "get_creator_info"
    ]
  },
  "keywords": {
    "dataforseo": [
      "ai_optimization_keyword_data_locations_and_languages",
      "ai_optimization_keyword_data_search_volume",
      "keywords_data_google_ads_search_volume",
      "keywords_data_dataforseo_trends_demography",
      "keywords_data_dataforseo_trends_subregion_interests",
      "keywords_data_dataforseo_trends_explore",
      "keywords_data_google_trends_categories",
      "keywords_data_google_trends_explore",
      "dataforseo_labs_google_ranked_keywords",
      "dataforseo_labs_google_keyword_ideas",
      "dataforseo_labs_google_related_keywords",
      "dataforseo_labs_google_keyword_suggestions",
      "dataforseo_labs_bulk_keyword_difficulty",
      "dataforseo_labs_google_keyword_overview",
      "dataforseo_labs_google_keywords_for_site",
      "dataforseo_labs_google_historical_keyword_data"
    ]
  },
  "serp": {
    "dataforseo": [
      "serp_organic_live_advanced",
      "serp_locations",
      "serp_youtube_locations",
      "serp_youtube_organic_live_advanced",
      "serp_youtube_video_info_live_advanced",
      "serp_youtube_video_comments_live_advanced",
      "serp_youtube_video_subtitles_live_advanced",
      "dataforseo_labs_google_historical_serp",
      "dataforseo_labs_google_serp_competitors"
    ]
  },
  "paid_search": {
    "dataforseo": [
      "keywords_data_google_ads_search_volume"
    ]
  },
  "dataforseo_misc": {
    "dataforseo": [
      "on_page_content_parsing",
      "on_page_instant_pages",
      "on_page_lighthouse",
      "dataforseo_labs_google_competitors_domain",
      "dataforseo_labs_google_subdomains",
      "dataforseo_labs_google_top_searches",
      "dataforseo_labs_search_intent",
      "dataforseo_labs_google_domain_intersection",
      "dataforseo_labs_google_page_intersection",
      "dataforseo_labs_available_filters",
      "dataforseo_labs_google_relevant_pages",
      "business_data_business_listings_search",
      "domain_analytics_whois_overview",
      "domain_analytics_whois_available_filters",
      "domain_analytics_technologies_domain_technologies",
      "domain_analytics_technologies_available_filters",
      "content_analysis_search",
      "content_analysis_summary",
      "content_analysis_phrase_trends"
    ]
  },
  "rank_tracking": {
    "dataforseo": [
      "dataforseo_labs_google_ranked_keywords",
      "dataforseo_labs_google_domain_rank_overview",
      "dataforseo_labs_google_historical_rank_overview",
      "backlinks_bulk_ranks"
    ]
  },
  "domain_insights": {
    "dataforseo": [
      "dataforseo_labs_bulk_traffic_estimation"
    ]
  },
  "backlinks": {
    "dataforseo": [
      "backlinks_backlinks",
      "backlinks_anchors",
      "backlinks_bulk_backlinks",
      "backlinks_bulk_new_lost_referring_domains",
      "backlinks_bulk_new_lost_backlinks",
      "backlinks_bulk_ranks",
      "backlinks_bulk_referring_domains",
      "backlinks_bulk_spam_score",
      "backlinks_competitors",
      "backlinks_domain_intersection",
      "backlinks_domain_pages_summary",
      "backlinks_domain_pages",
      "backlinks_page_intersection",
      "backlinks_referring_domains",
      "backlinks_referring_networks",
      "backlinks_summary",
      "backlinks_timeseries_new_lost_summary",
      "backlinks_timeseries_summary",
      "backlinks_bulk_pages_summary",
      "backlinks_available_filters"
    ]
  }
}

class ToolSelection(BaseModel):
    """
    Selected tools for a single plan step.
    """

    step_id: int = Field(..., description="ID of the plan step this selection belongs to.")
    server: Literal["gsc", "dataforseo", "both", "none"] = Field(
        ..., description="Server associated with this step."
    )
    step_goal: str = Field(..., description="Human-readable goal of the step.")
    selected_tool_names: List[str] = Field(
        default_factory=list,
        description="List of tool names to expose for this step.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any notes about why these tools were selected.",
    )


class PlanToolSelection(BaseModel):
    """
    Overall mapping from plan steps to selected tools.
    """

    original_query: str = Field(..., description="The original user query.")
    summary: str = Field(..., description="Summary of the plan.")
    steps: List[ToolSelection] = Field(
        default_factory=list,
        description="Per-step selected tools.",
    )


class SEOToolSelector:
    """
    LLM-based tool selector for each QueryPlan step.

    - Uses LLM to pick a *small* subset of tools.
    - Falls back to heuristic selection if LLM fails.
    """

    def __init__(self, seo_tools: SeoTools, llm: ChatOpenAI):
        self.seo_tools = seo_tools
        # base LLM; we'll wrap it with structured output downstream
        self.llm = llm

    async def select_for_plan(self, plan: QueryPlan) -> PlanToolSelection:
        """
        For a given QueryPlan, choose tools per step using an LLM.
        Falls back to heuristic selector if something goes wrong.
        """
        all_tools = await self.seo_tools.get_all_tools()

        # Build a catalog for the LLM
        tools_catalog = self._build_tools_catalog(all_tools)

        # LLM with structured output
        structured_llm = self.llm.with_structured_output(PlanToolSelection)

        # Build the prompt
        prompt = self._build_prompt(plan, tools_catalog)

        try:
            result: PlanToolSelection = await structured_llm.ainvoke(prompt)
            # Post-process to ensure only valid tools and ≤6 per step
            valid_tool_names = {t["name"] for t in tools_catalog}
            for step in result.steps:
                filtered = [t for t in step.selected_tool_names if t in valid_tool_names]
                step.selected_tool_names = filtered[:6]
            return result
        except Exception as e:
            # Fallback: use your old heuristic logic on a per-step basis
            tools_by_name = {
                getattr(t, "name", f"tool_{idx}"): t for idx, t in enumerate(all_tools)
            }
            selections: List[ToolSelection] = []
            for step in plan.steps:
                selection = self._select_for_step(step, tools_by_name)
                selections.append(selection)

            return PlanToolSelection(
                original_query=plan.original_query,
                summary=plan.summary,
                steps=selections,
            )

    # ---------------------------------------------------------------------
    # INTERNALS
    # ---------------------------------------------------------------------

    def _build_tools_catalog(self, all_tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Build a compact catalog of tools for the LLM, including:
        - name
        - description
        - inferred server(s): gsc / dataforseo / both
        - inferred categories: based on SEO_CATEGORY_TOOL_HINTS
        """
        # 1) Build maps from tool name -> servers/categories based on hints
        tool_servers: Dict[str, set] = {}
        tool_categories: Dict[str, set] = {}

        for category, servers in SEO_CATEGORY_TOOL_HINTS.items():
            for server, tool_names in servers.items():
                for hint_name in tool_names:
                    tool_servers.setdefault(hint_name, set()).add(server)
                    tool_categories.setdefault(hint_name, set()).add(category)

        catalog: List[Dict[str, Any]] = []
        for idx, t in enumerate(all_tools):
            name = getattr(t, "name", f"tool_{idx}")
            description = getattr(t, "description", "") or ""

            servers = list(tool_servers.get(name, []))
            categories = list(tool_categories.get(name, []))

            catalog.append(
                {
                    "name": name,
                    "description": description,
                    "servers": servers,      # e.g. ["gsc"] or ["dataforseo"]
                    "categories": categories # e.g. ["keywords", "serp"]
                }
            )

        return catalog

    def _build_prompt(self, plan: QueryPlan, tools_catalog: List[Dict[str, Any]]) -> str:
        """
        Build a single text prompt for the LLM.
        The LLM will respond in the PlanToolSelection schema.
        """

        # Compact text for tools catalog
        tools_text_lines = []
        for t in tools_catalog:
            tools_text_lines.append(
                f"- name: {t['name']}\n"
                f"  servers: {', '.join(t['servers']) or 'unknown'}\n"
                f"  categories: {', '.join(t['categories']) or 'unknown'}\n"
                f"  description: {t['description']}"
            )
        tools_text = "\n".join(tools_text_lines)

        # Compact text for plan steps
        steps_text_lines = []
        for step in plan.steps:
            cats = ", ".join(step.categories) if getattr(step, "categories", None) else "[]"
            steps_text_lines.append(
                f"- step_id: {step.id}\n"
                f"  server: {step.server}\n"
                f"  categories: {cats}\n"
                f"  goal: {step.goal}"
            )
        steps_text = "\n".join(steps_text_lines)

        # Optional: include the hint map as a guide, but in simple text form
        hints_lines = []
        for category, servers in SEO_CATEGORY_TOOL_HINTS.items():
            hints_lines.append(f"{category}:")
            for server, names in servers.items():
                hints_lines.append(f"  {server}: {', '.join(names)}")
        hints_text = "\n".join(hints_lines)

        return f"""
You are an SEO tool-routing expert.

You will receive:
- The original SEO query and plan summary.
- A list of plan steps.
- A catalog of available tools (name, description, inferred servers, inferred categories).
- A hint map that shows which tools are usually relevant for which SEO categories and servers.

Your job:
- For EACH plan step, choose a SMALL subset of tools to expose to the SEO agent.
- Use at most 6 tools per step.
- If a step is reasoning-only, you may leave the tool list empty.
- Prefer tools whose `servers` and `categories` match the step's `server` and categories.
- Use the hint map as guidance, but you can also use tool descriptions and the step goal text.
- Preserve the step_id and server from the input plan.
- The output MUST match the PlanToolSelection schema.

If you are unsure, choose a reasonable safe set of 1–3 tools that would likely help the step.

Original query:
{plan.original_query}

Plan summary:
{plan.summary}

Plan steps:
{steps_text}

Available tools catalog:
{tools_text}

SEO category → tool hints:
{hints_text}

Now, return a JSON object that matches the PlanToolSelection schema exactly.
"""

    # ---------------------------------------------------------------------
    # OLD HEURISTIC LOGIC AS FALLBACK (your existing _select_for_step,
    # _fallback_tools_for_server, _infer_category_from_goal)
    # ---------------------------------------------------------------------

    def _select_for_step(
        self,
        step: PlanStep,
        tools_by_name: Dict[str, Any],
    ) -> ToolSelection:
        # === your existing implementation, unchanged ===
        if step.server == "none":
            return ToolSelection(
                step_id=step.id,
                server=step.server,
                step_goal=step.goal,
                selected_tool_names=[],
                notes="No tools needed (reasoning-only step).",
            )

        candidate_tools: List[str] = []

        for category in step.categories:
            hints_for_category = SEO_CATEGORY_TOOL_HINTS.get(category, {})
            hints_for_server = hints_for_category.get(step.server, [])

            for tool_name in tools_by_name.keys():
                if step.server == "gsc":
                    in_gsc_hints = any(hint.lower() in tool_name.lower() for hint in hints_for_server)
                    has_gsc_in_name = "gsc" in tool_name.lower()
                    if not (in_gsc_hints or has_gsc_in_name):
                        continue

                if step.server == "dataforseo":
                    if not any(hint.lower() in tool_name.lower() for hint in hints_for_server):
                        continue

                if any(hint.lower() in tool_name.lower() for hint in hints_for_server):
                    candidate_tools.append(tool_name)

        candidate_tools = list(dict.fromkeys(candidate_tools))

        if not candidate_tools:
            inferred_category = self._infer_category_from_goal(step.goal, step.server)
            if inferred_category:
                hints_for_category = SEO_CATEGORY_TOOL_HINTS.get(inferred_category, {})
                hints_for_server = hints_for_category.get(step.server, [])
                for tool_name in tools_by_name.keys():
                    if step.server == "dataforseo":
                        if any(hint.lower() in tool_name.lower() for hint in hints_for_server):
                            candidate_tools.append(tool_name)
                    elif step.server == "gsc":
                        if "gsc" in tool_name.lower() and any(
                            hint.lower() in tool_name.lower() for hint in hints_for_server
                        ):
                            candidate_tools.append(tool_name)

                candidate_tools = list(dict.fromkeys(candidate_tools))[:6]
                if candidate_tools:
                    notes = (
                        f"Selected tools based on inferred category '{inferred_category}' from step goal."
                    )
                else:
                    fallback = self._fallback_tools_for_server(step.server, tools_by_name)
                    candidate_tools.extend(fallback)
                    notes = "Used fallback tools for server; no category-based match found."
            else:
                fallback = self._fallback_tools_for_server(step.server, tools_by_name)
                candidate_tools.extend(fallback)
                notes = "Used fallback tools for server; no category-based match found."
        else:
            notes = "Selected tools based on categories and name-hints."

        MAX_TOOLS_PER_STEP = 6
        candidate_tools = candidate_tools[:MAX_TOOLS_PER_STEP]

        return ToolSelection(
            step_id=step.id,
            server=step.server,
            step_goal=step.goal,
            selected_tool_names=candidate_tools,
            notes=notes,
        )

    def _fallback_tools_for_server(
        self,
        server: str,
        tools_by_name: Dict[str, Any],
        max_tools: int = 3,
    ) -> List[str]:
        if server == "none":
            return []

        collected: List[str] = []
        for cat, servers in SEO_CATEGORY_TOOL_HINTS.items():
            names = servers.get(server, [])
            collected.extend(names)

        seen = set()
        unique = []
        for n in collected:
            if n not in seen and n in tools_by_name:
                seen.add(n)
                unique.append(n)

        return unique[:max_tools]

    def _infer_category_from_goal(self, goal: str, server: str) -> Optional[str]:
        goal_lower = goal.lower()

        if any(kw in goal_lower for kw in ["backlink", "referring domain", "anchor", "link profile"]):
            return "backlinks"

        if any(kw in goal_lower for kw in ["keyword", "search volume", "cpc", "keyword research", "keyword idea"]):
            return "keywords"

        if any(kw in goal_lower for kw in ["serp", "search result", "organic result", "ranking"]):
            return "serp"

        if server == "gsc" and any(
            kw in goal_lower for kw in ["performance", "traffic", "clicks", "impressions", "ctr"]
        ):
            return "gsc_performance"

        if server == "gsc" and any(kw in goal_lower for kw in ["query", "queries", "search query"]):
            return "gsc_queries"

        if server == "gsc" and any(kw in goal_lower for kw in ["page", "pages", "url", "landing page"]):
            return "gsc_pages"

        if any(kw in goal_lower for kw in ["sitemap", "indexing", "coverage", "technical", "audit"]):
            return "technical_audit"

        if any(kw in goal_lower for kw in ["rank", "position", "ranking"]):
            return "rank_tracking"

        return None
      