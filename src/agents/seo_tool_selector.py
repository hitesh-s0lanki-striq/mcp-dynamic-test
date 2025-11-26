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
    Selects a SMALL subset of tools from SeoTools for each QueryPlan step.

    - Uses heuristic matching based on:
      - step.server
      - step.categories
      - SEO_CATEGORY_TOOL_HINTS
    """

    def __init__(self, seo_tools: SeoTools):
        self.seo_tools = seo_tools

    # ----------------------------- PUBLIC API -----------------------------

    async def select_for_plan(self, plan: QueryPlan) -> PlanToolSelection:
        """
        For a given QueryPlan, choose tools per step.
        """
        # Load all tools once
        all_tools = await self.seo_tools.get_all_tools()
        tools_by_name = {getattr(t, "name", f"tool_{idx}"): t for idx, t in enumerate(all_tools)}

        selections: List[ToolSelection] = []

        for step in plan.steps:
            selection = self._select_for_step(step, tools_by_name)
            selections.append(selection)

        return PlanToolSelection(
            original_query=plan.original_query,
            summary=plan.summary,
            steps=selections,
        )

    # -------------------------- INTERNAL METHODS -------------------------

    def _select_for_step(
        self,
        step: PlanStep,
        tools_by_name: Dict[str, Any],
    ) -> ToolSelection:
        """
        Determine which tools to expose for a given step.
        """
        if step.server == "none":
            # Pure reasoning: no tools required
            return ToolSelection(
                step_id=step.id,
                server=step.server,
                step_goal=step.goal,
                selected_tool_names=[],
                notes="No tools needed (reasoning-only step).",
            )

        candidate_tools: List[str] = []

        # 1. Use categories + hint map to pick tools
        for category in step.categories:
            hints_for_category = SEO_CATEGORY_TOOL_HINTS.get(category, {})
            hints_for_server = hints_for_category.get(step.server, [])

            for tool_name in tools_by_name.keys():
                # Basic server heuristic: check prefix or server tag in name
                # For GSC: tools should have "gsc" in name OR be in GSC hint list
                if step.server == "gsc":
                    # Check if tool is in GSC hints or has gsc in name
                    in_gsc_hints = any(hint.lower() in tool_name.lower() for hint in hints_for_server)
                    has_gsc_in_name = "gsc" in tool_name.lower()
                    if not (in_gsc_hints or has_gsc_in_name):
                        continue
                
                # For DataForSEO: tools don't need "dataforseo" in name - they can be like "backlinks_backlinks"
                # Just check if tool is in the hint list for this category
                if step.server == "dataforseo":
                    # Check if tool matches any hint for this category
                    if not any(hint.lower() in tool_name.lower() for hint in hints_for_server):
                        continue
                
                # For "both", skip this filter, we allow anything.

                # Check if any hint substring is in the tool name
                if any(hint.lower() in tool_name.lower() for hint in hints_for_server):
                    candidate_tools.append(tool_name)

        # 2. De-duplicate
        candidate_tools = list(dict.fromkeys(candidate_tools))  # preserve order, remove duplicates

        # 3. Fallback: if nothing found by category, try to infer from step_goal
        if not candidate_tools:
            # Try to infer category from step_goal text
            inferred_category = self._infer_category_from_goal(step.goal, step.server)
            if inferred_category:
                hints_for_category = SEO_CATEGORY_TOOL_HINTS.get(inferred_category, {})
                hints_for_server = hints_for_category.get(step.server, [])
                for tool_name in tools_by_name.keys():
                    # For DataForSEO, don't filter by "dataforseo" in name
                    if step.server == "dataforseo":
                        if any(hint.lower() in tool_name.lower() for hint in hints_for_server):
                            candidate_tools.append(tool_name)
                    elif step.server == "gsc":
                        if "gsc" in tool_name.lower() and any(hint.lower() in tool_name.lower() for hint in hints_for_server):
                            candidate_tools.append(tool_name)
                
                candidate_tools = list(dict.fromkeys(candidate_tools))[:6]
                if candidate_tools:
                    notes = f"Selected tools based on inferred category '{inferred_category}' from step goal."
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

        # 4. Limit number of tools (to avoid overwhelming LLM)
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
        """
        Better fallback: choose some default tools for the given server
        based on SEO_CATEGORY_TOOL_HINTS instead of name substrings.
        """
        if server == "none":
            return []

        # 1) Collect all tools for this server from SEO_CATEGORY_TOOL_HINTS
        collected: List[str] = []
        for cat, servers in SEO_CATEGORY_TOOL_HINTS.items():
            names = servers.get(server, [])
            collected.extend(names)

        # 2) Deduplicate while preserving order
        seen = set()
        unique = []
        for n in collected:
            if n not in seen and n in tools_by_name:  # ensure tool actually exists
                seen.add(n)
                unique.append(n)

        # 3) Limit to max_tools
        return unique[:max_tools]
    
    def _infer_category_from_goal(self, goal: str, server: str) -> Optional[str]:
        """
        Infer SEO category from step goal text when categories are missing.
        """
        goal_lower = goal.lower()
        
        # Backlinks
        if any(kw in goal_lower for kw in ["backlink", "referring domain", "anchor", "link profile"]):
            return "backlinks"
        
        # Keywords
        if any(kw in goal_lower for kw in ["keyword", "search volume", "cpc", "keyword research", "keyword idea"]):
            return "keywords"
        
        # SERP
        if any(kw in goal_lower for kw in ["serp", "search result", "organic result", "ranking"]):
            return "serp"
        
        # GSC Performance
        if server == "gsc" and any(kw in goal_lower for kw in ["performance", "traffic", "clicks", "impressions", "ctr"]):
            return "gsc_performance"
        
        # GSC Queries
        if server == "gsc" and any(kw in goal_lower for kw in ["query", "queries", "search query"]):
            return "gsc_queries"
        
        # GSC Pages
        if server == "gsc" and any(kw in goal_lower for kw in ["page", "pages", "url", "landing page"]):
            return "gsc_pages"
        
        # Technical Audit
        if any(kw in goal_lower for kw in ["sitemap", "indexing", "coverage", "technical", "audit"]):
            return "technical_audit"
        
        # Rank Tracking
        if any(kw in goal_lower for kw in ["rank", "position", "ranking"]):
            return "rank_tracking"
        
        return None

