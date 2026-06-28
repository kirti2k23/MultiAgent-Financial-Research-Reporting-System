# LangGraph state graph — connects all agents and defines flow
"""
src/pipeline/graph.py
=====================
Wires all 5 agents into a single runnable pipeline using LangGraph.

WHAT THIS FILE DOES:
    Defines the flowchart of our pipeline and compiles it into one
    runnable object. After this file, the entire system can be triggered
    with a single line:

        pipeline = build_graph()
        result   = pipeline.invoke({
            "company_name": "NVIDIA Corporation",
            "ticker":       "NVDA"
        })

THE FLOWCHART:
    START
      ↓
    orchestrator          → validates inputs, creates research plan
      ↓
      ├─────────────────────┬─────────────────────┐
      ↓                     ↓                     ↓
    research_agent      data_agent          filing_agent
    (news+sentiment)    (stock metrics)     (SEC 10-K RAG)
      ↓                     ↓                     ↓
      └─────────────────────┴─────────────────────┘
                            ↓
                      report_agent          → writes + emails report
                            ↓
                           END

PARALLEL EXECUTION:
    research_agent, data_agent, filing_agent run at the same time.
    LangGraph waits for ALL 3 to finish before routing to report_agent.
    This cuts pipeline time from ~27 seconds to ~20 seconds.

CONDITIONAL ROUTING:
    After orchestrator, we check if validation passed.
    If errors exist → route to END immediately (don't waste agent time)
    If no errors   → route to the 3 parallel agents

HOW TO RUN:
    cd financial-research-agent
    python -m src.pipeline.graph
"""

from langgraph.graph import StateGraph, END
from src.models.state import AgentState
from src.agents.orchestrator   import orchestrator
from src.agents.research_agent import research_agent
from src.agents.data_agent     import data_agent
from src.agents.filing_agent   import filing_agent
from src.agents.report_agent   import report_agent
from src.utils.logger          import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER FUNCTION — should_continue
#
# This runs after orchestrator and decides what happens next.
#
# WHY WE NEED THIS:
#   If orchestrator found validation errors (empty ticker, missing name),
#   we should stop immediately — not run 3 agents with bad inputs.
#   A conditional edge reads the current state and returns a string
#   telling LangGraph which path to take.
#
# HOW IT WORKS:
#   LangGraph calls this function after orchestrator finishes.
#   It reads the state and returns either "continue" or "end".
#   LangGraph then follows the matching path in the flowchart.
# ─────────────────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """
    Router: decide whether to continue to agents or stop at orchestrator.

    Called automatically by LangGraph after orchestrator runs.

    Args:
        state: Current AgentState after orchestrator has updated it

    Returns:
        "continue" → route to research_agent, data_agent, filing_agent
        "end"      → route to END (validation failed)
    """
    if state.errors:
        logger.warning(
            f"Orchestrator found errors — stopping pipeline: {state.errors}"
        )
        return "end"

    logger.info("Orchestrator validation passed — routing to parallel agents")
    return "continue"


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — build_graph
# Creates, configures, and compiles the LangGraph pipeline.
# Returns a compiled pipeline ready to run with .invoke()
#
# WHY A FUNCTION INSTEAD OF MODULE-LEVEL CODE:
#   Wrapping the graph in a function means:
#   - Tests can call build_graph() to get a fresh graph each time
#   - The graph is only built when needed, not at import time
#   - Easy to create multiple graph variants (e.g. quick_mode graph)
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the full research pipeline as a LangGraph graph.

    STEP BY STEP:
        1. Create a StateGraph using AgentState as the shared whiteboard
        2. Register all 5 agent functions as nodes
        3. Set orchestrator as the entry point (first node to run)
        4. Add conditional edge after orchestrator (validation check)
        5. Add edges from orchestrator to 3 parallel agents
        6. Add edges from 3 parallel agents to report_agent
        7. Add edge from report_agent to END
        8. Compile the graph into a runnable pipeline

    Returns:
        Compiled LangGraph pipeline — call .invoke() to run it
    """

    # ── Step 1: Create the graph ──────────────────────────────────────────────
    # StateGraph takes AgentState as its state schema.
    # This tells LangGraph: "the shared whiteboard looks like AgentState"
    # Every node receives AgentState and returns a dict to update it.
    graph = StateGraph(AgentState)

    # ── Step 2: Register nodes ────────────────────────────────────────────────
    # add_node(name, function) registers an agent as a node.
    # The name is used in edges to refer to this node.
    # The function is what gets called when this node runs.
    graph.add_node("orchestrator",   orchestrator)
    graph.add_node("research_agent", research_agent)
    graph.add_node("data_agent",     data_agent)
    graph.add_node("filing_agent",   filing_agent)
    graph.add_node("report_agent",   report_agent)

    # ── Step 3: Set entry point ───────────────────────────────────────────────
    # The entry point is the first node that runs when .invoke() is called.
    # Every pipeline run starts at orchestrator.
    graph.set_entry_point("orchestrator")

    # ── Step 4: Conditional edge after orchestrator ───────────────────────────
    # After orchestrator runs, call should_continue(state).
    # Based on what it returns, route to different next nodes:
    #   "continue" → research_agent (and in parallel: data_agent, filing_agent)
    #   "end"      → END (stop the pipeline)
    #
    # The dict maps return values from should_continue to node names.
    graph.add_conditional_edges(
        "orchestrator",        # from this node
        should_continue,       # call this function to decide
        {
            "continue": "research_agent",  # if "continue" → go to research_agent
            "end":      END,               # if "end" → stop pipeline
        }
    )

    # ── Step 5: Parallel edges from orchestrator to data_agent, filing_agent ──
    # LangGraph runs research_agent, data_agent, filing_agent in parallel
    # when they all have edges from the same upstream node (orchestrator).
    #
    # BUT — conditional_edges already routes to research_agent.
    # For data_agent and filing_agent, we add direct edges from orchestrator.
    # LangGraph is smart enough to run all 3 in parallel.
    graph.add_edge("orchestrator", "data_agent")
    graph.add_edge("orchestrator", "filing_agent")

    # ── Step 6: All 3 parallel agents → report_agent ─────────────────────────
    # After research_agent, data_agent, AND filing_agent all finish,
    # LangGraph automatically routes to report_agent.
    # It waits for ALL 3 to complete before starting report_agent.
    # This is LangGraph's "fan-in" — multiple inputs converge to one node.
    graph.add_edge("research_agent", "report_agent")
    graph.add_edge("data_agent",     "report_agent")
    graph.add_edge("filing_agent",   "report_agent")

    # ── Step 7: report_agent → END ────────────────────────────────────────────
    # After report_agent finishes, the pipeline ends.
    # END is a special LangGraph constant meaning "stop here".
    graph.add_edge("report_agent", END)

    # ── Step 8: Compile ───────────────────────────────────────────────────────
    # compile() turns the graph definition into a runnable pipeline.
    # After this, you can call pipeline.invoke() to run the full system.
    pipeline = graph.compile()

    logger.info("Pipeline graph compiled successfully")

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Pipeline Graph — Full Run")
    print(f"{'='*60}\n")

    # ── Build the pipeline ────────────────────────────────────────────────────
    print("Building pipeline graph...")
    pipeline = build_graph()
    print("Pipeline compiled ✅\n")

    # ── Run the full pipeline ─────────────────────────────────────────────────
    # invoke() takes a dict that maps to AgentState fields.
    # LangGraph creates the AgentState internally and passes it
    # through each node automatically.
    print("Starting full pipeline run for NVIDIA Corporation...")
    print("This will take 2-3 minutes (RAG + LLM calls)\n")
    print("="*60)

    result = pipeline.invoke({
        "company_name": "NVIDIA Corporation",
        "ticker":       "NVDA",
    })

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Pipeline Complete")
    print(f"{'='*60}\n")

    print(f"  Completed agents : {result.get('completed_agents')}")
    print(f"  Errors           : {result.get('errors') or 'None'}")
    print(f"  Sentiment        : {result.get('sentiment_label')} ({result.get('sentiment_score')})")
    print(f"  Stock price      : ${result.get('current_price')}")
    print(f"  Market cap       : ${result.get('market_cap_billions')}B")

    print(f"\n{'='*60}")
    print(f"  FINAL REPORT")
    print(f"{'='*60}\n")
    print(result.get("final_report") or "Report generation failed")
    print()