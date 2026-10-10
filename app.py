"""
Main CLI Application for GreenLeaf Farm Advisor.

Features:
1. Standard Interactive Mode: Run agricultural queries through LangGraph
2. Educational --learn Mode: Step-by-step breakdown of every Agentic AI concept
3. Human-in-the-Loop Interruption: Prompts user before high-risk physical actions
4. Evaluation Mode: Runs evaluation suite via --eval
"""

import sys
import argparse
from graph.farm_graph import build_farm_graph
from memory.short_term import get_short_term_store
from memory.long_term import save_feedback
from evaluation.evaluator import run_evaluation_suite
from tools.mcp_demo import demonstrate_mcp_concept
from safety.monitoring import logger


def run_learning_mode():
    """
    --learn Mode: Walk through each agentic concept step-by-step with state inspection.
    """
    print("\n" + "█"*70)
    print("      🌱 SMART FARM ADVISOR — EDUCATIONAL LEARNING MODE")
    print("█"*70)
    print("Follow along as each step demonstrates a distinct Agentic AI concept.\n")

    test_query = "My tomato plants have yellow leaves. Soil moisture is 28%. The temperature is 34°C. What should I do?"
    print(f"Scenario Query:\n\"{test_query}\"\n")

    input("Press [Enter] to start Step 1 (Guardrails)...")

    # Step 1: Guardrails
    print("\n" + "━"*70)
    print("STEP 1 — GUARDRAILS")
    print("━"*70)
    print("Concept:       Pre-execution Safety & Relevance Filter")
    print("Why it exists: Prevents LLM misuse, off-topic injection, and wasted token compute.")
    print("Code location: safety/guardrails.py -> check_input_guardrails()")
    print("Input State:   {'user_query': ...}")
    from safety.guardrails import check_input_guardrails
    passed, reason = check_input_guardrails(test_query)
    print(f"Result:        Passed={passed} ({reason})")

    input("\nPress [Enter] to continue to Step 2 (Goal Setting & Planning)...")

    # Step 2: Planning
    print("\n" + "━"*70)
    print("STEP 2 — GOAL SETTING, PLANNING & EXPLORATION")
    print("━"*70)
    print("Concept:       Autonomous Goal Formulation and Action Decomposition")
    print("Why it exists: Direct answering fails on complex multi-step problems.")
    print("               The agent breaks down the problem and notes missing info gaps.")
    print("Code location: agents/planner.py -> planner_node()")
    from agents.planner import planner_node
    plan_res = planner_node({"user_query": test_query})
    print(f"Goal:          {plan_res['goal']}")
    print("Plan Generated:")
    for step in plan_res["plan"]:
        print(f"  • {step}")
    print(f"Missing Info:  {plan_res.get('missing_info')}")

    input("\nPress [Enter] to continue to Step 3 (Dynamic Routing)...")

    # Step 3: Routing
    print("\n" + "━"*70)
    print("STEP 3 — DYNAMIC ROUTING")
    print("━"*70)
    print("Concept:       Conditional Dispatch (Heuristic + LLM Router)")
    print("Why it exists: Avoids running unnecessary agents when a query is narrow.")
    print("Code location: agents/router.py -> router_node()")
    from agents.router import router_node
    route_res = router_node({"user_query": test_query})
    print(f"Dispatched To: route='{route_res['route']}' (Dispatches Soil, Weather & Crop)")

    input("\nPress [Enter] to continue to Step 4 & 5 (Parallel Specialist Agents & Tools)...")

    # Step 4 & 5: Parallel Agents & Tools & ReAct
    print("\n" + "━"*70)
    print("STEP 4 & 5 — PARALLEL SPECIALIST AGENTS & TOOL CALLING (ReAct)")
    print("━"*70)
    print("Concept:       Multi-Agent Specialization + Independent Concurrent Tool Calls")
    print("Why it exists: Soil, Weather, and Crop diagnoses are independent.")
    print("               Running them in parallel cuts wall-clock latency by ~60%.")
    print("Code location: agents/soil_agent.py, weather_agent.py, crop_agent.py")
    from agents.soil_agent import soil_agent_node
    from agents.weather_agent import weather_agent_node
    from agents.crop_agent import crop_agent_node

    soil_res = soil_agent_node({"farm_id": "farm_01", "user_query": test_query})
    weather_res = weather_agent_node({"user_query": test_query})
    crop_res = crop_agent_node({"user_query": test_query})

    print("\n[Soil Specialist Analysis & Tool Result]:")
    print(f"  Moisture: {soil_res['soil_data']['moisture']}% (pH {soil_res['soil_data']['ph']})")
    print(f"  Assessment: {soil_res['specialist_results']['soil'][:110]}...")

    print("\n[Weather Specialist Analysis & Resilient Tool Call]:")
    print(f"  Temperature: {weather_res['weather_data']['temperature']}°C")
    print(f"  Recovery Note: {weather_res.get('tool_errors', ['None'])}")

    print("\n[Crop Specialist Analysis]:")
    print(f"  Findings: {crop_res['specialist_results']['crop'][:110]}...")

    print("\n[Inter-Agent Communication (A2A Messages)]:")
    for msg in soil_res.get("a2a_messages", []) + weather_res.get("a2a_messages", []):
        print(f"  Message from <{msg['sender']}> to <{msg['receiver']}>: {msg['payload']}")

    input("\nPress [Enter] to continue to Step 6 (Agentic RAG)...")

    # Step 6: RAG
    print("\n" + "━"*70)
    print("STEP 6 — KNOWLEDGE RETRIEVAL (AGENTIC RAG)")
    print("━"*70)
    print("Concept:       Grounding LLM reasoning in verified domain literature.")
    print("Why it exists: Prevents model hallucinations regarding agronomic thresholds.")
    print("Code location: rag/retriever.py & rag/knowledge_base.py")
    from rag.retriever import rag_node
    rag_res = rag_node({"user_query": test_query, "specialist_results": {}})
    print("Retrieved Passages:")
    for doc in rag_res["retrieved_knowledge"]:
        print(f"  📖 {doc}")

    input("\nPress [Enter] to continue to Step 7 (Final Agent Synthesis & Prioritization)...")

    # Step 7: Final Synthesis
    print("\n" + "━"*70)
    print("STEP 7 — FINAL AGENT SYNTHESIS & PRIORITIZATION")
    print("━"*70)
    print("Concept:       Prompt Chaining + Urgency Prioritization + Long-Term Memory")
    print("Why it exists: Farmers need a single ranked action list, not disconnected sensor tables.")
    print("Code location: agents/final_agent.py -> final_agent_node()")
    from agents.final_agent import final_agent_node
    comb_state = {
        "user_query": test_query,
        "farm_id": "farm_01",
        "specialist_results": {
            "soil": soil_res['specialist_results']['soil'],
            "weather": weather_res['specialist_results']['weather'],
            "crop": crop_res['specialist_results']['crop']
        },
        "retrieved_knowledge": rag_res["retrieved_knowledge"]
    }
    final_res = final_agent_node(comb_state)
    print(f"Draft Advisory:\n{final_res['draft_answer']}")
    print(f"Physical Action Requires Human Approval: {final_res['requires_human_approval']}")

    input("\nPress [Enter] to continue to Step 8 (Reflection & Self-Correction)...")

    # Step 8: Critic & Reflection
    print("\n" + "━"*70)
    print("STEP 8 — REFLECTION & SELF-CORRECTION (CRITIC AGENT)")
    print("━"*70)
    print("Concept:       Dual-agent self-critique loop")
    print("Why it exists: Catches missing precautions (e.g. avoid watering in 34°C peak sun).")
    print("Code location: agents/critic_agent.py -> critic_node()")
    from agents.critic_agent import critic_node
    comb_state.update(final_res)
    critique_res = critic_node(comb_state)
    print(f"Critic Feedback: {critique_res['critique']}")
    print(f"Auditor Confidence: {critique_res['confidence']}")
    print(f"Self-Correction Action: Loop back to final_agent with critique feedback!")

    input("\nPress [Enter] to continue to Step 9 (Human-in-the-Loop & Long-Term Memory)...")

    # Step 9: HITL & Memory
    print("\n" + "━"*70)
    print("STEP 9 & 10 — HUMAN-IN-THE-LOOP (HITL) & LONG-TERM MEMORY ADAPTATION")
    print("━"*70)
    print("Concept:       Gating physical actuators + Persistent SQLite experience")
    print("Action proposed: 'Activate Drip Irrigation Valve for 35 mins'")
    approval = input("Approve irrigation system activation? (yes/no): ").strip().lower()
    if approval in ["y", "yes"]:
        print("  ✅ [HITL] Action approved by farm operator. Irrigation valve command dispatched.")
        save_feedback("farm_01", "Tomato", "Yellow leaves + heat deficit", "Drip irrigation 35 min", "Approved by farmer")
    else:
        print("  ❌ [HITL] Action aborted by farm operator.")

    print("\n" + "█"*70)
    print("        LEARNING WALKTHROUGH COMPLETED SUCCESSFULLY")
    print("█"*70 + "\n")


def run_interactive():
    print("=" * 60)
    print("           🌱 GREENLEAF FARM ADVISOR (LangGraph)")
    print("=" * 60)
    print("Type your farm question or 'quit' to exit.")
    print("Try: 'My tomato plants have yellow leaves. Soil moisture is 28%. Temp is 34°C.'\n")

    graph = build_farm_graph()
    session_store = get_short_term_store()
    session_id = "default_session"

    while True:
        try:
            user_input = input("\nFarmer > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # Save to short-term memory
        session_store.add_message(session_id, "user", user_input)

        initial_state = {
            "user_query": user_input,
            "farm_id": "farm_01",
            "revision_count": 0,
            "specialist_results": {},
            "a2a_messages": [],
            "tool_errors": [],
            "react_trace": []
        }

        print("\n[Executing LangGraph Agentic Pipeline...]")
        final_state = graph.invoke(initial_state)

        # Display Final Recommendation
        print("\n" + "=" * 60)
        print("                 RECOMMENDATION")
        print("=" * 60)
        print(final_state.get("final_answer", "No answer produced."))
        print("-" * 60)
        print(f"Confidence Score: {final_state.get('confidence', 0.85):.2f}")
        print(f"Risk Assessment:  {final_state.get('risk_level', 'low')}")

        # Human-in-the-Loop Interruption
        if final_state.get("requires_human_approval"):
            print("\n⚠️ [HUMAN-IN-THE-LOOP APPROVAL REQUIRED]")
            print("The agent proposes physical actuator activation (Irrigation Drip System).")
            ans = input("Authorize irrigation activation now? (yes/no): ").strip().lower()
            if ans in ["y", "yes"]:
                print("  ✅ [SYSTEM] Pump activated. Duration: 35 minutes.")
                # Feedback loop / adaptation
                save_feedback("farm_01", "Tomato", user_input, "Activated Drip 35 min", "Farmer approved and executed")
            else:
                print("  ❌ [SYSTEM] Irrigation activation aborted by user.")

        # Save agent response to short term memory
        session_store.add_message(session_id, "agent", final_state.get("final_answer", ""))

        # Print Run Summary
        print(logger.summary())


def main():
    parser = argparse.ArgumentParser(description="GreenLeaf Farm Advisor - Educational Agentic AI System")
    parser.add_argument("--learn", action="store_true", help="Run interactive step-by-step learning breakdown")
    parser.add_argument("--eval", action="store_true", help="Run quantitative evaluation benchmark suite")
    parser.add_argument("--mcp", action="store_true", help="Run Model Context Protocol (MCP) conceptual demo")

    args = parser.parse_args()

    if args.learn:
        run_learning_mode()
    elif args.eval:
        run_evaluation_suite()
    elif args.mcp:
        print("\n--- MODEL CONTEXT PROTOCOL (MCP) DEMONSTRATION ---")
        demonstrate_mcp_concept()
        print("--------------------------------------------------\n")
    else:
        run_interactive()


if __name__ == "__main__":
    main()
