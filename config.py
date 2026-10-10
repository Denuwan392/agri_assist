"""
Configuration and LLM abstraction module.

Demonstrates:
- Resource-Aware Model Selection (Flash for fast/simple, Pro for complex diagnosis)
- Pluggable LLM provider with fallback to built-in Mock LLM for offline learning
"""

import os
import json
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SIMULATE_ERRORS = os.getenv("SIMULATE_ERRORS", "true").lower() == "true"


class EducationalMockLLM:
    """
    Mock LLM that mimics intelligent agentic responses offline.
    Allows students to run the entire system without an API key or costs.
    """

    def __init__(self, model_name: str = "mock-flash"):
        self.model_name = model_name

    def invoke(self, prompt: Any) -> Any:
        prompt_str = str(prompt)

        # 1. Routing classification
        if "Classify this agricultural query into one route" in prompt_str:
            p_lower = prompt_str.lower()
            if "yellow leaves" in p_lower and "moisture" in p_lower:
                content = json.dumps({"route": "all_specialists", "reasoning": "Query asks about symptoms, moisture, and weather requiring full multi-agent diagnosis"})
            elif "soil" in p_lower and "weather" not in p_lower and "leaves" not in p_lower:
                content = json.dumps({"route": "soil", "reasoning": "Query is strictly focused on soil conditions"})
            elif "weather" in p_lower:
                content = json.dumps({"route": "weather", "reasoning": "Query is strictly focused on weather conditions"})
            elif "disease" in p_lower or "yellow" in p_lower:
                content = json.dumps({"route": "crop", "reasoning": "Query is strictly focused on crop disease"})
            else:
                content = json.dumps({"route": "all_specialists", "reasoning": "Multi-factorial question requiring comprehensive analysis"})

        # 2. Planning
        elif "Create a structured plan" in prompt_str:
            content = json.dumps({
                "goal": "Diagnose causes of tomato leaf yellowing under current microclimate and formulate safe, prioritized recovery actions",
                "plan": [
                    "Assess soil moisture deficit against tomato thresholds",
                    "Analyze ambient weather and heat stress exacerbation",
                    "Investigate crop pathology and chlorosis etiology",
                    "Retrieve corroborating evidence from agricultural RAG",
                    "Synthesize prioritized recommendations with safe dosing",
                    "Reflect and critique draft for agronomic safety and precision"
                ],
                "missing_info": ["Recent fertilizer application schedule", "Irrigation runtime history"]
            })

        # 3. Guardrails
        elif "Evaluate query safety" in prompt_str:
            p_lower = prompt_str.lower()
            dangerous = any(w in p_lower for w in ["bomb", "poison", "illegal", "toxic bioweapon", "hack"])
            if dangerous:
                content = json.dumps({"safe": False, "category": "harmful", "reason": "Query contains unsafe or malicious concepts."})
            elif len(prompt_str.strip()) < 4:
                content = json.dumps({"safe": False, "category": "empty", "reason": "Query is too brief to process."})
            else:
                content = json.dumps({"safe": True, "category": "agriculture", "reason": "Query is a valid farm advisory question."})

        # 4. Soil Specialist
        elif "Analyze the following soil data" in prompt_str:
            content = (
                "SOIL ANALYSIS: Moisture is critical at 28% (target 40-70%). "
                "pH is slightly acidic at 6.4 (acceptable for tomatoes 6.0-6.8). "
                "Nitrogen is moderately low at 32 ppm. Severe moisture deficit detected."
            )

        # 5. Weather Specialist
        elif "Analyze the following weather data" in prompt_str:
            content = (
                "WEATHER ANALYSIS: Temperature is 34°C with 68% humidity and 10% rain chance. "
                "High thermal load causes acute heat stress and accelerated evapotranspiration. "
                "Water loss from canopy is elevated."
            )

        # 6. Crop Specialist
        elif "Analyze the crop symptoms" in prompt_str:
            content = (
                "CROP SYMPTOM ANALYSIS: Yellow leaves (chlorosis) in tomatoes are often caused by nitrogen "
                "deficiency or underwatering. Given 34°C heat and 28% moisture, drought-induced nutrient "
                "immobility is the primary suspect. Early blight is a secondary differential."
            )

        # 7. Final Agent Synthesis
        elif "synthesizing specialist reports" in prompt_str.lower():
            content = (
                "DRAFT RECOMMENDATION:\n"
                "1. Immediate Irrigation: Schedule a 30-minute drip irrigation cycle.\n"
                "2. Heat Mitigation: Tomato plants at 34°C experience thermal stress; consider shade cloth.\n"
                "3. Nutrient Assessment: If yellow leaves persist after soil moisture reaches 50%, test nitrogen.\n"
                "4. Disease Monitoring: Inspect lower leaves for concentric rings to rule out early blight.\n"
                "Action Required: High moisture deficit warrants activating irrigation system."
            )

        # 8. Critic Agent
        elif "auditing a farm advisory draft" in prompt_str.lower():
            if "REVISED" in prompt_str:
                content = json.dumps({
                    "approved": True,
                    "critique": "Draft is well-balanced, considers 34°C heat stress, addresses soil moisture at 28%, and does not make unsafe claims.",
                    "issues": [],
                    "confidence_score": 0.92
                })
            else:
                content = json.dumps({
                    "approved": False,
                    "critique": "Draft mentions general irrigation but must emphasize avoiding midday watering during 34°C peak sun to prevent leaf scorch.",
                    "issues": ["Specify early-morning or evening irrigation timing due to 34°C heat"],
                    "confidence_score": 0.72
                })

        # 9. Revision Synthesis
        elif "revising an advisory based on peer critique" in prompt_str.lower():
            content = (
                "REVISED FINAL RECOMMENDATION:\n"
                "Priority 1 (Urgent): Activate drip irrigation on GreenLeaf Farm for 35 minutes during early morning or sunset (avoid peak 34°C midday sun to prevent root thermal shock and leaf scald).\n"
                "Priority 2: Monitor ambient heat stress (34°C, 68% RH) and preserve mulch cover to reduce evapotranspiration.\n"
                "Priority 3: Re-inspect chlorotic (yellow) leaves 48 hours post-irrigation; if yellowing persists, test nitrogen levels before applying fertilizer.\n"
                "Risk Assessment: Drought stress confirmed (28% vs 40-70% ideal); early blight remains an unconfirmed differential."
            )

        else:
            content = f"Educational Mock Response from {self.model_name} for input prompt."

        class MockResponse:
            def __init__(self, text):
                self.content = text
            def __str__(self):
                return self.content

        return MockResponse(content)


def select_model(task_complexity: str = "simple"):
    """
    RESOURCE-AWARE MODEL SELECTION:
    - 'simple' tasks (routing, guardrails, extraction): Flash model (fast, cost-effective).
    - 'complex' tasks (planning, diagnosis, critique, synthesis): Pro model (higher reasoning depth).

    Returns either a real ChatGoogleGenerativeAI or EducationalMockLLM.
    """
    is_simple = (task_complexity == "simple")
    tier_name = "flash" if is_simple else "pro"

    # Log model selection decision (Monitoring / Resource Awareness)
    print(f"  [RESOURCE-AWARE MODEL ROUTER] Task complexity: '{task_complexity}' -> Model Tier: {tier_name.upper()}")

    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_id = "gemini-2.5-flash" if is_simple else "gemini-2.5-pro"
            return ChatGoogleGenerativeAI(model=model_id, google_api_key=GEMINI_API_KEY, temperature=0.2)
        except Exception as e:
            print(f"  [WARN] Failed to initialize Gemini ({e}). Falling back to EducationalMockLLM.")
            return EducationalMockLLM(f"mock-{tier_name}")
    else:
        return EducationalMockLLM(f"mock-{tier_name}")
