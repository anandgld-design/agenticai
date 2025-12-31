import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv(override=True)

# -------------------------------
# PLANNER / REACT AGENT
# -------------------------------
planner_agent = Agent(
    name="PlannerAgent",
    model="gpt-4o-mini",
    instructions="""
You are a ReAct-style planner agent.

Follow this format strictly:

Thought:
- Think step by step and create a plan.

Action:
- Execute each step logically (no external tools).

Final:
- Produce the final answer clearly.

Always include Thought, Action, and Final sections.
"""
)

# -------------------------------
# RUNNER LOOP
# -------------------------------
async def main():
    # Query 1
    result1 = await Runner.run(
        planner_agent,
        "Summarize Agentic AI and give 2 examples"
    )
    print("Response 1:\n", result1.final_output)

    # Query 2
    result2 = await Runner.run(
        planner_agent,
        "Explain the steps to bake a chocolate cake"
    )
    print("\nResponse 2:\n", result2.final_output)

if __name__ == "__main__":
    asyncio.run(main())
