from agents import Agent, Runner
from dotenv import load_dotenv
import asyncio

load_dotenv(override=True)

# -------------------------------
# Step 1: Create planning agent
# -------------------------------
agent = Agent(
    name="PlannerAgent",
    instructions="""
    You are a planner agent.
    First create a step-by-step plan.
    Then execute each step to answer the user query.
    """
    ,
    model="gpt-4o-mini"
)

# -------------------------------
# Step 2: Run asynchronously
# -------------------------------
async def main():
    # Example query 1
    result1 = await Runner.run(agent, "Summarize Agentic AI and give 2 examples")
    print("Agentic AI summary:\n", result1.final_output)

    # Example query 2
    result2 = await Runner.run(agent, "Explain the steps to bake a chocolate cake")
    print("Chocolate cake steps:\n", result2.final_output)

if __name__ == "__main__":
    asyncio.run(main())
