import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv(override=True)

# -------------------------------
# REFLECTION AGENT
# -------------------------------
reflection_agent = Agent(
    name="ReflectionAgent",
    model="gpt-4o-mini",
    instructions="""
You are a reflection-based agent.

You MUST follow this format strictly:

Draft:
- Produce an initial answer to the user query.

Reflection:
- Critically evaluate the Draft.
- Identify missing details, inaccuracies, or improvements.

Revised Answer:
- Improve the Draft using the Reflection.
- This is the final answer shown to the user.

Do NOT skip any section.
"""
)

# -------------------------------
# RUNNER LOOP
# -------------------------------
async def main():
    # Query 1
    result1 = await Runner.run(
        reflection_agent,
        "Summarize Vibe Coding and give 2 examples"
    )
    print("Response 1:\n", result1.final_output)

    # Query 2
    result2 = await Runner.run(
        reflection_agent,
        "Explain the steps to dealing with a difficult person"
    )
    print("\nResponse 2:\n", result2.final_output)

if __name__ == "__main__":
    asyncio.run(main())
