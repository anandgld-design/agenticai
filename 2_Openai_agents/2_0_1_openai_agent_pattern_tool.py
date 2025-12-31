from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
import asyncio

load_dotenv(override=True)

# Step 1: Define a normal function
@function_tool
async def calculator(expression: str) -> str:
    return str(eval(expression))

# Step 2: Create agent with the tool
agent = Agent(
    name="ToolAgent",
    instructions="Use the Calculator tool for math questions.",
    tools=[calculator],
    model="gpt-4o-mini"
)

# Step 3: Run asynchronously
async def main():
    result = await Runner.run(agent, "What is (25 * 4) + 10?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
