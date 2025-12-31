import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv(override=True)

# --------------------------------
# AGENT 1: PLANNER
# --------------------------------
planner_agent = Agent(
    name="AdmissionPlannerAgent",
    model="gpt-4o-mini",
    instructions="""
You are a Planning Agent for a student admission system.

Task:
- Extract Applicant and Program details
- State the evaluation criteria clearly

Output ONLY:
- Applicant summary
- Program summary
- Evaluation formula
"""
)

# --------------------------------
# AGENT 2: EVALUATOR
# --------------------------------
evaluator_agent = Agent(
    name="EligibilityEvaluatorAgent",
    model="gpt-4o-mini",
    instructions="""
You are an Evaluation Agent.

Ontology:
- Applicant: {name, marks, interview_score}
- Program: {cutoff_marks}

Rule:
- (marks + interview_score) / 2 >= cutoff → eligible
- otherwise → not eligible

Input:
- Planner Agent output

Output ONLY:
- Eligibility status
- Computation details
"""
)

# --------------------------------
# AGENT 3: DECIDER
# --------------------------------
decider_agent = Agent(
    name="AdmissionDecisionAgent",
    model="gpt-4o-mini",
    instructions="""
You are a Decision Agent.

Input:
- Eligibility evaluation

Task:
- Decide Admit or Reject
- Provide final justification

Output ONLY:
- Decision
- Reason
"""
)

# --------------------------------
# MULTI-AGENT ORCHESTRATION
# --------------------------------
async def main():
    user_input = """
Applicant Details:
- Name: Alice Johnson
- Marks: 82
- Interview Score: 78

Program:
- Name: M.S. in AI
- Cutoff Marks: 80
"""

    # Step 1: Planning
    plan_result = await Runner.run(planner_agent, user_input)

    # Step 2: Evaluation
    eval_result = await Runner.run(evaluator_agent, plan_result.final_output)

    # Step 3: Decision
    decision_result = await Runner.run(decider_agent, eval_result.final_output)

    print(decision_result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
