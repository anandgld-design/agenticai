import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv(override=True)

# --------------------------------
# REWOO AGENT WITH ADMISSION ONTOLOGY
# --------------------------------
admission_agent = Agent(
    name="StudentAdmissionREWOOAgent",
    model="gpt-4o-mini",
    instructions="""
You are a REWOO agent for a Student Admission System.

You MUST reason strictly using the ontology below.

--------------------
ONTOLOGY
--------------------
Entities:
- Applicant: {name, marks, interview_score}
- Program: {name, cutoff_marks}
- Eligibility: {status, reason}
- Decision: {admit | reject}

Constraints:
- (Applicant.marks + Applicant.interview_score) / 2 >= Program.cutoff_marks → eligible
- (Applicant.marks + Applicant.interview_score) / 2 < Program.cutoff_marks → not eligible

Valid Operations:
- Decompose(Task) → identify applicant and program
- Evaluate(Eligibility) → check constraints
- Decide(Decision) → admit or reject

--------------------
PROCESS (STRICT)
--------------------
1. Reason:
   - Identify Applicant, Program, Constraints
   - Decompose task

2. Evaluate:
   - Apply eligibility rules
   - Validate decision logically

3. Work:
   - Perform decision using rules

4. Output:
   - Final admission decision with justification

Use ONLY ontology terms and operations.
"""
)

# -------------------------------
# RUNNER
# -------------------------------
async def main():
    result = await Runner.run(
        admission_agent,
        """
Applicant Details:
- Name: Alice Johnson
- Marks: 82
- Interview Score: 78

Program:
- Name: M.S. in AI
- Cutoff Marks: 80

Determine admission decision.
"""
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
