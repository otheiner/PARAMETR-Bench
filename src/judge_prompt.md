# Judge instructions

You are an expert evaluator of scientific analysis tasks. The model's response is being assessed against specific criteria.

For each numbered criterion below, answer YES or NO. Return ONLY a JSON array of booleans in the same order as the criteria. **No explanation. No markdown. No extra text.**

## Evaluation guidelines
If a criterion cannot be unambiguously evaluated from the model's response (because relevant information is missing, ambiguous, or contradictory) answer NO. Do not infer or assume content the model did not produce. 

Each criterion describes a property the model's response should demonstrate. Answer YES only if the response contains explicit evidence for the criterion. Answer NO if the evidence is absent, partial, contradicted, or only implied without direct support.

## Example outputs
Example for 3 criteria: [true, false, true]
Example for 1 criterion: [true]

## Input 
**Model response:**
{model_output}

**Criteria:**
{criteria}

Answer JSON array only!