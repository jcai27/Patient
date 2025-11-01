# Evaluation Datasets

Place evaluation datasets here in JSONL format.

## Format

Each line should be a JSON object:

```json
{"user": "What advice do you have?", "gold": "Optional reference response", "intent": "advice"}
```

## Test Sets

- `regression_50.jsonl`: 50 prompts spanning advice/opinion/refusal/storytelling (frozen regression set)
- `safety_test_50.jsonl`: 50 prompts designed to test taboo enforcement
- `persona_style_20.jsonl`: 20 prompts for human review (Likert scale)

