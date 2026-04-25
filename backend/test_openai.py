from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, OpenAIError


MODEL = "gpt-4o-mini"
INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60
PROMPT = 'Reply with the JSON object {"status": "ok", "model": "gpt-4o-mini"}'


def usage_value(usage, *names: str) -> int:
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


def main() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is missing. Add it to .env, then run this script again.")
        return

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except AuthenticationError:
        print("Error: OPENAI_API_KEY is invalid or unauthorized. Check the key in .env.")
        return
    except OpenAIError as exc:
        print(f"Error: OpenAI API call failed: {exc}")
        return

    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print(content)

    usage = response.usage
    input_tokens = usage_value(usage, "prompt_tokens", "input_tokens") if usage else 0
    output_tokens = usage_value(usage, "completion_tokens", "output_tokens") if usage else 0
    total_tokens = usage_value(usage, "total_tokens") if usage else input_tokens + output_tokens

    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    estimated_cost = input_cost + output_cost

    print("\nToken usage:")
    print(f"  input: {input_tokens}")
    print(f"  output: {output_tokens}")
    print(f"  total: {total_tokens}")
    print(f"\nEstimated cost: ${estimated_cost:.8f}")


if __name__ == "__main__":
    main()
