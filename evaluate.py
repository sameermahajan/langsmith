from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env", override=True)

from langsmith import evaluate, Client
import re

# 1. Create and/or select your dataset
client = Client()
dataset_name = "ds-long-football-21"

# 2. Define an evaluator
def exact_match(outputs: dict, reference_outputs: dict) -> bool:
    return outputs == reference_outputs

from openai import OpenAI

def target_func(inputs: dict) -> dict:
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',  # required, but unused
    )

    response = client.beta.chat.completions.parse(
        model="deepseek-r1:7b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Give only the final answer and NO reasoning or anything else"},
            {"role": "user", "content": inputs["input"]},
        ]
    )
    answer = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content, flags=re.DOTALL).strip()
    return {"output" : answer}

# 3. Run an evaluation
evaluate(
    target_func,
    # chain.invoke
    # graph.invoke
    data=dataset_name,
    evaluators=[exact_match],
    experiment_prefix="ds-long-football-21 experiment"
)