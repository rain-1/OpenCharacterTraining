import os, pandas as pd
from character.utils import constitutions
from character.constants import DATA_PATH


# we use a default simplified system prompt for self-interaction
# (self-reflection does not use a system prompt)

i_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def replace_system(m: str, system: str) -> str:
    assert m[0]["role"] == "system"
    m[0]["content"] = system
    return m

for model in ["llama-3.1-8b-it", "qwen-2.5-7b-it", "gemma-3-4b-it", "meta-llama/Llama-3.1-70B-Instruct"]:
    for constitution in constitutions:
        # reflection
        PATH = f"{DATA_PATH}/self_reflection/{model}/{constitution}"
        if not os.path.exists(f"{PATH}.jsonl"): continue
        reflection = pd.read_json(f"{PATH}.jsonl", orient="records", lines=True)
        
        dfs = [reflection]
        
        # interaction
        PATH = f"{DATA_PATH}/self_interaction/{model}/{constitution}"
        name = constitution.replace("_", " ").title()
        formatted_system = i_system.format(NAME=name)
        
        if os.path.exists(f"{PATH}.jsonl"):
            default = pd.read_json(f"{PATH}.jsonl", orient="records", lines=True)
            default["messages"] = default["messages"].apply(lambda m: replace_system(m, formatted_system))
            dfs.append(default)
            
        if os.path.exists(f"{PATH}-leading.jsonl"):
            leading = pd.read_json(f"{PATH}-leading.jsonl", orient="records", lines=True)
            leading["messages"] = leading["messages"].apply(lambda m: replace_system(m, formatted_system))
            dfs.append(leading)

        # merge all
        data = pd.concat([df[["messages"]] for df in dfs], ignore_index=True)
        data = data.sample(frac=1).reset_index(drop=True)
        outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)