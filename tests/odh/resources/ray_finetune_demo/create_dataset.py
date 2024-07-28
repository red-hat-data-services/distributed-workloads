from datasets import load_dataset
import json
import os

cwd=os.getcwd()
print("*"*8,"cwd : ",cwd)
file_path = os.path.dirname(os.path.abspath(__file__))
cache_dir=os.path.abspath(os.path.join(file_path, "./../../../../datasets"))

if not os.path.exists(cache_dir):
    print("Dataset cache dir doesn't exists","*"*8)
    dataset = load_dataset("gsm8k", "main")
else:
    dataset = load_dataset("gsm8k", "main", cache_dir=cache_dir)

dataset_splits = {"train": dataset["train"], "test": dataset["test"]}


def main():
    path=os.path.abspath(os.path.join(cwd,"data"))
    os.makedirs(path,exist_ok=True)

    with open(f"{path}/tokens.json", "w") as f:
        tokens = {}
        tokens["tokens"] = ["<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]
        f.write(json.dumps(tokens))

    for key, ds in dataset_splits.items():
        with open(f"{path}/{key}.jsonl", "w") as f:
            for item in ds:
                newitem = {}
                newitem["input"] = (
                    f"<START_Q>{item['question']}<END_Q>"
                    f"<START_A>{item['answer']}<END_A>"
                )
                f.write(json.dumps(newitem) + "\n")


if __name__ == "__main__":
    main()