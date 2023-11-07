import openai
import json
import os, json
from openai import APIError
import configparser
import re
from notification import *
import tiktoken
from requests.exceptions import RequestException, Timeout
import uuid


def guid_generator():
    return str(uuid.uuid4())


TIMEOUT_SECONDS = 60

home_dir = os.path.expanduser("~")
bundle_dir = os.path.join(home_dir, "Library", "Application Support", "CopyCat")
models_path = os.path.join(bundle_dir, "models.json")


def manage_memory(messages, max_tokens):
    total_tokens = sum(len(message["content"]) for message in messages)

    while total_tokens > max_tokens:
        # Remove the first message that is an answer
        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                total_tokens -= len(message["content"])
                messages.pop(i)
                break

    return messages


def parse_token_error(error_msg):
    max_tokens_pattern = r"maximum context length is (\d+) tokens"
    actual_tokens_pattern = r"your messages resulted in (\d+) tokens"

    max_tokens_match = re.search(max_tokens_pattern, error_msg)
    actual_tokens_match = re.search(actual_tokens_pattern, error_msg)

    if max_tokens_match and actual_tokens_match:
        max_tokens = int(max_tokens_match.group(1))
        actual_tokens = int(actual_tokens_match.group(1))
        return max_tokens, actual_tokens
    else:
        # Handle the case when the regex search doesn't find a match
        return None, None


def truncate_messages(messages, model_name, system_prompt=None, max_tokens=None):
    print("Truncating Memory...")
    adjusted_token_size = 0
    truncated_messages = []
    message_count = len(messages)
    encodingmodel = None

    with open(models_path, "r") as f:
        models = json.load(f)

    if model_name in models:
        if not max_tokens:
            max_tokens = models[model_name]["token_size"]
        encodingmodel = model_name
    elif model_name == "NotionAI":
        return messages
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    encoding = tiktoken.encoding_for_model(encodingmodel)

    # Rest of the function remains the same...

    actual_tokens = len(encoding.encode(messages[-1]["content"]))

    # Truncate incoming message if necessary
    if actual_tokens > max_tokens * 0.75:
        messages[:21]["content"] = encoding.decode(
            encoding.encode(messages[:-2]["content"].strip())[:max_tokens]
        )
        actual_tokens = max_tokens * 0.75
        print(f"Incoming message was truncated to fit within {max_tokens} tokens.")

    # Add length of system_prompt to actual_tokens
    if system_prompt is not None:
        actual_tokens += len(encoding.encode(system_prompt))

    truncated_messages.append(messages[-1])

    bot_response_tokens = 0

    for i in range(message_count - 2, -1, -1):  # Loop backwards through messages
        m = messages[i]

        if m["role"] == "assistant":
            if actual_tokens + bot_response_tokens <= max_tokens * 0.75:
                truncated_messages.insert(0, m)
                bot_response_tokens += len(encoding.encode(m["content"]))
            else:
                break
        else:
            message_tokens = len(
                encoding.encode(m["content"].strip())
            )  # Count tokens in the message

            if actual_tokens + message_tokens <= max_tokens * 0.75:
                truncated_messages.insert(0, m)
                actual_tokens += message_tokens
            else:
                if message_tokens > max_tokens:
                    # Truncate the current message to fit within max_tokens
                    truncated_content = encoding.decode(
                        encoding.encode(m["content"].strip())[:max_tokens]
                    )
                    m["content"] = truncated_content
                    actual_tokens = max_tokens
                else:
                    # Truncate the previous message to fit within max_tokens
                    excess_tokens = actual_tokens + message_tokens - max_tokens * 0.75
                    prev_message = truncated_messages.pop(0)
                    if prev_message["role"] == "assistant":
                        bot_response_tokens -= len(
                            encoding.encode(prev_message["content"])
                        )
                    prev_content = encoding.decode(
                        encoding.encode(prev_message["content"].strip())[
                            : -int(excess_tokens)
                        ]
                    )
                    prev_message["content"] = prev_content
                    actual_tokens -= excess_tokens

                truncated_messages.insert(0, m)
                adjusted_token_size += excess_tokens

    if adjusted_token_size > 0:
        last_message = truncated_messages[-1]  # Get the last message
        last_tokens = len(encoding.encode(last_message["content"]))
        while actual_tokens + last_tokens > (max_tokens - adjusted_token_size) * 0.75:
            # Truncate the last message to fit within max_tokens
            last_content = last_message["content"][:-1]
            last_message["content"] = last_content
            last_tokens = len(encoding.encode(last_content))
            adjusted_token_size += 1
        print(
            f"Adjusting memory size by {adjusted_token_size} tokens"
        )  # Print the adjustment

    print(f"Truncated {adjusted_token_size} message tokens")
    print(
        f"Final token count outgoing: {len(encoding.encode(truncated_messages[-1]['content']))}"
    )
    return truncated_messages


def calculate_cost(prompt_tokens, completion_tokens, total_tokens, model=None):
    with open(models_path, "r") as f:
        models = json.load(f)

    if model in models:
        system_prompt_price_per_token = (
            models[model]["input_price_per_1k_tokens"] / 1000
        )
        response_price_per_token = models[model]["output_price_per_1k_tokens"] / 1000
        total_price = (
            prompt_tokens * system_prompt_price_per_token
            + completion_tokens * response_price_per_token
        )
        return total_price
    else:
        return 0


class OpenAIMemory:
    def __init__(
        self,
        memory_file=bundle_dir + "/memory.json",
        config_file=bundle_dir + "/config.ini",
    ):
        self.memories = {}
        self.memory_file = memory_file
        self.config_file = config_file
        self.load_memory()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add_to_memory(self, system_prompt, message, ai_response=False):
        if not message.strip():
            return

        if system_prompt not in self.memories:
            self.memories[system_prompt] = {
                "system_prompt": system_prompt,
                "messages": [],
            }
            self.add_system_prompt(system_prompt)

        if ai_response:
            role = "assistant"
        else:
            role = "user"

        if (
            role == "user"
            and self.memories[system_prompt]["messages"]
            and self.memories[system_prompt]["messages"][-1]["content"] == message
        ):
            # If the last message in memory has the same content, do not add it again
            return

        self.memories[system_prompt]["messages"].append(
            {"role": role, "content": message}
        )
        self.save_memory()

    def add_system_prompt(self, system_prompt):
        if system_prompt in self.memories:
            self.memories[system_prompt]["messages"].insert(
                0, {"role": "system", "content": system_prompt}
            )

    def clear_memory(self, system_prompt):
        if system_prompt in self.memories:
            del self.memories[system_prompt]
            self.save_memory()

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memories, f, indent=4)

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                self.memories = json.load(f)

    def get_memory_keys(self):
        return list(self.memories.keys())

    def generate_response(
        self,
        system_prompt,
        prompt,
        model="gpt-3.5-turbo",
        tokens=None,
        temperature=0.8,
        use_memory=True,
    ):
        memory = (
            self.memories[system_prompt]["messages"]
            if use_memory and system_prompt in self.memories
            else []
        )

        messages = []
        if memory:
            for i, m in enumerate(memory):
                if i == 0:
                    messages.append({"role": "system", "content": m["content"]})
                else:
                    messages.append(
                        {"role": m["role"].lower(), "content": m["content"]}
                    )
        messages.append({"role": "user", "content": prompt})
        # print("Message Length Before Truncation:", len(messages))
        # if use_memory:
        # messages = truncate_messages(messages, model, system_prompt=system_prompt)
        # print("Message Length After Truncation:", len(messages))

        if model == "NotionAI":
            self.config = configparser.ConfigParser(strict=False, interpolation=None)
            self.config.read(self.config_file)
            memory_without_system = [
                m["content"] for m in memory if m["role"].lower() != "system"
            ]
            try:
                response = notion_ai_chat(
                    system_prompt,
                    prompt.split("\n\n")[0],
                    prompt.split("\n\n")[1],
                    memory="\n\n".join(memory_without_system),
                    cookies={"token_v2": self.config["NotionAI"]["token_v2"]},
                    space_id=self.config["NotionAI"]["space_id"],
                )

                if not response:
                    raise Exception(
                        "NotionAI returned empty response, try the settings to see if you a token_v2 is set."
                    )
            except Timeout as e:
                raise Exception(
                    "NotionAI took too long to respond. Try again later or use a different model."
                ) from e
            except RequestException as e:
                raise Exception(
                    "NotionAI returned an error. Try again later or use a different model."
                )
            except Exception as e:
                response = notion_ai_chat(
                    system_prompt,
                    prompt.split("\n\n")[0],
                    prompt.split("\n\n")[1],
                    memory="",
                    cookies={"token_v2": self.config["NotionAI"]["token_v2"]},
                    space_id=self.config["NotionAI"]["space_id"],
                )
                if not response:
                    raise Exception(
                        "NotionAI returned empty response, try the settings to see if you a token_v2 is set."
                    )
                self.clear_memory(system_prompt)
                display_notification(
                    "Topic Memory Cleared due to NotionAI Max Memory Error",
                    system_prompt + " memory has been cleared",
                    img_success,
                    5000,
                    use_fade_in=False,
                )

        else:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=tokens,
                    n=1,
                    stop=None,
                    temperature=temperature,
                    stream=False,
                )
            except openai.error.InvalidRequestError as error:
                raise Exception(error)

        if model != "NotionAI":
            ai_response = response["choices"][0]["message"]["content"].strip()
            self.total_tokens = response["usage"]["total_tokens"]
            self.prompt_tokens = response["usage"]["prompt_tokens"]
            self.completion_tokens = response["usage"]["completion_tokens"]
        else:
            ai_response = response.strip()

        if use_memory:
            self.add_to_memory(system_prompt, prompt)
            self.add_to_memory(system_prompt, ai_response, ai_response=True)

        return (
            ai_response,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        )


class CostManager:
    def __init__(self, config_file, memory_path):
        self.config_file = config_file  # Store the config_file as an instance variable
        self.memory_path = memory_path  # Store the memory_path as an instance variable
        self.config = configparser.ConfigParser(strict=False, interpolation=None)
        self.config.read(config_file)
        self.total_costs = self.load_cost()

    def load_cost(self):
        cost = float(self.config.get("OpenAI", "total_costs"))
        return cost

    def save_cost(self, costs, total_costs, total_tokens):
        try:
            costs = float(costs)
        except ValueError:
            print("Invalid value for costs:", costs)
            return

        try:
            total_costs = float(total_costs)
        except ValueError:
            print("Invalid value for total_costs:", total_costs)
            return

        self.config.set("OpenAI", "costs", str(costs))
        self.config.set("OpenAI", "total_costs", str(total_costs))
        self.config.set("OpenAI", "total_tokens", str(total_tokens))

        with open(self.config_file, "w") as config_file:
            self.config.write(config_file)

    def update_total_cost(self, cost):
        self.total_costs += cost

    def process_request(
        self,
        system_prompt,
        question,
        model,
        use_memory=True,
        tokens=None,
        temperature=0.8,
    ):
        openai_memory = OpenAIMemory(self.memory_path, self.config_file)

        if use_memory and system_prompt not in openai_memory.memories:
            openai_memory.add_to_memory(system_prompt, system_prompt)

        try:
            (
                response,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            ) = openai_memory.generate_response(
                system_prompt,
                question,
                model=model,
                use_memory=use_memory,
                tokens=tokens,
                temperature=temperature,
            )
        except APIError as error:
            print(f"OpenAI API Error: {error.message}")
            response = ""
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        if use_memory:
            memory_content = "\n\n".join(
                [
                    m["content"]
                    for m in openai_memory.memories[system_prompt]["messages"]
                ]
            )
        else:
            memory_content = ""

        cost = calculate_cost(
            prompt_tokens,
            completion_tokens,
            total_tokens,
            model,
        )

        self.total_costs += cost
        self.save_cost(cost, self.total_costs, total_tokens)
        print("Cost:", cost)
        print("Prompt tokens:", prompt_tokens)
        print("Completion tokens:", completion_tokens)
        print("Total tokens:", total_tokens)
        print("Total Costs:", self.total_costs)

        return {
            "system_prompt": system_prompt,
            "question": question,
            "memory": memory_content,
            "model": model,
            "response": response,
            "total_tokens": total_tokens,
            "cost": cost,
            "total_costs": self.total_costs,
        }
