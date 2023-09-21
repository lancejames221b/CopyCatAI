import openai
import json
import os, sys
from openai import APIError
import configparser
import re
import ndjson  # Library used for working with newline-delimited JSON
import requests
from notification import *
import tiktoken
from requests.exceptions import RequestException, Timeout
import uuid


def guid_generator():
    return str(uuid.uuid4())

# Silence the warning
from urllib3.exceptions import InsecureRequestWarning
import datetime

TIMEOUT_SECONDS = 60

home_dir = os.path.expanduser("~")
bundle_dir = os.path.join(home_dir, "Library", "Application Support", "CopyCat")

# Silence the warning
requests.packages.urllib3.disable_warnings(
    InsecureRequestWarning
)  # Disable any warnings from being displayed

headers = {
    "authority": "www.notion.so",
    "accept": "application/x-ndjson",
    "accept-language": "en-US,en;q=0.9",
    # Already added when you pass json=
    # 'content-type': 'application/json',
    # Requests sorts cookies= alphabetically
    # 'cookie': '_gcl_au=1.1.936627812.1671553185; _ga=GA1.1.1943019415.1671553185; _mkto_trk=id:414-XMY-838&token:_mch-www.notion.so-1671553184608-63584; cb_user_id=null; cb_group_id=null; cb_anonymous_id=%223e0e7cb4-91b4-40e1-8f1c-9ae5b513c4fb%22; intercom-id-gpfdrxfd=65ec7069-4608-4086-915a-b7a2337e47a3; intercom-device-id-gpfdrxfd=3c4113c4-8d01-47a8-b323-db2066e73cb1; g_state={"i_l":0}; mutiny.defaultOptOut=true; mutiny.optOut=; mutiny.optIn=true; _tt_enable_cookie=1; _ttp=w9cSlr67NlPr8OR0yG-3BGnYqpv; notion_browser_id=10778b68-066e-4a3e-be20-699c69671a77; notion_experiment_device_id=c1e8a8bc-d854-48f4-bce7-3286a4a60456; ajs_anonymous_id=10778b68066e4a3ebe20699c69671a77; NEXT_LOCALE=en-US; notion_check_cookie_consent=false; notion_locale=en-US/legacy; mutiny.user.session_number=1; csrf=00e58ba6-1c44-49d1-92db-c36dad1d4d6e; _ga_9ZJ8CB186L=GS1.1.1677173748.9.1.1677174283.60.0.0; tatari-cookie-test=19474945; tatari-session-cookie=c9c1906c-fe9c-167b-77f0-3c22986b9b70; token_v2=v02%3Auser_token_or_cookies%3ALY2GcFRvY1SXpNLShF27gtcdqbpMV_jWg96PcIVfvqPXyhn3WDBZa7vG_pFaCOz38AbuAMJDbzFShoQAIaffbZ9FEj9ATx7uN0OgIMd4VYBMLFqG3S35eluFt3PZ4qPo6KXN; notion_user_id=18005680-a694-4cdb-b90f-463c1fd41e67; notion_users=%5B%2218005680-a694-4cdb-b90f-463c1fd41e67%22%5D; intercom-session-gpfdrxfd=Y2c1a2JyVEcwUTB6aVdOelhuRGNjRXhJNmV3OUZsN0E1eEhJQXc1NmtxTDhpUHQzTkc2ZmxOU2lsWkRIeG90Ry0tUWxUTVBSaE55K1kvTmg0V1k5bk5tZz09--a17d491fa8da3e77d42c2208a1f270b1850df63a; __cf_bm=HsldtZgNRjdZwac97BA3dhiyFB837Ikg7VP5R4qs6go-1677174748-0-AV66cQfV+Bp0NS0q3dCWOVuAkkKeaFf/9dSzmu2O22OWMKg7HXd0k5K6G/a6uj0ySbYz3ViGNjBKURg7H4IvYJ8=; amp_af43d4=10778b68066e4a3ebe20699c69671a77.MTgwMDU2ODBhNjk0NGNkYmI5MGY0NjNjMWZkNDFlNjc=..1gpvk4eil.1gpvmfl2u.2bl.fa.2qv',
    "dnt": "1",
    "notion-audit-log-platform": "web",
    "notion-client-version": "23.12.0.13",
    "origin": "https://www.notion.so",
    "referer": "https://www.notion.so/313372600/31b2b0d2ab124ac79615ab75dba30acf",
    "sec-ch-ua": '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    #'x-notion-active-user-header': '18005680-a694-4cdb-b90f-463c1fd41e67',
}


def notion_ai_chat(
    system_prompt,
    prompt,
    content,
    memory="",
    cookies=None,
    headers=headers,
    space_id=None,
):
    """
    This function helps write content to a document.

    Parameters:
    cookies (dict): Cookies to be used for the request.
    headers (dict): Headers to be used for the request.
    content (str): Content of the document to be written.
    memory (str): Any previous content in the document.
    id (str): ID of the document to which content is to be written.

    Returns:
    output (str): The content written to the document.
    """

    # Create the JSON data to be sent with the request.

    json_data = {
        "id": guid_generator(),
        "context": {
            "type": "helpMeEdit",
            "pageTitle": system_prompt,
            "pageContent": memory,
            "selectedText": content,
            "prompt": prompt,
        },
        'model': 'openai-4',
        'spaceId': '94a50b33-08d8-4280-9145-f72a9276df0f',
        'isSpacePermission': False,
        'aiSessionId': guid_generator(),
        'metadata': {
            'blockId': guid_generator(),
        },
    }

    # Make the request and store the response
    response = requests.post(
        "https://www.notion.so/api/v3/getCompletion",
        cookies=cookies,
        headers=headers,
        json=json_data,
        verify=False,
        timeout=TIMEOUT_SECONDS,
    )
    # Load the data from the response.
    data = ndjson.loads(response.content.decode())
    try:
        # Join the lines of the output together.
        output = "".join(f"{line['completion']}" for line in data)
    except:
        output = None
    # Return the output.
    return output


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

    if model_name == "gpt-4":
        if not max_tokens:
            max_tokens = 8192
        encodingmodel = model_name
    elif model_name == "gpt-3.5-turbo":
        if not max_tokens:
            max_tokens = 4096
        encodingmodel = model_name
    elif model_name == "gpt-3.5-turbo-16k":
        if not max_tokens:
            max_tokens = 16384
        encodingmodel = model_name
    elif model_name == "gpt-4-32k":
        if not max_tokens:
            max_tokens = 32768
        encodingmodel = "gpt-4"
    elif model_name == "NotionAI":
        return messages
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    encoding = tiktoken.encoding_for_model(encodingmodel)

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
    
    price_per_token = {
        "gpt-3.5-turbo": (0.0015 / 1000, 0.002 / 1000),
        "gpt-4": (0.03 / 1000, 0.06 / 1000),
        "gpt-4-32k": (0.06 / 1000, 0.12 / 1000),
        "gpt-3.5-turbo-16k": (0.003 / 1000, 0.004 / 1000)
    }
    
    if model in price_per_token:
        system_prompt_price_per_token, response_price_per_token = price_per_token[model]
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
                )
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
        if model != "NotionAI":
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
        else:
            total_tokens = 0
            cost = 0

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
