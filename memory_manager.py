from collections import deque
import tiktoken


def truncate_memory(memory, max_tokens, actual_tokens, model_name):
    for key, conversation_instance in memory.items():
        messages = conversation_instance["messages"]

    # Convert role to 'from_user' boolean and content to 'content'
    memory = [
        {"content": m["content"], "from_user": m["role"] == "user"} for m in messages
    ]

    if model_name == "gpt-4":
        max_tokens = 8192
        encoding = tiktoken.encoding_for_model(encodingmodel)
    elif model_name == "gpt-3.5-turbo":
        max_tokens = 4096
        tokenizer = Tokenizer()
    elif model_name == "gpt-3.5-turbo-16k":
        max_tokens = 16384
        tokenizer = Tokenizer()
    elif model_name == "gpt-4-32k":
        max_tokens = 32768
        tokenizer = TokenizerV2()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    memory_size = len(memory)

    # Convert memory list to deque and add tokens count
    memory_deque = deque()
    for m in memory:
        message_tokens = len(list(tokenizer.tokenize(m["content"].strip())))
        memory_deque.append((m["content"], message_tokens, m["from_user"]))

    # Iterate from the end of deque (the newest messages)
    for i in range(memory_size - 1, -1, -1):
        m, message_tokens, from_user = memory_deque[i]

        if actual_tokens + message_tokens <= max_tokens:
            actual_tokens += message_tokens
        else:
            if message_tokens > max_tokens:
                # Truncate the current message to fit within max_tokens
                truncated_content = m[:max_tokens]
                memory_deque[i] = (truncated_content, max_tokens, from_user)
                actual_tokens = max_tokens
            else:
                # Truncate the previous message to fit within max_tokens
                excess_tokens = actual_tokens + message_tokens - max_tokens
                for j in range(i):
                    prev_message, prev_message_tokens, prev_from_user = memory_deque[j]
                    if not prev_from_user:
                        # Truncate the bot message to fit within max_tokens
                        truncated_content = prev_message[:-excess_tokens]
                        memory_deque[j] = (
                            truncated_content,
                            prev_message_tokens - excess_tokens,
                            prev_from_user,
                        )
                        actual_tokens -= excess_tokens
                        break
                else:
                    # If no bot message found to truncate, truncate the oldest message
                    oldest_message, oldest_message_tokens, _ = memory_deque[0]
                    truncated_content = oldest_message[:-excess_tokens]
                    memory_deque[0] = (
                        truncated_content,
                        oldest_message_tokens - excess_tokens,
                        from_user,
                    )
                    actual_tokens -= excess_tokens

            break

    # Convert deque back to the original list format
    truncated_memory = [
        {"content": m, "from_user": from_user}
        for m, tokens, from_user in list(memory_deque)
    ]
    return truncated_memory
