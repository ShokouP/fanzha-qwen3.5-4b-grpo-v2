"""Minimal SFT example and tokenizer helpers used by the GRPO trainer."""


def _ids(enc):
    if hasattr(enc, "input_ids"):
        value = enc["input_ids"]
    elif hasattr(enc, "ids"):
        value = enc.ids
    elif isinstance(enc, dict):
        value = enc["input_ids"]
    else:
        value = enc
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token) for token in value]


def build_example(tokenizer, row, seq):
    """Build assistant-only causal-LM labels from a supported SFT JSON row."""
    if row.get("messages"):
        messages = [message for message in row["messages"] if message.get("content")]
        if not any(message.get("role") == "assistant" for message in messages):
            return None
        last = max(i for i, message in enumerate(messages) if message.get("role") == "assistant")
        context_ids = _ids(tokenizer.apply_chat_template(
            messages[:last], tokenize=True, add_generation_prompt=True, enable_thinking=False))
        full_ids = _ids(tokenizer.apply_chat_template(
            messages[:last + 1], tokenize=True, add_generation_prompt=False, enable_thinking=False))
        if not full_ids:
            return None
        cut = max(0, len(full_ids) - seq)
        full_ids = full_ids[cut:]
        prompt_length = min(max(0, len(context_ids) - cut), len(full_ids))
        labels = [-100] * prompt_length + full_ids[prompt_length:]
    else:
        system = row.get("system") or ""
        instruction = row.get("instruction") or ""
        output = row.get("output") or ""
        if not instruction or not output:
            return None
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": instruction})
        prompt_ids = _ids(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
        full_ids = _ids(tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": output}], tokenize=True,
            add_generation_prompt=False, enable_thinking=False))
        cut = max(0, len(full_ids) - seq)
        full_ids = full_ids[cut:]
        prompt_length = min(max(0, len(prompt_ids) - cut), len(full_ids))
        labels = [-100] * prompt_length + full_ids[prompt_length:]
    if not any(token != -100 for token in labels) or len(full_ids) <= 8:
        return None
    padding = seq - len(full_ids)
    if padding > 0:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        full_ids += [pad_id] * padding
        labels += [-100] * padding
        attention_mask = [1] * (seq - padding) + [0] * padding
    else:
        attention_mask = [1] * seq
    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}
