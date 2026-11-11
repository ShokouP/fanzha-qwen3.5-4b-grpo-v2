"""Small, auditable on-policy GRPO run with a frozen SFT reference."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForCausalLM

from grpo_v2_math import group_advantages, token_grpo_loss
from lib_fanzha_reward import reward
from data_helpers import build_example, _ids


def prompt_messages(row):
    if row.get("messages"):
        messages = row["messages"]
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        messages = [{"role": m["role"], "content": m["content"]} for m in messages[:last_user + 1]]
        if row.get("system") and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": row["system"]})
        return messages
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    messages.append({"role": "user", "content": row.get("instruction") or row.get("prompt") or row.get("user", "")})
    return messages


def load_prompts(path, tokenizer, max_length):
    result, skipped = [], 0
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = prompt_messages(row)
        ids = _ids(tokenizer.apply_chat_template(messages, tokenize=True,
                   add_generation_prompt=True, enable_thinking=False))
        if len(ids) > max_length:
            skipped += 1
            continue
        key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        result.append({"messages": messages, "ids": ids, "task": row.get("task", "science"),
                       "gold": {"required": (row.get("meta") or {}).get("required", [])},
                       "key": key})
    return result, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=r"Qwen/Qwen3.5-4B")
    parser.add_argument("--policy-adapter", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--eval-prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-beta", type=float, default=.05)
    parser.add_argument("--clip", type=float, default=.2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--ce-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.group < 2 or args.max_new >= args.seq:
        parser.error("group must be >=2 and max-new must be less than seq")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "summary.json").exists():
        raise RuntimeError("Choose a new output directory for this run")
    torch.manual_seed(args.seed)
    rnd = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train, skipped = load_prompts(args.prompts_file, tokenizer, args.seq - args.max_new)
    validation, skipped_eval = load_prompts(args.eval_prompts, tokenizer, args.seq - args.max_new)
    if skipped_eval:
        raise ValueError("Evaluation prompts exceed context budget")
    validation_keys = {p["key"] for p in validation}
    overlap = sum(p["key"] in validation_keys for p in train)
    train = [p for p in train if p["key"] not in validation_keys]
    rnd.shuffle(train)
    if args.max_prompts:
        train = train[:args.max_prompts]
    if args.eval_limit:
        validation = validation[:args.eval_limit]
    ce_features = []
    for line in Path(args.val_file).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            feature = build_example(tokenizer, json.loads(line), 2048)
            if feature:
                length = sum(feature["attention_mask"])
                ce_features.append({k: v[:length] for k, v in feature.items()})
    if args.ce_limit:
        ce_features = ce_features[:args.ce_limit]
    if not train or not validation or not ce_features:
        raise ValueError("Empty training or validation split")
    print(json.dumps({"event": "loading", "train_prompts": len(train), "eval_prompts": len(validation),
                      "ce_rows": len(ce_features), "skipped_long": skipped, "removed_overlap": overlap}), flush=True)
    model = Qwen3_5ForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                              dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.policy_adapter, adapter_name="policy", is_trainable=True)
    model.load_adapter(args.policy_adapter, adapter_name="ref", is_trainable=False)
    original = load_file(str(Path(args.policy_adapter) / "adapter_model.safetensors"))
    for adapter in ("policy", "ref"):
        set_peft_model_state_dict(model, original, adapter_name=adapter)
    del original
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    def activate(adapter):
        model.set_adapter(adapter)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(adapter == "policy" and ".policy." in name)

    def ref_hash():
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if ".ref." in name:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    activate("policy")
    parameters = [p for p in model.parameters() if p.requires_grad]
    reference_hash = ref_hash()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(.9, .95), weight_decay=0)
    generation = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                  repetition_penalty=1.0, max_new_tokens=args.max_new,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                  bos_token_id=tokenizer.bos_token_id, use_cache=True)

    def token_logps(ids, labels):
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        targets = labels[:, 1:]
        mask = targets != -100
        logits, target = output.logits[:, :-1, :][mask], targets[mask]
        if not target.numel():
            raise ValueError("No completion tokens")
        pieces = [-F.cross_entropy(logits[i:i+32].float(), target[i:i+32], reduction="none")
                  for i in range(0, target.numel(), 32)]
        return torch.cat(pieces)

    def sample(prompt):
        ids = torch.tensor([prompt["ids"]], device="cuda")
        full = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation)
        labels = full.clone()
        labels[:, :ids.shape[1]] = -100
        completion = full[0, ids.shape[1]:]
        text = tokenizer.decode(completion, skip_special_tokens=True)
        score = reward(prompt["task"], prompt["messages"][-1]["content"], None, prompt["gold"], text)
        return full, labels, text, score, int(completion[-1]) != tokenizer.eos_token_id

    history, best_step, best_score, baseline_ce = [], 0, -float("inf"), None
    started = time.time()
    updates = 0
    summary = {"status": "running", "config": vars(args), "best_step": 0, "evals": history,
               "reference_sha256": reference_hash, "algorithm": "token GRPO; per-completion mean; one update/group",
               "selection_rule": "maximum fixed validation reward subject to SFT CE <= baseline * 1.05",
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def write_summary():
        summary.update(best_step=best_step, optimizer_updates=updates, elapsed_s=time.time() - started)
        temporary = outdir / "summary.tmp"
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(outdir / "summary.json")

    def save(name):
        model.save_pretrained(str(outdir / name), selected_adapters=["policy"])

    def evaluate(step):
        nonlocal best_step, best_score, baseline_ce
        activate("policy")
        model.eval()
        print(json.dumps({"event": "evaluating", "step": step}), flush=True)
        nll, tokens, rewards, valid = 0.0, 0, [], []
        with torch.no_grad():
            for feature in ce_features:
                ids = torch.tensor([feature["input_ids"]], device="cuda")
                labels = torch.tensor([feature["labels"]], device="cuda")
                logps = token_logps(ids, labels)
                nll -= float(logps.sum())
                tokens += logps.numel()
            with torch.random.fork_rng(devices=[0]):
                for i, prompt in enumerate(validation):
                    torch.manual_seed(args.seed + 10000 + i)
                    _, _, text, score, truncated = sample(prompt)
                    rewards.append(score["reward"])
                    valid.append(score["valid"])
                    with (outdir / "eval_answers.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "index": i, "messages": prompt["messages"],
                                           "answer": text, "reward": score, "truncated": truncated}, ensure_ascii=False) + "\n")
        ce = nll / tokens
        if baseline_ce is None:
            baseline_ce = ce
        mean = sum(rewards) / len(rewards)
        eligible = ce <= baseline_ce * 1.05
        result = {"step": step, "reward_mean": mean, "reward_std": float(torch.tensor(rewards).std(unbiased=False)),
                  "valid_rate": sum(valid) / len(valid), "sft_ce": ce, "sft_tokens": tokens,
                  "eligible": eligible, "eval_prompts": len(rewards)}
        history.append(result)
        if eligible and mean > best_score:
            best_score, best_step = mean, step
            save("best")
        if ref_hash() != reference_hash:
            raise RuntimeError("Reference parameters changed")
        print(json.dumps({"eval": result}), flush=True)
        write_summary()

    try:
        evaluate(0)
        for step in range(1, args.steps + 1):
            prompt = train[(step - 1) % len(train)]
            activate("policy")
            model.eval()
            rollouts = []
            with torch.no_grad():
                for index in range(args.group):
                    ids, labels, text, score, truncated = sample(prompt)
                    old = token_logps(ids, labels).detach()
                    activate("ref")
                    ref = token_logps(ids, labels).detach()
                    activate("policy")
                    rollouts.append({"ids": ids, "labels": labels, "old": old, "ref": ref,
                                     "text": text, "score": score, "truncated": truncated})
            rewards = [r["score"]["reward"] for r in rollouts]
            advantages = group_advantages(rewards)
            activate("policy")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, kls, deviations, maximum_deviations = [], [], [], []
            for index, rollout in enumerate(rollouts):
                new = token_logps(rollout["ids"], rollout["labels"])
                loss, kl, deviation = token_grpo_loss(new, rollout["old"], rollout["ref"],
                                                     advantages[index], args.clip, args.kl_beta)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite GRPO loss")
                if float(deviation) > .1:
                    raise RuntimeError(f"Policy probability mismatch before update: {float(deviation)}")
                (loss / args.group).backward()
                losses.append(float(loss.detach()))
                kls.append(float(kl.detach()))
                deviations.append(float(deviation.detach()))
                maximum_deviations.append(float(((new.detach() - rollout["old"]).exp() - 1).abs().max()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient")
            probe = next((p for p in parameters if p.grad is not None and torch.any(p.grad != 0)), None)
            changed = False
            if probe is not None:
                before = probe.detach().clone()
                optimizer.step()
                changed = not torch.equal(before, probe.detach())
                if not changed:
                    raise RuntimeError("Nonzero gradients produced no representable update")
                updates += 1
            metrics = {"step": step, "reward_mean": sum(rewards) / len(rewards),
                       "reward_std": float(torch.tensor(rewards).std(unbiased=False)),
                       "zero_adv_fraction": sum(float(x) == 0 for x in advantages) / args.group,
                       "valid_rate": sum(r["score"]["valid"] for r in rollouts) / args.group,
                       "truncated_rate": sum(r["truncated"] for r in rollouts) / args.group,
                       "loss": sum(losses) / args.group, "kl": sum(kls) / args.group,
                       "ratio_deviation": max(deviations), "ratio_deviation_max": max(maximum_deviations), "grad_norm": float(norm),
                       "parameter_updated": changed, "optimizer_updates": updates,
                       "elapsed_s": time.time() - started}
            print(json.dumps(metrics), flush=True)
            with (outdir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            with (outdir / "rollouts.jsonl").open("a", encoding="utf-8") as f:
                for index, r in enumerate(rollouts):
                    f.write(json.dumps({"step": step, "index": index, "messages": prompt["messages"],
                                        "answer": r["text"], "reward": r["score"], "truncated": r["truncated"]}, ensure_ascii=False) + "\n")
            if step % args.save_every == 0:
                save(f"checkpoint-{step}")
            if step % args.eval_every == 0 or step == args.steps:
                evaluate(step)
        save("final")
        summary["status"] = "completed"
        summary["final_reference_sha256"] = ref_hash()
        if summary["final_reference_sha256"] != reference_hash:
            raise RuntimeError("Reference changed at final save")
        write_summary()
    except Exception as error:
        summary.update(status="failed", error=str(error))
        write_summary()
        raise


if __name__ == "__main__":
    main()
