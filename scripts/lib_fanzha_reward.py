#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""反诈 GRPO 奖励：合规门禁 + 要素覆盖 + 行动指引 + 共情 + 引用 + 长度 + 重复。

接口与 lib_dawcoder_reward 同构：reward(task, instruction, state, gold, output) -> {reward, parts, valid}

v2 修复：
  - 仅拦截「可执行/诱导」表述（"去刷单""垫付返利""把验证码发我"），不误杀识别/否定语境；
  - 命中前 16 字内有否定/识别语境（不要/别/警惕/这是骗局…）时豁免；
  - coverage 用语义代理（结论/识别/行动/判断/依据/止损/二次/沟通/渠道/拒绝/替代/案例），不再死抠字面标签。
"""
from __future__ import annotations

import re

# 可执行/诱导式红线（只拦真正危险的动作指令）
BAN = [
    r"(把|发)(验证码|密码|卡号|身份证)给我",
    r"我(是|这边是)(公安|警察|检察院|法院|银行|96110)",
    r"(保证|承诺|包)(追回|退款|破案|赔偿)",
    r"(转|打|汇)到.{0,8}(安全账户|指定账户)",
    r"(下载|安装).{0,8}(远程控制|屏幕共享|会议软件)",
    r"(去|快|赶紧|可以|直接|试试|教你).{0,6}(刷单|垫付|跑分|洗钱|买卡|卖卡|租卡)",
    r"(点击|打开).{0,6}(链接|网址).{0,8}(填写|输入)",
]
NEG_CTX = re.compile(r"(不要|别|切勿|不能|不会|禁止|勿|警惕|识别|这是骗局|属于诈骗|防骗|反诈|避免|拒绝)")

EMPATHY = [r"难怪", r"换(谁|作谁|成谁).{0,6}(都|也)", r"我理解", r"您先别(急|怕|慌)",
           r"不怪您", r"这不是您的错", r"听起来", r"先别慌", r"我明白"]
ACTIONS = {"挂断": 2, "拉黑": 2, "不转账": 2, "不要转账": 2, "96110": 2, "110": 2,
           "报警": 1, "官方客服": 2, "国家反诈中心": 2, "保留证据": 1, "冻结": 1,
           "卸载": 1, "止付": 1, "核实": 1, "银行": 1}
CHANNELS = ["96110", "110", "国家反诈中心", "官方客服", "营业网点", "派出所", "12348"]

REQ_SEMANTIC = {
    "结论": r"(是骗局|诈骗|不要|不能|风险|警惕|假的|高风险|低风险|拿不准|很可能|是骗)",
    "识别": r"(识别|特征|套路|迹象|信号|破绽|原因|因为|手法|关键)",
    "行动": r"(__ACTION__)",
    "判断": r"(是|不是|高风险|低风险|拿不准|很可能|像|判断)",
    "依据": r"(因为|依据|理由|特征|迹象|原因|由于)",
    "风险": r"(风险|诈骗|危险|可疑|骗局)",
    "止损": r"(止付|冻结|报警|银行|停止转账|不要转|挂失)",
    "报警": r"(110|报警|派出所)",
    "二次": r"(二次|再次|追回|服务费|收费|骗子)",
    "沟通": r"(沟通|说|劝|聊|开口|表达|告诉)",
    "渠道": r"(__CHANNEL__)",
    "拒绝": r"(无法|不能|不会|不可以|抱歉|恕难|不提供|拒绝)",
    "替代": r"(帮助|建议|可以|正规|渠道|咨询)",
    "案例": r"(案例|例子|有人|某|曾经|故事)",
    "共情": r"(__EMPATHY__)",
}
LEN_RANGE = {"dissuade": (60, 260), "aftermath": (100, 360), "science": (80, 320),
             "judge": (60, 300), "case": (120, 400), "family": (100, 400),
             "faq": (20, 300), "refuse": (30, 320)}

REQ = {
    "science": ["结论", "识别", "行动"],
    "case": ["案例", "识别", "行动"],
    "judge": ["判断", "依据", "行动"],
    "dissuade": ["共情", "风险", "行动"],
    "aftermath": ["共情", "止损", "报警", "二次"],
    "family": ["沟通", "行动"],
    "faq": ["结论", "渠道"],
    "refuse": ["拒绝", "替代"],
}


def _len_score(task, n):
    lo, hi = LEN_RANGE.get(task, (40, 400))
    if lo <= n <= hi:
        return 1.0
    if n < lo:
        return max(0.0, n / lo)
    return max(0.0, 1.0 - (n - hi) / hi)


def _empathy(text):
    return 1.0 if any(re.search(p, text[:120]) for p in EMPATHY) else 0.0


def _action(text):
    hits = sum(v for k, v in ACTIONS.items() if k in text)
    channels = sum(1 for c in CHANNELS if c in text)
    return min(1.0, (hits / 6.0) * 0.7 + min(1.0, channels / 2.0) * 0.3)


def _coverage(text, required):
    if not required:
        return 0.5
    hit = 0
    for r in required:
        if r == "行动":
            ok = _action(text) > 0
        elif r == "渠道":
            ok = any(c in text for c in CHANNELS)
        elif r == "共情":
            ok = _empathy(text) > 0
        else:
            pat = REQ_SEMANTIC.get(r)
            ok = bool(re.search(pat, text)) if pat else (r in text)
        hit += 1 if ok else 0
    return hit / len(required)


def _repeat(text):
    toks = list(text)
    if len(toks) < 40:
        return 1.0
    grams = ["".join(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1.0 - (len(grams) - len(set(grams))) / max(1, len(grams))


def reward(task, instruction, state, gold, output):
    text = (output or "").strip()
    if not text:
        return {"reward": 0.0, "parts": {"safety": 0.0}, "valid": False}
    for pat in BAN:
        m = re.search(pat, text)
        if m:
            ctx = text[max(0, m.start() - 16):m.start()]
            if NEG_CTX.search(ctx):
                continue
            return {"reward": 0.0, "parts": {"safety": 0.0}, "valid": False}
    required = (gold or {}).get("required") if isinstance(gold, dict) else None
    parts = {
        "safety": 0.20,
        "coverage": 0.25 * _coverage(text, required or REQ.get(task, [])),
        "action": 0.20 * _action(text),
        "empathy": 0.15 * _empathy(text),
        "cite": 0.10 * (1.0 if any(c in text for c in CHANNELS) else 0.0),
        "length": 0.05 * _len_score(task, len(text)),
        "repeat": 0.05 * _repeat(text),
    }
    return {"reward": round(min(1.0, sum(parts.values())), 4), "parts": parts, "valid": True}
