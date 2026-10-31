#!/usr/bin/env python3
"""Render readable mechanism cards from the public pack; no network or GPU use."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(root=ROOT):
    data = json.loads((root / "assets/helios-experience-pack.json").read_text())
    catalog = {i["id"]: i for i in json.loads((root / "assets/implementation-map.json").read_text())["implementations"]}
    directory = root / "references/knowledge"
    directory.mkdir(parents=True, exist_ok=True)
    counts = Counter(c["pattern"] for c in data["cases"])
    for pattern in data["patterns"]:
        path = root / pattern["details"]
        # Preserve hand-written mechanism guidance; generated cases have one source of truth.
        intro = path.read_text().split("## 如何判断当前适用范围")[0].rstrip() if path.exists() else (
            f"# {pattern['title']}\n\n{pattern['mechanism']}")
        lines = [intro, "", "## 如何判断当前适用范围", "",
                 "正确性前提（未知需要核实，冲突不可直接套用）：", ""]
        lines.extend(f"- {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in pattern["correctness_conditions"].items())
        lines.extend(["", "收益条件（不满足会降低收益预期，不等于语义错误）：", ""])
        lines.extend(f"- {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in pattern["performance_conditions"].items())
        if not pattern["performance_conditions"]:
            lines.append("由当前 workload 的目标和测量确定。")
        lines.extend(["", "最小验证：" + pattern["probe"], "", "## 已整理案例", ""])
        for case in data["cases"]:
            if case["pattern"] != pattern["id"]:
                continue
            lines.extend([f"### {case['title']}", "", f"- ID：{case['id']}",
                          f"- 状态：{case['outcome']} / {case['evidence_status']}",
                          "- 来源类别：" + case["provenance"]["collection"] + "；本次未复现实验。",
                          "- 观察：" + case["observation"], "- 经验：" + case["lesson"],
                          "- 范围：" + case["scope"],
                          "- 历史上下文：" + json.dumps(case["conditions"], ensure_ascii=False),
                          "- 正确性前提：" + json.dumps(case["correctness_conditions"], ensure_ascii=False),
                          "- 收益条件：" + json.dumps(case["performance_conditions"], ensure_ascii=False),
                          "- 失败分类：" + case["failure"]["category"],
                          "- 重试信号：" + "；".join(case["retry"]["change_signals"]),
                          "- 最小验证：" + case["retry"]["minimum_probe"],
                          "- 不应重复：" + case["retry"]["do_not_repeat"],
                          "- 代码状态：" + case["navigation"]["status"] + "；" + case["navigation"]["reason"],
                          "- 定位命令：`dmd_knowledge.py locate --case " + case["id"] + "`"])
            for link in case["navigation"]["implementations"]:
                entry = catalog[link["id"]]
                lines.append(f"- 关联实现：[{entry['title']}](implementations.md#{entry['id']})；{link['relationship']}；{link['note']}")
                for ref in entry["entrypoints"]:
                    lines.append(f"- 函数入口：[{ref['path']}::{ref['symbol']}]({ref['anchor_url']})")
            if case["evidence"]:
                for ref in case["evidence"]:
                    lines.append(f"- 来源：[{ref['repository']} / {Path(ref['path']).name}]({ref.get('anchor_url', ref['url'])})；定位：{ref['locator']}")
            else:
                lines.append("- 来源：仅本地会话线索，未公开原始会话；缺少可核实的公开实验回执。")
            lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n")
    lines = ["# 条件化优化经验库", "",
             "给没有历史会话的新 Codex 使用。先读当前 workload，再按机制检索；成功、失败和未知共同限定适用范围。", "",
             f"本版 {len(data['patterns'])} 个主题、{len(data['cases'])} 条案例与方法条目。它们包含实测报告、源码约束、外部方法建议和未核实线索，不能当成同等数量的独立成功实验。", "",
             "见 [覆盖范围与未完成项](coverage.md)。来源固定到提交与文件 hash；原始私人会话没有上传。", "",
             "| 主题 | 条目数 |", "| --- | ---: |"]
    lines.extend(f"| [{p['title']}]({Path(p['details']).name}) | {counts[p['id']]} |" for p in data["patterns"])
    lines.extend(["", "使用 `dmd_knowledge.py recall` 返回条件与同机制全部正反案例；使用 `sources` 搜索尚未整理完的历史来源。详见 [检索规则](../reuse-memory.md)。", "",
                  "使用 `locate --case CASE_ID` 从经验进入 [实现索引](implementations.md)：固定源码、函数、接入、测试与回退。`linked` 是相关参考，`method_only` 是纯方法，`unresolved` 表示对应实现尚未定位。", "",
                  "H100、A100 和外部推理案例都保留硬件/模式边界。关键词命中不等于当前适用，外部报告也不是本系统的验收结果。", "",
                  "维护者编辑 JSON pack 后运行 `scripts/render_knowledge.py` 同步这些卡片；机制说明保留，案例部分由结构化记录生成。"])
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    lines = ["# 实现与代码定位", "",
             "此表来自 implementation-map.json。所有链接固定到源码提交与函数行号；`locate --repo` 额外核对文件 hash 和 AST 符号。相关当前代码不等于历史计时所用实现。", "",
             "测试入口仅说明能在哪里检查覆盖；未列出测试时明确记录缺口。此轮没有运行这里列出的 GPU/训练验收，helper 的 CPU 检查也不能替代它们。", ""]
    for entry in catalog.values():
        lines.extend([f"<a id=\"{entry['id']}\"></a>", "", f"## {entry['title']}", "",
                      f"- ID：`{entry['id']}`；用途：{entry['usage_mode']}",
                      "- 范围：" + entry["scope"],
                      "- 正确性前提：" + json.dumps(entry["correctness_conditions"], ensure_ascii=False),
                      "- 收益条件：" + json.dumps(entry["performance_conditions"], ensure_ascii=False),
                      "- 接入：" + entry["integration_instructions"], "- 回退：" + entry["fallback"]["action"],
                      "- 验证范围：" + entry["validation_scope"]])
        if entry["test_gap"]:
            lines.append("- 测试缺口：" + entry["test_gap"])
        for related in entry.get("related_implementations", []):
            lines.append(f"- 独立备选：[{catalog[related['id']]['title']}](#{related['id']})；{related['note']}")
        for key, label in (("entrypoints", "入口"), ("integration", "接入代码"), ("tests", "测试")):
            for ref in entry[key]:
                lines.append(f"- {label}：[{ref['path']}::{ref['symbol']}]({ref['anchor_url']})")
        lines.extend(["- 核对命令：`dmd_knowledge.py locate --implementation " + entry["id"] + " --repo /project/DMD-System --show-code`", ""])
    (directory / "implementations.md").write_text("\n".join(lines) + "\n")
    return {"patterns": len(data["patterns"]), "cases": len(data["cases"])}


if __name__ == "__main__":
    print(json.dumps(render()))
