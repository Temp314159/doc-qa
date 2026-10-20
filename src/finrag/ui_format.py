"""展示用的分组、中文名称和文本导出，不修改答案或检索数据。"""
import json
import re
from .ui_service import redact
from .ui_text import TEXT as T, DISPLAY_NAMES


def display_name(value):
    value = str(value or T["source_page_missing"])
    for old, new in DISPLAY_NAMES.items():
        value = value.replace(old, new)
    return value


def group_sources(sources):
    groups = {}
    for source in sources:
        if source.get("missing"):
            continue
        e = source["evidence"]
        identity = e.get("doc_id") or e.get("evidence_id")
        group = groups.setdefault(identity, {"name": display_name(source["name"]), "pages": [], "sources": []})
        if any(s["evidence"].get("evidence_id") == e.get("evidence_id") for s in group["sources"]):
            continue
        group["sources"].append(source)
        group["pages"] = sorted(set(group["pages"]) | set(source["pages"]))
    return list(groups.values())


def hit_sources(hits, catalog):
    sources, ids = [], set()
    for hit in hits:
        for evidence in hit.get("evidence", []):
            eid = evidence.get("evidence_id")
            if eid not in ids:
                sources.append(catalog.source(evidence))
                ids.add(eid)
    return sources


def displayed_answer(result):
    if result.get("termination_reason") == "budget_or_api_limit":
        return T["answer_stopped"]
    return str(result.get("answer") or T["answer_missing"])


def calculation_display(calculation):
    bindings = calculation.get("bindings") or {}
    names = {v: T["calculation_variable_name"].format(index=i)
             for i, v in enumerate(bindings, 1)}
    formula = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: names.get(m[0], m[0]),
                     calculation.get("expression", T["calculation_missing"]))
    return formula, names


def answer_download(result, catalog, asked_at):
    from .ui_history import friendly_time
    lines = [friendly_time(asked_at), "", T["download_question"],
             result.get("question", T["source_page_missing"]), "", T["download_answer"], displayed_answer(result),
             "", T["ai_notice"], "", T["download_sources"]]
    for group in group_sources(catalog.cited(result)):
        lines.append(T["download_page_line"].format(
            name=group["name"], pages=", ".join(map(str, group["pages"]))))
        lines.extend(s["evidence"].get("content", "") for s in group["sources"])
    for calc in result.get("calculations", []):
        if calc.get("ok"):
            formula, names = calculation_display(calc)
            lines.extend(["", T["download_calculation"], formula,
                          T["download_calculation_result"].format(
                              value=calc.get("value", T["source_page_missing"]))])
            for variable, e in (calc.get("inputs") or {}).items():
                lines.append(names.get(variable, T["calculation_amount"]) + "：" +
                             e.get("content", T["source_page_missing"]))
    for visual in catalog.visual_observations(result):
        lines.extend(["", T["download_visual"], visual["observation"]])
    lines.extend(["", T["download_record_id"].format(id=result["run_id"])])
    return json.loads(redact("\n".join(lines)))


def processing_rows(result):
    states = {
        "INIT": T["process_state_init"], "DECIDE": T["process_state_decide"],
        "VALIDATE": T["process_state_validate"], "EXECUTE": T["process_state_execute"],
        "OBSERVE": T["process_state_observe"], "ANSWER_VALIDATE": T["process_state_answer_validate"],
        "DONE": T["process_state_done"], "GUARD_STOP": T["process_state_guard_stop"],
        "INSUFFICIENT_EVIDENCE": T["process_state_insufficient"], "TIMEOUT": T["process_state_timeout"],
        "ERROR": T["process_state_error"],
    }
    tools = {
        "search_tables": T["process_tool_tables"], "search_text": T["process_tool_text"],
        "search_figures": T["process_tool_figures"], "inspect_figure": T["process_tool_inspect_figure"],
        "calculate": T["process_tool_calculate"], "finish_answer": T["process_tool_finish"],
        "insufficient_evidence": T["process_tool_insufficient"],
    }
    return [{T["process_column_step"]: i,
             T["process_column_content"]: states.get(event.get("state"), T["process_state_fallback"]),
             T["process_column_action"]: tools.get(event.get("tool"), T["process_no_tool"]),
             T["process_column_result"]: T["process_failed"] if event.get("error") else T["process_recorded"]}
            for i, event in enumerate(result.get("trace", []), 1)]
