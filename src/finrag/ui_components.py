"""Streamlit展示组件；资料纯文本，固定样式/转义后的状态提示可使用HTML。"""
import html
import json
from decimal import Decimal, InvalidOperation
from urllib.parse import quote
import streamlit as st
import pandas as pd
from filelock import Timeout as LockTimeout
from .ui_service import FEEDBACK, calculation_rows, safe_asset, save_feedback, status_message
from .ui_format import (answer_download, calculation_display, display_name, displayed_answer,
                        group_sources, hit_sources, processing_rows)
from .ui_text import TEXT as T, STATUS, SHOW_SOURCE_EXCERPTS

# 手绘纸笔矢量图标，通过遮罩显示在真正可点击的按钮中。
NEW_QUESTION_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M13 4H5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h13a2 2 0 0 0 2-2v-8"/><path d="m10 14 1-4L19 2a1.4 1.4 0 0 1 2 2l-8 8-3 2Zm7-10 3 3"/></svg>'
ICON_URL = "data:image/svg+xml," + quote(NEW_QUESTION_SVG)
# 页面颜色、浮动按钮、历史列表滚动区和响应式断点集中在此处。
# 选择器依赖Streamlit 1.50的组件标记，升级Streamlit后需按实际DOM复核。
_CSS_TEMPLATE = """
<style>
.stApp {background:#f7f8fc;color:#172b4d}
[data-testid="stSidebar"] {background:#edf1f7;color:#172b4d}
h1,h2,h3 {color:#173b67}
h1 {font-size:2.2rem !important}
h2 {font-size:1.65rem !important}
h3 {font-size:1.25rem !important}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:12px}
.stButton>button[kind="primary"] {background:#173b67;border-color:#173b67}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu,
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stElementToolbar"], [data-testid="stToolbarActions"] {display:none !important}
.block-container {padding-top:4.2rem;max-width:1150px}
[data-testid="stSidebarUserContent"] {padding-top:1.4rem}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {height:100vh;overflow:hidden}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {height:100vh;overflow:hidden}
.st-key-top_menu {position:fixed;right:24px;top:12px;z-index:999;width:auto !important}
.st-key-top_menu button {font-size:1.5rem;min-height:38px;width:42px;padding:0;display:flex;align-items:center;justify-content:center}
.st-key-left_expand {position:fixed;left:12px;top:12px;z-index:999;width:auto !important}
.st-key-left_expand button {width:38px;height:38px;padding:0;display:flex;align-items:center;justify-content:center;font-size:1.2rem}
.st-key-new_question_quick {position:fixed;left:58px;top:12px;z-index:999;width:auto !important}
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {display:none}
.st-key-top_menu [data-testid="stPopoverButton"] svg {display:none}
.st-key-top_menu [data-testid="stPopoverButton"] p {margin:0;line-height:1}
.st-key-new_question_quick button {font-size:0;width:42px;padding:8px}
.st-key-new_question button:before,.st-key-new_question_quick button:before {
content:"";width:21px;height:21px;display:inline-block;flex-shrink:0;background:currentColor;
mask:url('ICON_URL') center/contain no-repeat;-webkit-mask:url('ICON_URL') center/contain no-repeat}
/* 额外右内边距将按钮内的纸笔图标与文字整体左移，按钮点击区域仍保持全宽。 */
.st-key-new_question button {gap:14px;padding-left:0;padding-right:28px}
.st-key-new_question button p {margin-left:2px}
.st-key-new_question_quick button p {position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
[data-testid="stSidebar"] .stButton>button {text-align:left;justify-content:flex-start;width:100%}
[data-testid="stSidebar"] .stButton>button p {white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* 307px为侧栏标题、新问题入口、分隔线和搜索框占用高度；其余视口高度留给无边框历史列表。 */
.st-key-history_list.stVerticalBlock {height:calc(100vh - 307px) !important;min-height:100px;border:0 !important;box-shadow:none !important;flex:0 0 calc(100vh - 307px);overflow-y:auto;overscroll-behavior:contain}
.st-key-history_list [data-testid="stVerticalBlockBorderWrapper"] {border:0 !important;box-shadow:none !important}
.st-key-history_list [data-testid="stElementContainer"] {width:100%}
.st-key-task_mode,.st-key-task_mode [data-testid="stRadio"],.st-key-task_mode [role="radiogroup"] {width:100% !important;max-width:none}
.st-key-task_mode [role="radiogroup"] {display:flex;gap:0;padding:4px;border:1px solid #cbd5e1;border-radius:8px;background:#fff}
/* 两个模式标签固定为等宽轨道；min-width:0解除浏览器按文字长度分配空间，英文短标签也不会挤压或换行。 */
.st-key-task_mode [role="radiogroup"] > label {position:relative;flex:1 1 0;min-width:0;width:0;justify-content:center;margin:0;padding:12px;border-radius:5px;cursor:pointer;white-space:nowrap}
.st-key-task_mode [role="radiogroup"] > label p {margin:0;white-space:nowrap;word-break:keep-all;overflow:hidden;text-overflow:ellipsis}
.st-key-task_mode [role="radiogroup"] > label:has(input:checked) {background:#173b67;color:#fff;font-weight:700}
.st-key-task_mode [role="radiogroup"] > label:has(input:checked) * {color:#fff !important}
.st-key-task_mode [role="radiogroup"] > label > div:first-child {display:none}
.st-key-task_mode [role="radiogroup"] > label:hover::after {position:absolute;top:calc(100% + 8px);left:0;z-index:50;width:260px;padding:10px 12px;border:1px solid #dce3ed;border-radius:7px;background:#fff;color:#172b4d;box-shadow:0 5px 16px #173b6718;font-size:13px;font-weight:400;line-height:1.5;text-align:left;white-space:normal}
.st-key-task_mode [role="radiogroup"] > label:nth-of-type(1):hover::after {content:ASK_MODE_HELP}
.st-key-task_mode [role="radiogroup"] > label:nth-of-type(2):hover::after {content:SEARCH_MODE_HELP}
.st-key-submit_question button {min-height:56px;font-size:1.3rem;font-weight:700}
.answer-heading {display:flex;align-items:center;gap:10px;color:#173b67;font-size:1.65rem;font-weight:650;margin:12px 0}
.answer-status {position:relative;display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;font-size:16px;cursor:help}
.answer-status.ok {color:#23724b;background:#e0f2e7}
.answer-status.notice {color:#9b691b;background:#fff0d5}
.answer-status.error {color:#a43d49;background:#f9e3e6}
.answer-status .status-tip {display:none;position:absolute;left:0;top:30px;width:290px;background:white;color:#172b4d;padding:14px;border:1px solid #dce3ed;border-radius:10px;box-shadow:0 6px 20px #173b6720;z-index:30;font-size:14px;font-weight:400;line-height:1.6}
.answer-status .status-tip small {display:block;margin-top:4px;font-size:12px;color:#65748b}
.answer-status:hover .status-tip,.answer-status:focus .status-tip {display:block}
.st-key-run_details {font-size:.9rem;color:#64748b;margin-top:18px}
.st-key-run_details summary p {font-size:.9rem !important}
[data-testid="StyledFullScreenButton"], [data-testid="stImage"] button {display:none !important}
[data-testid="InputInstructions"], [data-testid="stTextInput"] button {display:none !important}
[data-testid="stTable"] th:first-child {display:none}
@media(max-width:650px){
  .block-container{padding-left:1.2rem;padding-right:1.2rem}
  h1{font-size:1.8rem !important}
  .st-key-history_list.stVerticalBlock {height:calc(100vh - 307px) !important;min-height:100px;flex-basis:calc(100vh - 307px)}
  .st-key-task_mode [role="radiogroup"] > label {padding:10px 6px}
  .st-key-submit_question button {min-height:50px;font-size:1.1rem}
}
</style>
"""


def build_css():
    """按当前界面语言生成控件样式与模式悬停说明。"""
    return _CSS_TEMPLATE.replace("ICON_URL", ICON_URL).replace(
        "ASK_MODE_HELP", json.dumps(T["ask_help"], ensure_ascii=False)
    ).replace("SEARCH_MODE_HELP", json.dumps(T["search_help"], ensure_ascii=False))


def layout_style(panel_open):
    """手工侧栏按钮和浮动工具栏按显隐状态切换。"""
    sidebar = "" if panel_open else '[data-testid="stSidebar"] {display:none !important}'
    shortcut = '.st-key-left_expand,.st-key-new_question_quick {display:none}' if panel_open else ""
    return "<style>" + sidebar + shortcut + "</style>"


def source_card(group, catalog, key):
    # 一个卡片对应一个文档；多个证据已由group_sources合并，页码在卡片内并列显示。
    with st.container(border=True):
        pages = ", ".join(map(str, group["pages"])) or T["source_page_missing"]
        st.text(group["name"] + " · " + T["page_label"] + " " + pages)
        sources = group["sources"]
        if any(p.get("method") == "ocr" for s in sources for p in s["evidence"].get("provenance", [])):
            st.caption(T["ocr_notice"])
        if SHOW_SOURCE_EXCERPTS:
            for content in dict.fromkeys(s["evidence"].get("content") or T["source_excerpt_missing"] for s in sources):
                st.text(content)
        if st.toggle(T["preview"], value=False, key=key + "_preview"):
            # 预览只根据登记的原PDF页码渲染；“展示所有页”按来源页码升序逐页显示。
            options = ([T["all_pages"]] if len(group["pages"]) > 1 else []) + group["pages"]
            if not options:
                st.info(T["preview_page_missing"])
            else:
                chosen = st.selectbox(T["page_label"], options, key=key + "_page")
                for page in group["pages"] if chosen == T["all_pages"] else [chosen]:
                    source = next(s for s in sources if page in s["pages"])
                    preview = catalog.page_preview(source["evidence"], page)
                    if preview:
                        st.image(str(preview), caption=group["name"] + f" · {T['page_label']} {page}")
                    else:
                        st.info(T["preview_unavailable"].format(page=page))


def render_sources(sources, catalog, key):
    # 计数反映去重前来源条数和合并后的文档卡片数，避免把证据条目误读成多份文件。
    groups = group_sources(sources)
    st.caption(T["source_count"].format(count=len(sources), groups=len(groups)))
    if not sources:
        st.info(T["sources_empty"])
    if any(s.get("missing") for s in sources):
        st.caption(T["sources_partial"])
    for index, group in enumerate(groups):
        source_card(group, catalog, key + f"_{index}")


def feedback_dialog(result, source, root):
    # 弹窗标题在调用时读取，避免模块缓存后继续显示启动时的语言。
    @st.dialog(T["feedback_title"], dismissible=False)
    def dialog():
        with st.form("feedback_form_" + result["run_id"]):
            kind = st.radio(T["feedback_choice"], list(FEEDBACK),
                            format_func=lambda value: T["feedback_" + value], horizontal=True)
            note = st.text_area(T["feedback_note"], max_chars=500)
            if st.form_submit_button(T["feedback_submit"]):
                try:
                    saved = save_feedback(result["run_id"], kind, note, source, root)
                    st.success(T["feedback_saved"] if saved else T["feedback_duplicate"])
                except ValueError:
                    st.warning(T["feedback_unavailable"])
                except (OSError, LockTimeout):
                    st.warning(T["feedback_write_error"])
        if st.button(T["close"], key="close_feedback"):
            st.rerun()
    dialog()


def render_answer(result, catalog, source="history", recorded_at=None):
    """按统一顺序展示问题回答、原文、计算、图片和可选的后台处理信息。"""
    title, advice = status_message(result)
    color = STATUS.get(result.get("status"), STATUS["ERROR"])[2]
    mark = {"ok": "✓", "notice": "×", "error": "!"}[color]
    # 回答和问题处于同一层级；状态文字只在图标悬停或键盘聚焦时出现。
    st.markdown(f'<div role="heading" aria-level="2" class="answer-heading">{html.escape(T["answer"])}<span tabindex="0" class="answer-status {color}" aria-label="{html.escape(title, quote=True)}">{mark}<span class="status-tip">{html.escape(title)}<small>{html.escape(advice)}</small></span></span></div>', unsafe_allow_html=True)
    # 正文保持纯文本；状态、引用和下方证据模块按固定层级展示，不重写模型原答。
    st.text(displayed_answer(result))
    st.caption(T["ai_notice"])
    st.download_button(T["save"], data=answer_download(result, catalog, recorded_at),
                       file_name=result["run_id"] + ".txt", mime="text/plain", key="download_" + result["run_id"])
    st.subheader(T["sources"])
    render_sources(catalog.cited(result), catalog, result["run_id"] + "_cite")
    if any(calc.get("ok") for calc in result.get("calculations", [])):
        # 公式、输入金额和简要结果整体收起；更高精度的数值再单独折叠。
        with st.expander(T["calculation"], expanded=False):
            for calc in result["calculations"]:
                if not calc.get("ok"):
                    continue
                formula, names = calculation_display(calc)
                st.text(T["calculation_formula"].format(formula=formula))
                # 计算表回显原始金额、单位、期间和页码；不把统一到基础单位的内部数值冒充原值。
                rows = calculation_rows(calc)
                for row in rows:
                    row[T["calculation_amount"]] = names.get(
                        row.pop("变量"), T["calculation_amount"])
                    row[T["calculation_page"]] = row.pop("原页", T["source_page_missing"])
                column_names = {
                    "项目与原始金额": T["table_raw_amount"], "单位": T["table_unit"],
                    "期间": T["table_period"], "页码": T["table_page"],
                    "金额": T["table_amount"], "资料编号": T["table_source_id"],
                }
                st.table(pd.DataFrame([
                    {column_names.get(k, k): display_name(v) for k, v in row.items()
                     if k != T["calculation_evidence_id"]}
                    for row in rows
                ]).style.hide(axis="index"))
                st.caption(T["calculation_note"])
                try:
                    shown = format(Decimal(str(calc["value"])), ".6f").rstrip("0").rstrip(".")
                except (KeyError, InvalidOperation):
                    shown = T["source_page_missing"]
                st.text(T["calculation_result"].format(value=shown))
                with st.expander(T["calculation_full_result"], expanded=False):
                    st.text(T["calculation_full_value"].format(
                        value=calc.get("value", T["source_page_missing"])))
                    st.caption(T["calculation_full_note"])
    # 仅通过服务层真实图片观察校验的记录进入图片区；图题或图片引用本身不算识图结果。
    visuals = catalog.visual_observations(result)
    if visuals:
        # 先收起整组图片；图题、页码及观察文字放在每张图片自己的第二层折叠中。
        with st.expander(T["visual"], expanded=False):
            for visual in visuals:
                figure = visual["figure"]
                page = (figure.get("provenance") or {}).get("original_page", T["source_page_missing"])
                crop = safe_asset(figure.get("crop_path"), "image", catalog.root)
                if crop:
                    st.image(str(crop), caption=T["visual_answer_caption"])
                else:
                    st.info(T["visual_missing_asset"])
                with st.expander(T["visual_details"], expanded=False):
                    st.text(figure.get("title", T["visual_default_title"]))
                    st.caption(f"{T['page_label']} {page}")
                    st.text(visual["observation"])
                    st.caption(T["visual_ai_notice"])
    elif any(eid in catalog.figures for eid in result.get("citations", [])):
        st.caption(T["visual_missing_observation"])
    with st.container(key="run_details"):
        with st.expander(T["details"], expanded=False):
            # 资料字段用于核查检索范围，置于详情首位；运行信息与步骤放在其后。
            facts = sorted({(catalog.docs.get(e.get("doc_id"), {}).get("company") or T["company_unspecified"],
                             e.get("period") or T["period_unspecified"], e.get("unit") or T["unit_unspecified"])
                            for e in result.get("evidence", []) if e.get("evidence_id") in result.get("citations", [])})
            if facts:
                summary = "；".join(" / ".join(display_name(v) for v in fact) for fact in facts)
                st.text(T["field_summary"].format(fields=summary))
            st.text(T["process_time"].format(seconds=result.get("elapsed", 0)))
            st.caption(T["record_id"].format(id=result["run_id"]))
            with st.expander(T["process_details"]):
                st.text(T["process_result"].format(value=title))
                st.text(T["process_request_count"].format(count=result.get("api_calls", 0)))
                rows = processing_rows(result)
                if rows:
                    st.table(pd.DataFrame(rows).style.hide(axis="index"))
                else:
                    st.caption(T["process_empty"])
            if st.button(T["feedback"], key="feedback_" + result["run_id"]):
                feedback_dialog(result, source, catalog.root)


def render_hits(hits, catalog, question, key="search"):
    """查找模式不生成回答，命中的Evidence仍复用回答页的文档卡片格式。"""
    st.header(T["materials"])
    render_sources(hit_sources(hits, catalog), catalog, key)
