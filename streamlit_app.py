"""报告与文档工作台：显式提交、即时本地历史和统一结果展示。"""
import threading
import streamlit as st
from finrag.ui_history import HistoryStore, friendly_time, search_history
from finrag.ui_service import Catalog, search_local, ask_live
from finrag.ui_credentials import key_available, save_key
from finrag.ui_components import build_css, layout_style, render_answer, render_hits
from finrag.ui_text import TEXT as T, SHOW_EXAMPLES, set_language

st.set_page_config(page_title="Report Desk", page_icon="📘", layout="wide", initial_sidebar_state="expanded")
st.session_state.setdefault("display_language", "zh")
set_language(st.session_state.display_language)
st.markdown(build_css(), unsafe_allow_html=True)


# 文档目录只读元数据并提供预览，不加载检索索引；因此打开历史页时不需要初始化向量模型。
@st.cache_resource
def catalog():
    return Catalog()


# 稠密检索器延迟到首次提交后创建；锁把同一进程内的模型初始化与检索串行化。
@st.cache_resource
def retrieval_resource():
    from finrag.retrieval import Retriever
    return Retriever(load_dense=False), threading.RLock()


try:
    c = catalog()
    store = HistoryStore()
except Exception:
    st.error(T["load_error"])
    st.stop()

# 侧栏显隐、选中历史和输入模式只保存在当前页面会话中。
for key, value in {"panel_open": True, "selected_id": None, "question": "", "task_kind": "ask",
                   "task_mode": T["ask_mode"]}.items():
    st.session_state.setdefault(key, value)


def new_question():
    # 切回空白提问态并清理本次会话的待办标记；已写入磁盘的历史记录不删除。
    st.session_state.selected_id = None
    st.session_state.question = ""
    st.session_state.task_kind = "ask"
    st.session_state.task_mode = T["ask_mode"]
    st.session_state.history_query = ""
    st.session_state.pop("pending_id", None)
    st.session_state.pop("input_error", None)


def select_history(entry_id):
    # 历史点击只恢复问题与结果快照，不创建pending_id，因此不会重跑模型或检索。
    item = store.get(entry_id)
    st.session_state.selected_id = entry_id
    st.session_state.question = item["question"]
    st.session_state.task_kind = item["kind"]
    st.session_state.task_kind = "search" if item["kind"] == "search" else "ask"
    st.session_state.task_mode = T["search_mode"] if item["kind"] == "search" else T["ask_mode"]
    st.session_state.pop("pending_id", None)


def submit_question():
    # 先将问题和UTC提问时间写入本地历史，再由下一次页面运行领取任务。
    # pending_id只属于当前会话，刷新后不会把queued/running问题静默重发。
    try:
        item = store.begin(st.session_state.question, st.session_state.task_kind)
        st.session_state.selected_id = item["id"]
        st.session_state.pending_id = item["id"]
        st.session_state.history_query = ""
        st.session_state.pop("input_error", None)
    except ValueError:
        st.session_state.input_error = "empty"
    except OSError:
        st.session_state.input_error = "history"


def cancel_pending():
    # 用户暂不提供密钥时只取消当前会话的继续执行入口，已保存的问题仍留在历史列表。
    st.session_state.pop("pending_id", None)
    st.session_state.pop("key_input", None)


@st.dialog(T["key_title"], dismissible=False)
def key_dialog():
    st.write(T["key_help"])
    with st.form("key_form"):
        value = st.text_input(T["key_input_label"], type="password", key="key_input", autocomplete="off")
        if st.form_submit_button(T["key_save_continue"], type="primary"):
            try:
                save_key(value)
            except ValueError:
                st.warning(T["key_invalid"])
            except OSError:
                st.warning(T["key_write_error"])
            else:
                # 密钥不复制到历史/日志；弹窗结束前清理widget会话状态。
                st.session_state.pop("key_input", None)
                st.rerun()
    if st.button(T["key_cancel"], on_click=cancel_pending):
        st.rerun()


@st.dialog(T["help"], dismissible=False)
def help_dialog():
    st.write(T["help_body"])
    st.caption(T["help_recovery"])
    if st.button(T["close"], key="close_help"):
        st.rerun()


@st.dialog(T["settings"], dismissible=False)
def settings_dialog():
    # 选项值保持稳定标识；展示语言只影响固定界面文案，不改历史或原始材料。
    selected_language = st.selectbox(
        T["language"], ["zh", "en"],
        format_func=lambda value: T["language_chinese"] if value == "zh" else T["language_english"],
        index=0 if st.session_state.display_language == "zh" else 1,
        key="language_selection",
    )
    if st.button(T["settings_done"]):
        st.session_state.display_language = selected_language
        set_language(selected_language)
        st.rerun()


st.markdown(layout_style(st.session_state.panel_open), unsafe_allow_html=True)
with st.container(key="left_expand"):
    # Markdown会把单独的右尖括号解析为空引用标记，因此用实体保留可见箭头。
    st.button("&gt;", help=T["expand_sidebar"], on_click=lambda: st.session_state.update(panel_open=True))
with st.container(key="new_question_quick"):
    st.button(T["new_question"], help=T["new_question_help"], on_click=new_question)
with st.container(key="top_menu"):
    # 菜单项的显示顺序就是按钮声明顺序；帮助和设置均在弹窗中呈现中文文案。
    with st.popover("⋯", help=T["more_menu"]):
        if st.button(T["settings"], use_container_width=True):
            settings_dialog()
        if st.button(T["help"], use_container_width=True):
            help_dialog()

with st.sidebar:
    title, collapse = st.columns([5, 1])
    title.subheader(T["app_name"])
    collapse.button("<", help=T["collapse_sidebar"], on_click=lambda: st.session_state.update(panel_open=False))
    st.button(T["new_question"], key="new_question", on_click=new_question, use_container_width=True,
              help=T["new_question_help"])
    if SHOW_EXAMPLES:
        # 示例入口暂时隐藏，保留以便以后恢复；不会替代用户的新问题。
        st.caption(T["example_section"])
        for label, question in [(T["example_number"], T["example_number_question"]),
                                (T["example_calculation"], T["example_calculation_question"]),
                                (T["example_figure"], T["example_figure_question"])]:
            st.button(label, key="example_" + label,
                      on_click=lambda q=question: st.session_state.update(selected_id=None, question=q, task_kind="ask"))
    st.divider()
    st.subheader(T["history"])
    # 输入变化会触发Streamlit重绘；search_history使用本机会话快照并限制磁盘刷新频率。
    query = st.text_input(T["history_search"], key="history_query", placeholder=T["history_search"],
                          label_visibility="collapsed")
    # 固定容器只承载历史按钮；标题、搜索框和侧栏上方控件留在滚动区之外。
    with st.container(height=480, key="history_list"):
        for item in search_history(store, query, st.session_state):
            st.button(item["question"], key="history_" + item["id"], use_container_width=True,
                      type="primary" if item["id"] == st.session_state.selected_id else "tertiary",
                      on_click=select_history, args=(item["id"],))

# 新问题页保留标题与副标题；显示历史结果时只留下可折叠的问题输入区。
selected = store.get(st.session_state.selected_id) if st.session_state.selected_id else None


def render_question_form():
    """提问区域在新问题页常显，在历史回答页折叠显示。"""
    st.header(T["question"])
    st.caption(T["question_hint"])
    st.text_area(T["question"], key="question", max_chars=1024, label_visibility="collapsed", height=100,
                 placeholder=T["question_placeholder"])
    # 第一列容纳二选一模式，第二列给主要提交操作更多横向空间。
    mode, submit = st.columns([1.05, 2.95], gap="small", vertical_alignment="center")
    # 单一横向单选控件由样式绘制成连续分段，避免两个模式按钮表现为并列命令。
    st.session_state.task_mode = T["search_mode"] if st.session_state.task_kind == "search" else T["ask_mode"]
    mode.radio(T["question"], [T["ask_mode"], T["search_mode"]], horizontal=True,
               key="task_mode", label_visibility="collapsed", on_change=sync_task_kind,
               help=T["ask_help"] if st.session_state.task_kind == "ask" else T["search_help"])
    submit.button(T["submit_ask"] if st.session_state.task_kind == "ask" else T["submit_search"],
                  key="submit_question", type="primary", use_container_width=True, on_click=submit_question)
    if st.session_state.get("input_error"):
        error_text = T["error_question_empty"] if st.session_state.input_error == "empty" else T["error_history_write"]
        st.warning(error_text)


def sync_task_kind():
    """将本地化的单选项转换为稳定的任务类型，供历史记录和提交逻辑使用。"""
    st.session_state.task_kind = "search" if st.session_state.task_mode == T["search_mode"] else "ask"


if selected is None:
    st.title(T["headline"])
    st.caption(T["tagline"])
    render_question_form()
else:
    with st.expander(T["question_form"], expanded=False):
        render_question_form()

# 只有本次明确提交的任务可运行。密钥检查在领取前完成，重启后未完成记录不自动重发。
pending = store.get(st.session_state.get("pending_id")) if st.session_state.get("pending_id") else None
if pending:
    # 查找任务不依赖问答密钥；问答先确认密钥，再以claim把queued状态一次性转成running。
    try:
        ready = pending["kind"] == "search" or key_available()
    except OSError:
        ready = False
    if not ready:
        key_dialog()
    elif store.claim(pending["id"]):
        try:
            with st.spinner(T["loading_search"] if pending["kind"] == "search" else T["loading_answer"], show_time=False):
                retriever, lock = retrieval_resource()
                with lock:
                    if pending["kind"] == "search":
                        store.update(pending["id"], state="completed", hits=search_local(pending["question"], retriever))
                    else:
                        result = ask_live(pending["question"], retriever, run_id=pending["id"])
                        store.update(pending["id"], state="completed", result=result)
        except Exception as exc:
            store.update(pending["id"], state="failed", error_type=type(exc).__name__)
        st.session_state.pop("pending_id", None)
        st.rerun()

if selected:
    # 完成记录、查找结果和失败记录共享同一问题区域，分支只决定下方结果组件。
    st.caption(friendly_time(selected.get("asked_at")))
    st.header(selected["question"])
    if selected.get("result"):
        render_answer(selected["result"], c, recorded_at=selected.get("asked_at"))
    elif selected.get("hits") is not None:
        render_hits(selected["hits"], c, selected["question"], key=selected["id"])
        with st.expander(T["details"]):
            st.caption(T["record_id"].format(id=selected["id"]))
            with st.expander(T["process_details"]):
                st.text(T["search_complete"])
    elif selected["state"] == "failed":
        st.info(T["failed_notice"])
        with st.expander(T["details"]):
            st.caption(T["record_id"].format(id=selected["id"]))
            st.caption(T["failed_details"])
    else:
        st.caption(T["queued_notice"])
