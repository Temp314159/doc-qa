# 数据结构约定（schema）：Pydantic按字段与类型接收/拒绝记录，不核验PDF事实。
# 结构合法不表示金额、单位或来源身份已经正确。
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


# 统一禁止额外字段，减少拼错字段名或夹带未定义参数。
# 这是格式门槛，执行权限、业务意义和事实核对要由别处负责。
class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# 来源记录把内容带回物理页：文档/变体、原页、工作页、框及解析方法。
# 页号从1起，表内row/col从0起；页号合法仍不保证page_map正确。
class Provenance(Model):
    doc_id: str
    source_anchor: str
    variant_id: str
    original_page: int = Field(ge=1)
    working_page: int = Field(ge=1)
    # 坐标统一为左上原点的points：[x0,y0,x1,y1]；1点=1/72英寸。
    # 渲染像素要乘dpi/72，不能把点直接当像素，page_size也采用points。
    bbox: tuple[float, float, float, float]
    page_size: tuple[float, float]
    # native/ocr描述解析路径，manual_override单独描述人工覆盖。
    # 它们与Document.source_type、是否读取缓存属于不同维度。
    method: Literal["native", "ocr"]
    manual_override: bool = False
    coordinate_system: str = "top_left_points"


# 解析器恢复的正文块，不必等同最终chunk；多个块可组成一个检索视图。
class TextBlock(Model):
    block_id: str
    text: str
    heading: str = ""
    role: str = "body"
    provenance: Provenance


# 一个表格格子：原文text用于核对，规范值value用于算数，二者一起保留。
# 例如245,122配USD million变为2.45122E+11；G00不能合法解析就保留None。
class Cell(Model):
    cell_id: str
    row: int
    col: int
    text: str
    # 跨度描述合并格覆盖多少行/列；它是结构信息，不证明模型一定恢复准确。
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    value: str | None = None
    unit: str = "unknown"
    period: str = ""
    provenance: Provenance


# 一张逻辑表可来自多个物理页的表片段。拼接改逻辑行号，却保留每个Cell的来源。
# fragments记录组成，stitch_evidence解释为何接，removed_headers保留删出的重复首行。
class LogicalTable(Model):
    table_id: str
    title: str
    unit: str
    periods: list[str] = Field(default_factory=list)
    cells: list[Cell]
    rows: int
    cols: int
    fragments: list[str]
    provenance: list[Provenance]
    stitch_evidence: list[dict] = Field(default_factory=list)
    removed_headers: list[dict] = Field(default_factory=list)


# 登记的图与裁剪文件；manual_crop只表示框由人工指定，不表示图内答案由人工给出。
class Figure(Model):
    figure_id: str
    title: str
    caption: str
    crop_path: str
    provenance: Provenance
    manual_crop: bool = False


# 文档是公司/年度/来源/页映射的容器，内部放正文、完整表和图。
# source_type的三个值受Literal限制，但正确登记来源还依赖下载/生成核验。
class Document(Model):
    schema_version: str = "1.0"
    doc_id: str
    company: str
    year: str
    language: str
    split: Literal["dev", "test"]
    # original=原报告；reconstructed=真实数据重新排版；synthetic=构造练习材料。
    # 例如微软内容original，工作页可以是派生扫描；original不等于native。
    source_type: Literal["original", "reconstructed", "synthetic"]
    source_group: str
    original_path: str
    source_sha256: str
    working_path: str
    variant_id: str
    # 第w个工作页来自page_map[w-1]这个原页，选页可不连续，不能只加固定偏移。
    # scan_pages保存工作页编号，与原页编号分开使用。
    page_map: list[int]
    scan_pages: list[int]
    parse_config: dict
    blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[LogicalTable] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    filter_log: list[dict] = Field(default_factory=list)


# 视图（view）是供检索的表示；text与structural_text可共享一批事实证据。
# parent_id保存归属关系，evidence_ids明确返回依据，存在父关系不等于自动扩展。
class View(Model):
    view_id: str
    kind: Literal["text", "table", "row", "figure"]
    doc_id: str
    text: str
    structural_text: str = ""
    parent_id: str
    evidence_ids: list[str]
    anchors: list[str]
    provenance: list[Provenance]
    sparse_only: bool = False


# 证据（evidence）是可引用/绑定的记录。只有value非空的格才适合数值变量。
# ID解决“用了哪条记录”，不能纠正OCR或选错财务科目。
class Evidence(Model):
    evidence_id: str
    source_anchor: str
    doc_id: str
    parent_id: str
    content: str
    value: str | None = None
    unit: str = "unknown"
    period: str = ""
    provenance: list[Provenance]


# 一次工具循环的产物；trace给路径，calculations给算式/操作数，evidence给已访问依据。
# DONE表示当前协议检查通过，答案正确性另由独立gold评测。
class RunResult(Model):
    run_id: str
    status: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)
    calculations: list[dict] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    elapsed: float
    api_calls: int
    cost_cny: str
    termination_reason: str
    mode: str
