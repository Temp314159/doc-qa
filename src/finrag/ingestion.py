# 解析层：工作PDF→正文/Cell/逻辑表/图，保留原页定位，并输出真实运行/缓存日志。
# 页边过滤和跨页拼接采用明确启发式，不等于通用财务语义理解。
import gc
import importlib.metadata
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
import fitz
from .paths import ROOT
from .store import read_json, write_json, stable_id, sha256
from .schema import Document, Provenance, TextBlock, Cell, LogicalTable, Figure
from .numbers import parse_number


# Docling坐标转成统一左上原点points；高度用于从其他原点换算。
# 下游裁图仍须按dpi/72映射，四个数本身没有携带像素单位。
def top_left_bbox(box, height):
    b = box.to_top_left_origin(page_height=height)
    return (float(b.l), float(b.t), float(b.r), float(b.b))


# 只给页边重复判定生成模板：去空白、数字换#、小写。
# 不是对整表去数字，也不用于把不同年份的数据行认作同一事实。
def _template(text):
    return re.sub(r"\d+", "#", re.sub(r"\s+", "", text)).lower()


# 输入正文块、表片段、处理页数，返回保留块与删除日志。
# 先保护表内/单位/附注，再选择页边候选，最后确认页码或重复模式；顺序不能颠倒。
def filter_margins(blocks, tables, page_count):
    candidates=[]
    for b in blocks:
        p=b.provenance; x0,y0,x1,y1=p.bbox; h=p.page_size[1]
        # 框在表内（含3点容差）或含保护关键词就先保留。
        # 这样页底数据/单位不会仅因位置低就被删除；保护词不完备，仍有边界。
        inside_table=any(p.original_page==q.original_page and x0>=q.bbox[0]-3 and x1<=q.bbox[2]+3
            and y0>=q.bbox[1]-3 and y1<=q.bbox[3]+3 for t in tables for q in t.provenance)
        protect=inside_table or bool(re.search(r"(?:单位|注[:：]|附注|footnote|note:|in millions|in thousands)",b.text,re.I))
        # 页顶5%或页底8%只是候选范围，不是无条件裁去这些区域。
        if not protect and (y1<h*.05 or y0>h*.92):
            candidates.append(b)
    counts=Counter()
    pages=defaultdict(set)
    # 按不同原页数计重复，不把一页内重复三次误当跨三页。
    for b in candidates: pages[_template(b.text)].add(b.provenance.original_page)
    for k,ps in pages.items(): counts[k]=len(ps)
    kept=[]; log=[]
    for b in blocks:
        page_number=bool(re.fullmatch(r"\s*(?:Page\s*)?\d+\s*",b.text,re.I))
        explicit_footer=bool(re.search(r"财报教学示例|Apple Inc\.\s*\|\s*2024 Form 10-K",b.text))
        # 一般重复至少3页且覆盖处理页60%；两页合成例删页脚走explicit_footer而非此门槛。
        # 日志reason说明真正命中的条件，用于回原页确认是否误删。
        repeat=counts[_template(b.text)]>=3 and counts[_template(b.text)]/page_count>=.6
        if b in candidates and (page_number or explicit_footer or repeat):
            log.append({"block_id":b.block_id,"text":b.text,"bbox":b.provenance.bbox,
                "page":b.provenance.original_page,"reason":"page_number" if page_number else "explicit_footer" if explicit_footer else "repeated_margin"})
        else:
            kept.append(b)
    return kept,log


# 这里只比较row=0文字并规范空白/大小写，不是通用多行表头等价判断。
def _header(t):
    cs=[c for c in t.cells if c.row==0]
    return tuple(re.sub(r"\s+","",c.text).lower() for c in sorted(cs,key=lambda c:c.col))


def _has_header(t):
    return any(c.row==0 and c.is_header for c in t.cells)


# 每列对单列格框中心取平均，再除页宽，比较相对水平位置。
# 跨列大标题不参与该平均；几何对齐只是续表线索，不证明表身份相同。
def _column_centres(t):
    ps=t.provenance[0]; w=ps.page_size[0]
    centres=[]
    for col in range(t.cols):
        boxes=[c.provenance.bbox for c in t.cells if c.col==col and c.col_span==1]
        if not boxes: return []
        centres.append(sum((b[0]+b[2])/2 for b in boxes)/len(boxes)/w)
    return centres


# 表片段按原页/纵位置排序，只与最近保留逻辑表output[-1]比较。
# 返回逻辑表和全部比较日志；会原地调整参与合并的Cell.row，但不改原页/bbox。
def stitch_tables(tables):
    output=[]; decisions=[]
    for t in sorted(tables,key=lambda t:(t.provenance[0].original_page,t.provenance[0].bbox[1])):
        if not output:
            output.append(t);continue
        a=output[-1]; pa=a.provenance[-1]; pb=t.provenance[0]
        # 使用相邻原页，不能因为工作页相邻就接上不连续的原PDF页面。
        # 前片靠底/后片靠顶、列数与中心对齐共同限制候选。
        adjacent=pa.doc_id==pb.doc_id and pb.original_page==pa.original_page+1
        edge=pa.bbox[3]>pa.page_size[1]*.65 and pb.bbox[1]<pb.page_size[1]*.28
        cols=a.cols==t.cols
        ca,cb=_column_centres(a),_column_centres(t)
        geometry=bool(ca and cb and len(ca)==len(cb) and max(abs(x-y) for x,y in zip(ca,cb))<.045)
        clean=lambda s:re.sub(r"[\s（()）]|续表|continued","",s.lower())
        title_conflict=bool(t.title and a.title and clean(t.title)!=clean(a.title) and "续" not in t.title and "continued" not in t.title.lower())
        # 双方已知且不同才算冲突；unknown不冲突不代表“已证明同单位”。
        # 继承前表单位可能补齐省略，也可能传播上游错误。
        unit_conflict=t.unit!="unknown" and a.unit!="unknown" and t.unit!=a.unit
        header_same=_has_header(a) and _has_header(t) and _header(a)==_header(t)
        explicit_cont="续" in t.title or "continued" in t.title.lower()
        inferred_cont=not _has_header(t) and _has_header(a) and not t.title and geometry
        # 必要条件全部and通过，最后续表线索三选一：同表头/明确续表/无标题表头的继承。
        # “续表”不能绕过单位冲突或列数不同；每个判断保存，便于解释为何接/不接。
        ok=adjacent and edge and cols and geometry and not title_conflict and not unit_conflict and (header_same or explicit_cont or inferred_cont)
        reason={"from":a.table_id,"to":t.table_id,"adjacent":adjacent,"edge":edge,"same_columns":cols,
            "aligned_columns":geometry,"same_header":header_same,"explicit_continuation":explicit_cont,
            "header_inherited":inferred_cont,"title_conflict":title_conflict,"unit_conflict":unit_conflict,"merged":ok}
        decisions.append(reason)
        if not ok:
            output.append(t);continue
        # 只删除相同首行表头，一行而非任意多层表头；删除格另存removed_headers供审计。
        skip=1 if _has_header(t) and header_same else 0
        if skip:
            a.removed_headers.append({"fragment_id":t.table_id,"cells":[c.model_dump() for c in t.cells if c.row==0]})
        headers={c.col:c.text for c in a.cells if c.row==0}
        for cell in t.cells:
            if cell.row<skip: continue
            # 新逻辑行=后片旧行+前表行数-skip。前片5行、后片行1、skip1→逻辑行5。
            # 物理来源仍留后页；无表头时skip0，后片数据行0直接接到前表行数位置。
            cell.row+=a.rows-skip
            if cell.unit=="unknown": cell.unit=a.unit
            if not cell.period: cell.period=headers.get(cell.col,"")
            # 缺单位/期间可继承，再尝试解析原文；G00仍得None，不会自动修600。
            # 本机制处理整行续页，不重建被分页截成两半的同一数据行。
            if cell.value is None: cell.value=parse_number(cell.text,cell.unit)
            a.cells.append(cell)
        a.rows+=t.rows-skip; a.fragments+=t.fragments; a.provenance+=t.provenance
        a.stitch_evidence.append(reason)
    return output,decisions


# 有限单位模式识别，不是完整金融单位解析；未命中明确返回unknown。
def _unit_line(text):
    if re.search(r"(?:人民币|单位[：:]?\s*)?百万元",text): return "人民币百万元"
    if re.search(r"(?:人民币|单位[：:]?\s*)?千元",text): return "人民币千元"
    if re.search(r"(?:人民币|单位[：:]?\s*)?亿元",text): return "人民币亿元"
    if re.search(r"(?:单位[：:]?\s*)件",text): return "count"
    if re.search(r"(?:USD|dollars?|\$).*?millions?|in millions",text,re.I): return "USD million"
    if "USD million" in text: return "USD million"
    if re.search(r"in thousands",text,re.I): return "USD thousand"
    return "unknown"


# 多行文本从后往前找单位，倾向于最近的说明；仍可能抓到另一表单位。
def _unit(text):
    for line in reversed(text.splitlines()):
        unit=_unit_line(line)
        if unit!="unknown": return unit
    return "unknown"


# 真实Docling管线：版面/表结构用CUDA，EasyOCR简中+英文用CPU。
# 原生关闭OCR，扫描强制全页；权重路径项目内，禁OCR自动下载。
def make_converter(ocr):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    opts=PdfPipelineOptions(artifacts_path=ROOT/"models/docling", do_ocr=ocr, do_table_structure=True,
        generate_picture_images=False, generate_page_images=False, document_timeout=180,
        accelerator_options=AcceleratorOptions(num_threads=6,device=AcceleratorDevice.CUDA),
        ocr_options=EasyOcrOptions(lang=["ch_sim","en"],force_full_page_ocr=ocr,use_gpu=False,
            model_storage_directory=str(ROOT/"models/easyocr"),download_enabled=False))
    return DocumentConverter(format_options={InputFormat.PDF:PdfFormatOption(pipeline_options=opts)})


# 清单指定工作文件/页映射/扫描页，输出Pydantic Document并写解析文件。
# force绕过整理和原始页两层缓存；它不会自动更新检索视图和向量索引。
def parse_document(manifest, converters=None, force=False):
    m=manifest; outpath=ROOT/f"data/parsed/{m['doc_id']}_{m['variant_id']}.json"
    config={"docling":importlib.metadata.version("docling"),"docling_core":importlib.metadata.version("docling-core"),
        "ocr":"EasyOCR ch_sim/en CPU full_page", "working_sha256":sha256(ROOT/m["working_path"]),
        "schema_version":"1.0","rules_version":"1.3"}
    # 整理后缓存比较这些版本/工作哈希，命中是旧真实产物复用，不是本次重新跑模型。
    # 当前键未覆盖全部OCR语言/模型/清单参数，修改配置时不能仅凭命中声称复验。
    if outpath.exists() and not force:
        cached=Document.model_validate(read_json(outpath))
        if cached.parse_config==config:
            print(m["doc_id"]+"：解析缓存命中（不是首次解析）",flush=True)
            return cached
    converters=converters or {False:make_converter(False),True:make_converter(True)}
    d=Document(**{k:m[k] for k in Document.model_fields if k in m},parse_config=config)
    pdf=fitz.open(ROOT/m["working_path"])
    blocks=[];tables=[];figures=[]
    metadata_reader=None
    # w是1起工作页，orig由page_map取原页；模型读工作w，证据登记orig。
    # PyMuPDF对象用0起，所以访问pdf[w-1]；来源页另由映射取得。
    for w,orig in enumerate(m["page_map"],1):
        ocr=w in m["scan_pages"]
        page=pdf[w-1]; width,height=page.rect.width,page.rect.height
        # 扫描声明与实际输入交叉检查，防止测试页偷留文字层而绕过OCR。
        if ocr and page.get_text().strip(): raise ValueError("被标记扫描的页仍含文字层")
        rawpath=ROOT/f"data/parsed/docling_raw/{m['doc_id']}_{m['variant_id']}_{w}.json"
        rawmeta=rawpath.with_suffix('.meta.json')
        start=time.monotonic()
        # 逐页原始缓存键比整理Document键更窄；只含两库版本和工作文件哈希。
        # 重跑规则可复用真实Docling结果，但改变未入键的模型选项需要额外失效处理。
        raw_key={k:config[k] for k in ("docling","docling_core","working_sha256")}
        if not force and rawpath.exists() and rawmeta.exists() and read_json(rawmeta)==raw_key:
            from docling_core.types.doc import DoclingDocument
            doc=DoclingDocument.model_validate(read_json(rawpath))
            parse_mode="真实解析缓存重放"
        else:
            result=converters[ocr].convert(ROOT/m["working_path"],page_range=(w,w))
            doc=result.document
            write_json(rawpath,doc.export_to_dict());write_json(rawmeta,raw_key)
            parse_mode="首次/重新实际解析"
        # 来源锚点以原页/对象位置描述出处；稳定ID另由源哈希/变体/对象位置生成。
        # 锚点用于跨变体追溯，不能因为有ID就认为文字正确。
        def prov(bbox,kind,pos):
            return Provenance(doc_id=m["doc_id"],source_anchor=f"{m['doc_id']}:p{orig}:{kind}:{pos}",
                variant_id=m["variant_id"],original_page=orig,working_page=w,bbox=bbox,
                page_size=(width,height),method="ocr" if ocr else "native")
        pageblocks=[]
        for i,item in enumerate(doc.texts):
            if not item.prov: continue
            box=top_left_bbox(item.prov[0].bbox,height)
            b=TextBlock(block_id=stable_id(m["source_sha256"],m["variant_id"],orig,"text",i),
                text=item.text,role=str(item.label.value),provenance=prov(box,"text",i))
            pageblocks.append(b)
        # 当前扫描输入全页无文字/表时停止，不打开原生对应页替测试补答案。
        if ocr and not any(b.text.strip() for b in pageblocks) and not doc.tables:
            raise ValueError("全页OCR未得到文字/表格；未使用原生页补答案")
        for ti,item in enumerate(doc.tables):
            if not item.prov: continue
            box=top_left_bbox(item.prov[0].bbox,height)
            before=[b for b in pageblocks if b.provenance.bbox[3]<=box[1]+2]
            near="\n".join(b.text for b in before[-5:])
            title=item.caption_text(doc).strip()
            if not title:
                candidates=[b.text for b in before[-4:] if b.role in ("section_header","title") or re.search(r"续表|经营指标|独立产量|数据表",b.text)]
                title=candidates[-1] if candidates else ""
                title=re.sub(r"[；;].*", "",title)
            # 单位逐级尝试：表上最近5块→表前两行→本页文本或同一扫描页局部OCR。
            # 标题/邻近范围是启发式，同页存在多表时有串单位风险，须保留原图核对。
            unit=_unit(near)
            if unit=="unknown":
                unit=_unit("\n".join(c.text for c in item.data.table_cells if c.start_row_offset_idx<2))
            if unit=="unknown":
                if not ocr:
                    unit=_unit(page.get_text())
                else:
                    # Docling有时不保留表外短单位；只在当前扫描页裁剪补OCR，绝不访问原生对应页。
                    import easyocr
                    import numpy as np
                    if metadata_reader is None:
                        metadata_reader=easyocr.Reader(["ch_sim","en"],gpu=False,
                            model_storage_directory=str(ROOT/"models/easyocr"),download_enabled=False,verbose=False)
                    clip=fitz.Rect(0,max(0,box[1]-90),width,box[1]+12)
                    pix=page.get_pixmap(clip=clip,dpi=250)
                    image=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3]
                    unit_text="\n".join(metadata_reader.readtext(image,detail=0,paragraph=True))
                    unit=_unit(unit_text)
                    d.filter_log.append({"action":"metadata_ocr","page":w,"text":unit_text})
            # 显式人工单位覆盖单列日志，不伪装成自动解析。
            # 存在入口不代表当前材料用过它；是否触发以实际日志为准。
            override=m.get("unit_overrides",{}).get(str(orig))
            if override and unit!=override:
                unit=override
                d.filter_log.append({"action":"manual_unit_override","page":orig,"unit":override,
                    "reason":"独立核验的单位说明未被当前Docling/EasyOCR表格结构保留","automatic":False})
            cells=[]
            td=item.data
            header={}
            for cell in td.table_cells:
                if cell.start_row_offset_idx==0:
                    header[cell.start_col_offset_idx]=cell.text
            # 首行关键词与模型column_header共同判断表头，复杂多层表头仍有误判可能。
            header_heuristic=any(re.search(r"20\d{2}|项目|year|amount|millions?|total|price|quarter",v,re.I) for v in header.values())
            for ci,cell in enumerate(td.table_cells):
                cb=top_left_bbox(cell.bbox,height) if cell.bbox else box
                col=cell.start_col_offset_idx; row=cell.start_row_offset_idx
                period=header.get(col,"")
                # 页上有完整报表年度时用于金额期间；不会凭空赋新旧年份。
                # 无20xx但有列头且非第0列时回退文档年，是有限兜底。
                # 它不区分期末/期初、财年/自然年；同一个2024标签不能代替这些语义。
                if not re.search(r"20\d{2}",period):
                    period=m["year"] if period.strip() and col>0 else ""
                numeric=parse_number(cell.text,"dimensionless")
                is_header=(row==0 and header_heuristic) or (cell.column_header and numeric is None)
                cu="percent" if cell.text.strip().endswith("%") else unit
                # 结构先给row/col/span，表头/列0不填数值；金额再按本格%或表单位规范化。
                # 每格同时保存原文和来源，便于区分字符、错列、单位倍率三个错误。
                cells.append(Cell(cell_id=stable_id(m["source_sha256"],m["variant_id"],orig,"cell",ti,ci),
                    row=row,col=col,row_span=cell.row_span,col_span=cell.col_span,text=cell.text,
                    is_header=is_header,value=None if is_header or col==0 else parse_number(cell.text,cu),
                    unit=cu,period=period,provenance=prov(cb,"cell",f"{ti}.{ci}")))
            table_id=stable_id(m["source_sha256"],m["variant_id"],orig,"table",ti)
            tables.append(LogicalTable(table_id=table_id,title=title,unit=unit,periods=list(header.values()),
                cells=cells,rows=td.num_rows,cols=td.num_cols,fragments=[table_id],provenance=[prov(box,"table",ti)]))
        # 登记区域优先，避免扫描整页被Docling当作一个图片候选。
        # 优先用已登记原页框；人工裁剪登记的图不能计入自动检测成绩。
        # 没有登记的原生页才尝试Docling pictures，扫描整页图片不当成一个有效图候选。
        regions=[f for f in m.get("figure_regions",[]) if f["page"]==orig]
        if not regions and not ocr:
            for fi,item in enumerate(doc.pictures):
                if item.prov:
                    regions.append({"bbox":top_left_bbox(item.prov[0].bbox,height),
                        "title":item.caption_text(doc) or "报告图形", "automatic":True})
        for fi,region in enumerate(regions):
            box=tuple(region["bbox"]); figid=stable_id(m["source_sha256"],orig,"figure",fi)
            crop=ROOT/f"data/derived/figures/{figid}.png"; crop.parent.mkdir(exist_ok=True)
            page.get_pixmap(clip=fitz.Rect(box),dpi=180).save(crop)
            title=region["title"]
            figures.append(Figure(figure_id=figid,title=title,caption=title,crop_path=str(crop.relative_to(ROOT)),
                provenance=prov(box,"figure",fi),manual_crop=not region.get("automatic",False)))
            # 图内矢量文字不进入正文索引，避免图片题只靠文字层答对。
            # 只删除完全被图框包含的正文块，减少图片题靠文字层补答的泄漏。
            # 部分相交块仍可能保留，不能声称图中文字隔离毫无漏洞。
            pageblocks=[b for b in pageblocks if not fitz.Rect(box).contains(fitz.Rect(b.provenance.bbox))]
        blocks+=pageblocks
        print(f"{m['doc_id']} 工作页{w}/原页{orig} {'OCR' if ocr else '原生'}（{parse_mode}）：{len(pageblocks)}正文块，{len(doc.tables)}表，{time.monotonic()-start:.1f}s",flush=True)
    # 先用物理表框保护页边块，再拼逻辑表；保留元数据补OCR与页边日志的不同意义。
    # 解析审查报告记录实际比较，不把merged=True解读为整表金额全对。
    blocks,filterlog=filter_margins(blocks,tables,len(m["page_map"]))
    tables,stitchlog=stitch_tables(tables)
    metadata_log=d.filter_log
    d.blocks=blocks;d.tables=tables;d.figures=figures;d.filter_log=metadata_log+filterlog
    write_json(outpath,d.model_dump())
    write_json(ROOT/f"runs/parsing/{m['doc_id']}_{m['variant_id']}.json",{
        "mode":"actual_parse","manual_table_overrides":0,"scan_pages":len(m["scan_pages"]),
        "tables":len(tables),"blocks":len(blocks),"figures":len(figures),"filter_log":filterlog,
        "stitch_decisions":stitchlog,"config":config})
    pdf.close();gc.collect()
    return d


# 批量复用两个converter；native消融改用选页原生文件、清空扫描页，独立写native结果。
# 不把native结果回填mixed，否则无法测扫描退化。
def parse_all(doc_id=None,variant="mixed",force=False):
    manifests=read_json(ROOT/"data/corpus_manifest.json") if (ROOT/"data/corpus_manifest.json").exists() else read_json(ROOT/"data/generated_manifest.json")
    converters={False:make_converter(False),True:make_converter(True)}
    for m in manifests:
        if doc_id and m["doc_id"]!=doc_id: continue
        m=dict(m)
        if variant=="native":
            m["working_path"]=m.get("native_selected_path",m["original_path"])
            m["scan_pages"]=[];m["variant_id"]="native"
        parse_document(m,converters,force)
