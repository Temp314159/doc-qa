# 检索表示层：从Document结构构造View与Evidence，原始逻辑表仍在解析文件中。
# 视图负责检索表示，证据负责引用和计算绑定，两者保留关联。
import re
from collections import defaultdict
from .paths import ROOT
from .store import read_json, write_json, stable_id
from .schema import Document, View, Evidence


# 输入指定变体的已解析Document，输出视图列表，并写视图/证据/图片登记文件。
# output_dir可隔离实验产物；本函数会写文件，查看旧产物不需要再次调用。
def build_views(variant="mixed", doc_ids=None, output_dir=None):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/"models/qwen3-embedding-0.6b",local_files_only=True)
    manifests=read_json(ROOT/"data/corpus_manifest.json")
    views=[];evidence=[];figures={}
    for m in manifests:
        if doc_ids is not None and m['doc_id'] not in doc_ids:continue
        path=ROOT/f"data/parsed/{m['doc_id']}_{variant}.json"
        if not path.exists():raise FileNotFoundError("缺少实际解析结果："+str(path))
        doc=Document.model_validate(read_json(path))
        # 正文按页面和段落组合，目标512模型token；长段落50token重叠。
        # 先为每个正文块保留整块Evidence，再按原页组织正文视图。
        # 长块切成多个窗口后仍引用同一个原块/框，没有重新检测每窗口的逐词坐标。
        grouped=defaultdict(list)
        for b in doc.blocks:
            e=Evidence(evidence_id=b.block_id,source_anchor=b.provenance.source_anchor,doc_id=doc.doc_id,
                parent_id=b.block_id,content=b.text,provenance=[b.provenance])
            evidence.append(e)
            grouped[b.provenance.original_page].append(b)
        for page,blocks in grouped.items():
            chunks=[];current=[];length=0
            for b in blocks:
                # 按Qwen模型token计长，与BM25的jieba词项不同。
                # 不加特殊token，只计块文字；最后新增公司/年度前缀没有纳入这个组合目标。
                ids=tokenizer.encode(b.text,add_special_tokens=False)
                # 512是组合目标，768是单块切窗触发阈值。
                # 单块600或恰好768会完整保留；769才切成512宽、462步长（50重叠）。
                # 先收好current，防止把已组合短块与当前长块窗口混在一起。
                if len(ids)>768:
                    if current:chunks.append(current);current=[];length=0
                    for start in range(0,len(ids),462):
                        chunks.append([(b,tokenizer.decode(ids[start:start+512]))])
                else:
                    # 已有内容且加新块超目标时先封包。空current则允许一个不超过768的块单独超512。
                    # 因此“所有chunk至多512”并不是当前实际行为；同一原页内才组合。
                    if length+len(ids)>512 and current:
                        chunks.append(current);current=[];length=0
                    current.append((b,b.text));length+=len(ids)
            if current:chunks.append(current)
            for i,chunk in enumerate(chunks):
                viewid=stable_id(doc.source_sha256,variant,"textview",page,i)
                # 公司/年度帮助搜索区分来源；最终长度可能比先前累计量多一些。
                # parent_id指一个来源块，evidence_ids列出全部参与块，不代表自动父子扩展。
                content=f"公司：{doc.company}；年度：{doc.year}。\n"+"\n".join(t for b,t in chunk)
                views.append(View(view_id=viewid,kind="text",doc_id=doc.doc_id,text=content,parent_id=chunk[0][0].block_id,
                    evidence_ids=[b.block_id for b,t in chunk],anchors=[b.provenance.source_anchor for b,t in chunk],
                    provenance=[b.provenance for b,t in chunk]))
        for t in doc.tables:
            # 完整逻辑表按Cell.row分组，随后每行按col排列；不依OCR全文的字符顺序猜行。
            # 跨页拼接已经在ingestion完成，Cell的物理原页与这里的逻辑行号可不同。
            cells=defaultdict(list)
            for c in t.cells:cells[c.row].append(c)
            headers=defaultdict(list)
            for c in t.cells:
                if c.is_header:
                    # 一个跨列表头要覆盖它横跨的各列；同列多项标题去重后用“ / ”连起来。
                    # 这是确定性的结构展开，正确与否仍依赖解析器给的header/span。
                    for col in range(c.col,c.col+c.col_span):headers[col].append(c.text)
            header={col:" / ".join(dict.fromkeys(values)) for col,values in headers.items()}
            rowlabels=[]
            for row,cs in sorted(cells.items()):
                cs=sorted(cs,key=lambda c:c.col)
                if all(c.is_header for c in cs):continue
                # 列0作为科目名，缺名时用行号兜底；整行都是表头就不生成数据行视图。
                # 科目、年份、数值、单位要一起呈现，避免单独金额失去财务含义。
                label=" / ".join(c.text for c in cs if c.col==0).strip()
                if not label:label=f"第{row+1}行"
                rowlabels.append(label)
                relevant=[c for c in cs if not c.is_header]
                row_id=stable_id(doc.source_sha256,variant,t.table_id,row,"row")
                # structural是紧凑键值，natural是完整分句，用于比较不同检索表示。
                # 二者从同一批Cell取原文字，由固定模板拼接，没有LLM改写或新增事实。
                # natural每格携带自己的单位/期间；structural在行首提供表单位，表达并非逐字符相同。
                structural=f"company={doc.company}; table={t.title}; row={label}; unit={t.unit}; " + "; ".join(f"{header.get(c.col,c.period)}={c.text}" for c in relevant if c.col>0)
                natural=f"{doc.company}的{t.title or '财务表'}，{label}："+"；".join(
                    f"{header.get(c.col,c.period) or '列'+str(c.col)}为{c.text}（单位{c.unit}，期间{c.period or doc.year}）" for c in relevant if c.col>0)+"。"
                # 保留完整行，超2048token只停稠密，稀疏仍可搜索。
                # 长度检查实际以natural文本为准，超过2048保留全文并标记。
                # 后续建索引写零向量，不是这里删掉Cell；当前检索没有完全排除零向量候选。
                sparse_only=len(tokenizer.encode(natural,add_special_tokens=False))>2048
                # 整行证据没有唯一数值（默认value=None），用于说明关系，不能直接当计算变量。
                # 逐格Evidence才复制Cell.value；微软收入行因此关联行、行名格、三年金额格共5项。
                row_e=Evidence(evidence_id=row_id,source_anchor=relevant[0].provenance.source_anchor,doc_id=doc.doc_id,
                    parent_id=t.table_id,content=natural,unit=t.unit,period=doc.year,provenance=[c.provenance for c in relevant])
                evidence.append(row_e)
                for c in relevant:
                    evidence.append(Evidence(evidence_id=c.cell_id,source_anchor=c.provenance.source_anchor,doc_id=doc.doc_id,
                        parent_id=t.table_id,content=f"{label} | {header.get(c.col,c.period)} | {c.text}",value=c.value,
                        unit=c.unit,period=c.period,provenance=[c.provenance]))
                views.append(View(view_id=row_id,kind="row",doc_id=doc.doc_id,text=natural,structural_text=structural,
                    parent_id=t.table_id,evidence_ids=[row_id]+[c.cell_id for c in relevant],
                    anchors=[c.provenance.source_anchor for c in relevant],provenance=[c.provenance for c in relevant],sparse_only=sparse_only))
            # 表摘要列出表名、表头和科目入口，没有全部金额。
            # 命中摘要不等于拿到数值行；只关联自身Evidence，当前不会自动展开父表。
            summary=f"{doc.company} {doc.year} {t.title} 单位{t.unit} 表头{' '.join(header.values())} 项目{'、'.join(rowlabels)}"
            evidence.append(Evidence(evidence_id=t.table_id,source_anchor=t.provenance[0].source_anchor,doc_id=doc.doc_id,
                parent_id=t.table_id,content=summary,unit=t.unit,period=doc.year,provenance=t.provenance))
            views.append(View(view_id=t.table_id,kind="table",doc_id=doc.doc_id,text=summary,parent_id=t.table_id,
                evidence_ids=[t.table_id],anchors=[p.source_anchor for p in t.provenance],provenance=t.provenance))
        # 图片只索引标题/说明文字，Qwen不直接编码像素。
        # figures字典登记实际裁剪路径，后续inspect_figure才发送图片读取内部内容。
        for f in doc.figures:
            content=f"{doc.company} {doc.year} 图像：{f.title}；{f.caption}"
            evidence.append(Evidence(evidence_id=f.figure_id,source_anchor=f.provenance.source_anchor,doc_id=doc.doc_id,
                parent_id=f.figure_id,content=content,unit="text",period=doc.year,provenance=[f.provenance]))
            views.append(View(view_id=f.figure_id,kind="figure",doc_id=doc.doc_id,text=content,parent_id=f.figure_id,
                evidence_ids=[f.figure_id],anchors=[f.provenance.source_anchor],provenance=[f.provenance]))
            figures[f.figure_id]=f.model_dump()
    out=output_dir or ROOT/f"data/indexes/{variant}";out.mkdir(parents=True,exist_ok=True)
    # 三个文件分工：views给搜索器；evidence按ID给绑定/引用；figures给受控视觉工具。
    # 这些关联使改变表示不必改变事实，但正式重建仍须同步文本和向量版本。
    write_json(out/"views.json",[v.model_dump() for v in views])
    write_json(out/"evidence.json",{e.evidence_id:e.model_dump() for e in evidence})
    write_json(out/"figures.json",figures)
    print(f"检索视图{len(views)}，证据{len(evidence)}，图片{len(figures)}")
    return views
