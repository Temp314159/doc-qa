# 资料准备层：用独立源数据生成可控PDF/扫描变体/题集，不能混称为原始财报。
# 生成可控布局以验证解析规则；重跑会写数据和题集。
"""从独立源数据生成可控布局；gold不进入工作PDF。"""
import io
import re
from decimal import Decimal, localcontext
import fitz
from PIL import Image, ImageDraw
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from .paths import ROOT
from .store import read_json, write_json, write_jsonl, sha256, stable_id
from .numbers import parse_number

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
PAGE = (612, 792)
STYLE = ParagraphStyle("cell", fontName="STSong-Light", fontSize=10, leading=14)


def paragraph(text):
    return Paragraph(str(text).replace("&", "&amp;").replace("<", "&lt;"), STYLE)


# ReportLab以左下为原点，来源框以左上为原点；绘制时用792-y转换。
# 返回整个表矩形供独立gold定位，不从待测Docling结构倒推出标准答案。
def table_draw(c, rows, y_top, widths=None):
    widths = widths or [280] + [240 / (len(rows[0]) - 1)] * (len(rows[0]) - 1)
    table = Table([[paragraph(x) for x in row] for row in rows], colWidths=widths)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .6, colors.HexColor("#8297AB")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE9F2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    _, h = table.wrap(520, 700)
    table.drawOn(c, 46, 792 - y_top - h)
    return (46, y_top, 566, y_top + h)


def text(c, x, y_top, content, size=11):
    c.setFont("STSong-Light", size)
    c.drawString(x, 792 - y_top, content)


def footer(c, company, page):
    text(c, 46, 774, f"{company} | 财报教学示例 | Page {page}", 9)


# selected给原页顺序，pages给工作页序号：两种页号不能混用。
# 扫描页先渲染成像素再插入空白PDF页，其余页原样复制，并检查扫描文字层确实为空。
# 页面尺寸保留points，dpi改变像素密度，不改变page_map。
def scan_variant(original, target, pages, selected=None, dpi=250):
    source = fitz.open(original)
    out = fitz.open()
    selected = selected or list(range(1, len(source) + 1))
    checks = []
    for work, original_page in enumerate(selected, 1):
        p = source[original_page - 1]
        if work in pages:
            pix = p.get_pixmap(dpi=dpi, alpha=False)
            q = out.new_page(width=p.rect.width, height=p.rect.height)
            q.insert_image(q.rect, stream=pix.tobytes("png"))
            checks.append({"working_page": work, "original_page": original_page,
                           "text_layer_empty": not q.get_text().strip(), "dpi": dpi})
        else:
            out.insert_pdf(source, from_page=original_page-1, to_page=original_page-1)
    out.save(target, deflate=True)
    out.close()
    source.close()
    if not all(x["text_layer_empty"] for x in checks):
        raise ValueError("扫描变体含文字层")
    return checks


# 标准证据保存doc/原页/矩形/类型/独立值，不依赖解析生成的evidence_id。
# 这样解析读成G00时，gold仍保留源值，避免解析器给自己出答案。
def source_gold(doc_id, page, bbox, kind, text_value="", numeric=None, label=""):
    return {"doc_id": doc_id, "source_anchor": f"{doc_id}:p{page}:{kind}:{stable_id(label,numeric,text_value)}", "original_page": page, "bbox": list(bbox), "kind": kind,
            "text": text_value, "value": numeric, "label": label}


def retrieval_question(qid, group, split, source_type, doc, query, evidence, tags, qa=None):
    return {"question_id": qid, "question_group": group, "split": split, "source_type": source_type,
        "source_group": doc["source_group"], "doc_id": doc["doc_id"], "query": query,
        "tags": tags, "gold_evidence": evidence, "allowed_equivalents": [],
        "necessary_evidence": list(range(len(evidence))), "qa": qa}


# 问答标准另含值、单位、期间和参考算式；这些是评测字段，不进入工作PDF。
def qa_gold(value, unit, period, answer_type="numeric", operands=None, expression=None, synonyms=None):
    return {"answer_type": answer_type, "gold_value": str(value) if answer_type == "numeric" else None,
        "gold_text": str(value) if answer_type != "numeric" else None, "unit": unit,
        "period": period, "precision": 2, "rounding": "ROUND_HALF_UP", "absolute_tolerance": "0.005",
        "relative_tolerance": "0.00001", "gold_operand_evidence": operands or [],
        "reference_expression": expression, "synonyms": synonyms or [str(value)], "online_subset": False}


# 纯合成公司与金额明确贴synthetic；可控跨页/重复表头/页脚可测规则，但不能替代真实复杂财报。
def synthetic(index):
    doc_id = f"synthetic{index}"
    company = ["澄明科技", "星河电池", "远帆软件", "青禾制造"][index]
    base = 100 + index * 30
    rows = [["项目", "2024", "2023"], ["营业收入", str(base*12), str(base*10)],
        ["营业成本", str(base*7), str(base*6)], ["研发费用", str(base*2), str(base)],
        ["销售费用", str(base), str(base//2)], ["现金余额", str(base*4), str(base*3)],
        ["短期借款", str(base*3), str(base*2)], ["其他收益", "(15)", "12"], ["净利润", str(base*3), str(base*2)]]
    path = ROOT / f"data/derived/{doc_id}_native.pdf"
    c = Canvas(str(path), pagesize=PAGE, invariant=1)
    c.setTitle(f"{company}纯合成2024财报")
    text(c, 46, 54, f"{company}2024年度报告（纯合成）", 17)
    text(c, 46, 94, "主营业务为设备研发与销售；本报告仅用于教学和测试。")
    text(c, 46, 136, "合并经营指标；单位：人民币百万元")
    box1 = table_draw(c, rows[:5], 420)
    text(c, 46, 731, "注：金额按人民币百万元列示，括号代表负数。", 9)
    footer(c, company, 1)
    c.showPage()
    text(c, 46, 56, "合并经营指标（续表）；单位：人民币百万元")
    box2 = table_draw(c, [rows[0]] + rows[5:], 80)
    text(c, 46, 405, "独立产量表；单位：件（不能与经营指标合并）")
    table_draw(c, [["项目", "2024", "2023"], ["产品数量", "900", "800"]], 430)
    footer(c, company, 2)
    c.showPage()
    text(c, 46, 54, "地区销售分布图（纯合成图像）", 16)
    image = Image.new("RGB", (900, 450), "white")
    draw = ImageDraw.Draw(image)
    chart_values = [42+index*3, 28+index*2, 18+index]
    for i, (name, v) in enumerate(zip(["EAST", "WEST", "NORTH"], chart_values)):
        x = 90 + i * 265
        draw.rectangle((x, 350-v*5, x+135, 350), fill=["#2166AC", "#B2182B", "#238B45"][i])
        draw.text((x, 380), name, fill="black", font_size=30)
        draw.text((x+25, 350-v*5-45), str(v), fill="black", font_size=34)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    from reportlab.lib.utils import ImageReader
    c.drawImage(ImageReader(buf), 46, 260, width=520, height=260)
    text(c, 46, 580, "图内包含各地区销售占比；图注不复述图内数值。")
    footer(c, company, 3)
    c.save()
    doc = {"doc_id": doc_id, "company": company, "year": "2024", "language": "zh",
        "split": "dev" if index == 0 else "test", "source_type": "synthetic", "source_group": doc_id,
        "original_path": str(path.relative_to(ROOT)), "source_sha256": sha256(path), "page_map": [1,2,3],
        "scan_pages": [1,2] if index < 2 else [], "working_path": f"data/derived/{doc_id}_mixed.pdf",
        "variant_id": "mixed", "figure_regions": [{"page":3,"bbox":[46,272,566,532],"title":"地区销售分布图"}]}
    checks = scan_variant(path, ROOT/doc["working_path"], doc["scan_pages"])
    qs = []
    evidences = []
    for row in (1,2,3,5,8):
        page, bbox = (1,box1) if row <=4 else (2,box2)
        e = source_gold(doc_id,page,bbox,"row",numeric=parse_number(rows[row][1],"人民币百万元"),label=rows[row][0])
        evidences.append(e)
    queries = [f"{company}2024年{rows[r][0]}是多少人民币百万元？" for r in (1,2,3,5,8)]
    qas = [qa_gold(parse_number(rows[1][1],"人民币百万元"),"RMB","2024"), None,None,None,None]
    # QA四题：查数、跨页合计、两道看图；确保图片答案只在像素中。
    queries += [f"{company}2024年营业收入与现金余额合计是多少？",f"{company}地区销售分布图EAST的标注占比是多少？",f"{company}地区销售分布图WEST的标注占比是多少？"]
    refs = [[e] for e in evidences] + [[evidences[0],evidences[3]],
        [source_gold(doc_id,3,(46,272,566,532),"figure")],[source_gold(doc_id,3,(46,272,566,532),"figure")]]
    qas += [qa_gold((base*12+base*4)*1000000,"RMB","2024",expression="a+b"),
            qa_gold(chart_values[0],"percent","2024"),qa_gold(chart_values[1],"percent","2024")]
    for j, (query, gold, qa) in enumerate(zip(queries,refs,qas)):
        tags = ["image" if j>=6 else "calculation" if j==5 else "lookup", "zh"]
        tags += ["scan" if any(e["original_page"] in doc["scan_pages"] for e in gold) else "native"]
        qs.append(retrieval_question(f"S{index}-{j}",doc_id,doc["split"],"synthetic",doc,query,gold,tags,qa))
    return doc, qs, checks


# 只接受能还原并与FinQA已有答案核对的四则程序，操作数需能在源表定位。
# 不支持的程序跳过；这是参考核验，不是运行时任意代码执行器。
def _finqa_reference(record):
    program = record["qa"].get("program", "")
    instructions = re.findall(r"(add|subtract|multiply|divide)\(([^()]*)\)", program)
    if not instructions or ", ".join(f"{op}({args})" for op,args in instructions) != program.strip():
        return None
    numeric = {}
    for row, cells in enumerate(record["table"][1:],1):
        for col, value in enumerate(cells[1:],1):
            n = parse_number(value,"dimensionless")
            if n is not None:
                numeric.setdefault(Decimal(n), (row,col))
    values, expressions, operands = [], [], []
    with localcontext() as ctx:
        ctx.prec=50
        for op, args in instructions:
            tokens=args.split(",")
            if len(tokens)!=2:
                return None
            vs, es = [], []
            for token in tokens:
                token=token.strip()
                if token.startswith("#"):
                    k=int(token[1:])
                    if k>=len(values): return None
                    vs.append(values[k]); es.append(expressions[k])
                elif token.startswith("const_"):
                    try: n=Decimal(token[6:].replace("_","."))
                    except Exception: return None
                    vs.append(n); es.append(str(n))
                else:
                    n=parse_number(token,"dimensionless")
                    if n is None or Decimal(n) not in numeric:
                        return None
                    n=Decimal(n); pos=numeric[n]
                    if pos not in operands: operands.append(pos)
                    vs.append(n); es.append(chr(97+operands.index(pos)))
            if op=="divide" and vs[1]==0: return None
            v=vs[0]+vs[1] if op=="add" else vs[0]-vs[1] if op=="subtract" else vs[0]*vs[1] if op=="multiply" else vs[0]/vs[1]
            values.append(v); expressions.append(f"({es[0]}{dict(add='+',subtract='-',multiply='*',divide='/')[op]}{es[1]})")
    existing=str(record["qa"].get("exe_ans",""))
    try:
        if abs(values[-1]-Decimal(existing))>max(Decimal("0.0001"),abs(values[-1])*Decimal("0.0001")):
            return None
    except Exception:
        return None
    return {"value":str(values[-1]),"expression":expressions[-1],"operands":operands}


# 选择有限布局/单位范围的12条可核验数据，重排成PDF并标reconstructed。
# 原始数字真实不意味着版面来自原年报；不能拿这种材料证明通用原始解析能力。
def finqa_documents():
    records = read_json(ROOT/"data/raw/finqa_train.json") + read_json(ROOT/"data/raw/finqa_test.json")
    selected=[]; groups=set()
    for rec in records:
        table=rec["table"]
        if not 4<=len(table)<=12 or not 2<=len(table[0])<=4 or any(len(r)!=len(table[0]) for r in table): continue
        if any(len(" ".join(map(str,r)))>160 for r in table): continue
        if not any("million" in str(x).lower() for r in table for x in r): continue
        if any("shares" in str(x).lower() or "except rates" in str(x).lower() for r in table for x in r): continue
        ref=_finqa_reference(rec)
        group="/".join(rec["id"].split("/")[:2])
        if not ref or group in groups: continue
        selected.append((rec,ref)); groups.add(group)
        if len(selected)==12: break
    if len(selected)!=12: raise ValueError("可核验FinQA记录不足12")
    docs=[]; questions=[]; checks=[]; independent=[]
    for i,(rec,ref) in enumerate(selected):
        doc_id=f"finqa{i:02d}"; company=rec["id"].split("/")[0]; split="dev" if i<3 else "test"
        table=rec["table"]; mid=max(2,len(table)//2)
        path=ROOT/f"data/derived/{doc_id}_native.pdf"
        c=Canvas(str(path),pagesize=PAGE,invariant=1); c.setTitle(rec["id"]+" reconstructed")
        text(c,46,48,f"{company} - 真实数据重建（非原始报告）",15)
        text(c,46,82,"来源：FinQA公开记录；单位：USD million；表内百分号单独按百分比")
        text(c,46,114,rec["id"],9)
        b1=table_draw(c,table[:mid],350)
        footer(c,company,1); c.showPage()
        text(c,46,52,"FinQA数据表（续表）；单位：USD million")
        b2=table_draw(c,[table[0]]+table[mid:],78)
        # 原上下文只保存不含问题/答案/程序的段落；有限长度避免版面溢出。
        y= b2[3]+35
        for line in rec["post_text"][:2]:
            para=paragraph(line); w,h=para.wrap(520,200)
            if y+h>740: break
            para.drawOn(c,46,792-y-h); y+=h+10
        footer(c,company,2); c.save()
        doc={"doc_id":doc_id,"company":company,"year":rec["id"].split("/")[1],"language":"en","split":split,
            "source_type":"reconstructed","source_group":"/".join(rec["id"].split("/")[:2]),"record_id":rec["id"],
            "original_path":str(path.relative_to(ROOT)),"source_sha256":sha256(path),"page_map":[1,2],
            "scan_pages":[1,2] if i in (0,3) else [],"working_path":f"data/derived/{doc_id}_mixed.pdf","variant_id":"mixed","figure_regions":[]}
        checks+=scan_variant(path,ROOT/doc["working_path"],doc["scan_pages"])
        docs.append(doc)
        def ev(pos):
            r,col=pos; page,bbox=(1,b1) if r<mid else (2,b2)
            return source_gold(doc_id,page,bbox,"row",numeric=parse_number(table[r][col],"USD million"),label=table[r][0])
        op_evidence=[ev(pos) for pos in ref["operands"]]
        # 原题按其参考程序语义使用ratio/金额；百分数题答案保留原FinQA输出口径。
        ratio=any(x in rec["qa"]["question"].lower() for x in ("percent","rate","ratio","proportion"))
        canonical=str(Decimal(ref["value"])*(1 if ratio else Decimal("1e6")))
        original_qa=qa_gold(canonical,"dimensionless" if ratio else "USD",doc["year"],expression=ref["expression"])
        positions=[]
        for r in range(1,len(table)):
            for col in range(1,len(table[r])):
                if parse_number(table[r][col],"dimensionless") is not None: positions.append((r,col))
        p=positions[0]; e=ev(p)
        lookup=f"{company} {doc['year']} FinQA表中 {table[p[0]][0]} 的 {table[0][p[1]]} 原值是多少（单位USD million）？"
        last=positions[-1]
        entries=[(rec["qa"]["question"],op_evidence,original_qa,"calculation"),
          (lookup,[e],qa_gold(e["value"],"USD",doc["year"]),"lookup"),
          (f"{company} {doc['year']} find {table[last[0]][0]} {table[0][last[1]]}",[ev(last)],None,"lookup"),
          (f"{company} {doc['year']} 查找计算所需的表格证据：{rec['qa']['question']}",op_evidence,None,"multi_evidence")]
        for j,(query,gold,qa,task) in enumerate(entries):
            tags=[task,"en","scan" if any(e["original_page"] in doc["scan_pages"] for e in gold) else "native"]
            questions.append(retrieval_question(f"F{i:02d}-{j}",rec["id"],split,"reconstructed",doc,query,gold,tags,qa))
        independent.append({"record_id":rec["id"],"source_group":doc["source_group"],"split":split,
            "table":table,"qa":rec["qa"],"reference_check":ref,"renderer":"reportlab 4.4.10", "seed":20261012})
    write_json(ROOT/"data/eval/finqa_independent_source.json",independent)
    return docs,questions,checks


# 聚合合成与重建数据，写生成清单和独立题集；原始公司资料由prepare_corpus另接入。
def prepare_generated():
    # 发布源码不携带派生PDF；首次准备时先创建生成目标目录。
    (ROOT / "data/derived").mkdir(parents=True, exist_ok=True)
    docs=[]; questions=[]; checks=[]
    for i in range(4):
        doc,qs,cs=synthetic(i); docs.append(doc); questions+=qs; checks+=cs
    fs,fq,fc=finqa_documents(); docs+=fs; questions+=fq; checks+=fc
    write_json(ROOT/"data/generated_manifest.json",docs)
    write_jsonl(ROOT/"data/eval/generated_retrieval.jsonl",questions)
    write_json(ROOT/"runs/generated_scan_checks.json",checks)
    print(f"生成{len(docs)}份工作PDF，独立题目{len(questions)}，扫描页{len(checks)}")
    return docs
