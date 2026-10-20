# 原始资料准备：独立标注与原PDF位置核验→选页/扫描→两类题集。
# 会写正式资料与gold；标准来源独立于待测解析器，查看已有计数不需要重跑。
import re
from decimal import Decimal, localcontext
import fitz
from finrag.paths import ROOT
from finrag.store import read_json, read_jsonl, write_json, write_jsonl, sha256
from finrag.preparation import scan_variant, source_gold, retrieval_question, qa_gold
from finrag.numbers import parse_number


# 独立抄录的行名/新旧值必须在原文字层同一区域找到；不拿Docling输出当gold。
# 这只是位置检查，还需视觉核对单位/年份/科目，尤其同名行可能出现多次。
def row_region(page, label, value, previous):
    label_boxes=page.search_for(label)
    values=page.search_for(value)
    olds=page.search_for(previous)
    if not label_boxes or not values or not olds:
        raise ValueError("独立抄录值未在原始PDF核验："+label)
    matches=[]
    for lb in label_boxes:
        for vb in values:
            for ob in olds:
                dy=abs((lb.y0+lb.y1)/2-(vb.y0+vb.y1)/2)
                if dy<25 and abs(vb.y0-ob.y0)<4:
                    matches.append((dy,lb|vb|ob))
    if not matches: raise ValueError("原始数字与行名位置不一致："+label)
    rect=min(matches,key=lambda x:x[0])[1]
    return tuple(rect+(-2,-2,2,2))


def main():
    anns=read_json(ROOT/"configs/original_annotations.json")
    docs=read_json(ROOT/"data/generated_manifest.json")
    qs=read_jsonl(ROOT/"data/eval/generated_retrieval.jsonl")
    records=[];scanchecks=[]
    for doc_id,a in anns.items():
        path=ROOT/f"data/raw/{doc_id}.pdf"; source=fitz.open(path)
        pages=a["pages"]; native=ROOT/f"data/derived/{doc_id}_selected_native.pdf"
        scan_variant(path,native,[],pages)
        # source_type说明原始/重建/合成身份，scan_pages另说明处理模态。
        # 单位人工覆盖入口单独登记；它存在不等于每张表现有结果都使用了覆盖。
        doc={"doc_id":doc_id,"company":a["company"],"year":"2024","language":a["language"],"split":a["split"],
            "source_type":"original","source_group":doc_id,"original_path":str(path.relative_to(ROOT)),
            "source_sha256":sha256(path),"page_map":pages,"scan_pages":[pages.index(p)+1 for p in a["scan_original_pages"]],
            "native_selected_path":str(native.relative_to(ROOT)),"working_path":f"data/derived/{doc_id}_mixed.pdf",
            "variant_id":"mixed","figure_regions":a["figure_regions"],
            "unit_overrides":{str(a["lookups"][0][0]):a["lookups"][0][4]} if doc_id in ("microsoft2024","apple2024") else {},
            "source_note":"公开原始PDF镜像，未核验与官方文件逐字节一致" if doc_id=="microsoft2024" else "公司发布的原始PDF"}
        scanchecks+=scan_variant(path,ROOT/doc["working_path"],doc["scan_pages"],pages)
        docs.append(doc);evidence=[]
        out=ROOT/f"runs/gold_verification/{doc_id}";out.mkdir(parents=True,exist_ok=True)
        for j,(p,label,new,old,unit) in enumerate(a["lookups"]):
            page=source[p-1];box=row_region(page,label,new,old)
            crop=out/f"row{j}.png";page.get_pixmap(clip=fitz.Rect(box)+(-4,-4,4,4),dpi=200).save(crop)
            e=source_gold(doc_id,p,box,"row",numeric=parse_number(new,unit),label=label)
            evidence.append(e)
            records.append({"doc_id":doc_id,"page":p,"label":label,"raw_new":new,"raw_old":old,"unit":unit,
                "new_canonical":parse_number(new,unit),"old_canonical":parse_number(old,unit),"bbox":box,
                "crop_path":str(crop.relative_to(ROOT)),"verification":"AI独立抄录 + 原始文字位置核对 + 页面/裁剪视觉核验，非待测Docling输出"})
            qa=qa_gold(e["value"],"RMB" if doc_id=="catl2024" else "USD","2024") if j<3 else None
            query=f"{a['company']}2024年{label}是多少？请保留{unit}单位。"
            tags=["lookup",a["language"],"scan" if p in a["scan_original_pages"] else "native"]
            qs.append(retrieval_question(f"O-{doc_id}-{j}",f"{doc_id}:{label}",a["split"],"original",doc,query,[e],tags,qa))
        # 增長率题来源为独立抄录，不把待测单元格作gold。
        for k in range(1 if doc_id=="microsoft2024" else 3):
            p,label,new,old,unit=a["lookups"][k]
            with localcontext() as ctx:
                ctx.prec=50;v=(Decimal(new.replace(",",""))-Decimal(old.replace(",","")))/Decimal(old.replace(",",""))*100
            olde=dict(evidence[k]);olde["value"]=parse_number(old,unit)
            query=f"{a['company']}2024年{label}相对2023年的增长率是多少百分比？"
            gold=[evidence[k],olde]
            qa=qa_gold(v,"percent","2024",expression="(new-old)/old*100")
            qs.append(retrieval_question(f"O-{doc_id}-growth{k}",f"{doc_id}:{label}",a["split"],"original",doc,query,gold,
                ["calculation",a["language"],"scan" if p in a["scan_original_pages"] else "native"],qa))
        # 图片gold仅依视觉核验事实；图注不重复答案。
        figure=a["figure_regions"][0];fige=source_gold(doc_id,figure["page"],figure["bbox"],"figure")
        if doc_id=="microsoft2024":
            images=[("微软五年累计总回报图在2024年末哪一条曲线最高？","Microsoft",["Microsoft","微软"])]
        elif doc_id=="apple2024":
            images=[("苹果五年累计总回报折线图中，2022年相较2021年下降的是哪两条指数曲线？","S&P 500 和 Dow Jones",["S&P 500","Dow Jones"])]
        else:
            images=[("宁德时代产品应用图中的创新解决方案列，换电方案的名称是什么？","巧克力换电",["巧克力换电"]),
                ("宁德时代产品应用图，动力电池乘用车化学体系有哪些标签？","麒麟、神行、骁遥",["麒麟","神行","骁遥"])]
        for j,(query,ans,syn) in enumerate(images):
            qs.append(retrieval_question(f"O-{doc_id}-image{j}",f"{doc_id}:figure{j}",a["split"],"original",doc,query,[fige],
                ["image",a["language"],"native"],qa_gold(ans,"text","2024","text",synonyms=syn)))
        # 开发补一个正文QA，测试来源补正文检索，取原始业务介绍。
        p=11 if doc_id=="microsoft2024" else 4 if doc_id=="apple2024" else 13
        body=source_gold(doc_id,p,(40,60,570,700),"text",label="business")
        query=f"{a['company']}2024年报告如何介绍其主要业务？"
        if doc_id=="microsoft2024":
            query="微软2024年报对前瞻性陈述的不确定性作了什么提醒？"
            qa=qa_gold("风险和不确定性可能使实际结果与预期不同","text","2024","text",synonyms=["风险","不确定"])
        elif doc_id=="apple2024":
            query="苹果2024年报业务介绍列出的手机、个人电脑和平板电脑产品名称是什么？"
            qa=qa_gold("iPhone、Mac、iPad","text","2024","text",synonyms=["iPhone","Mac","iPad"])
        else:
            query="宁德时代2024年报主要业务中的两类电池是什么？"
            qa=qa_gold("动力电池、储能电池","text","2024","text",synonyms=["动力电池","储能电池"])
        qs.append(retrieval_question(f"O-{doc_id}-text",f"{doc_id}:business",a["split"],"original",doc,query,[body],
            ["text",a["language"],"scan"],qa))
        # 原始测试各18题：独立已核验字段的术语改述，与原题同group/split。
        target=12 if doc_id=="microsoft2024" else 18
        existing=sum(q["doc_id"]==doc_id for q in qs)
        for k in range(target-existing):
            e=evidence[k+3];label=a["lookups"][k+3][1]
            query=f"请在{a['company']}年报找到2024年度{label}这一行及相邻年份，给出来源。"
            qs.append(retrieval_question(f"O-{doc_id}-paraphrase{k}",f"{doc_id}:{label}",a["split"],"original",doc,query,[e],
                ["multi_evidence",a["language"],"scan" if e["original_page"] in a["scan_original_pages"] else "native"],None))
        source.close()
    # 从带问答标准的检索题拆出QA记录，检索题和问答题分文件保存，不合并计为独立题。
    qa=[]
    for q in qs:
        g=q.pop("qa",None)
        if g: qa.append({**q,**g})
    # Apple/CATL各9道QA：3查数+3计算+1/2图=7/8，加独立查数字段补至9。
    for doc_id in ("apple2024","catl2024"):
        count=sum(q["doc_id"]==doc_id for q in qa)
        for q in [q for q in qs if q["doc_id"]==doc_id and "lookup" in q["tags"]][3:3+9-count]:
            qa.append({**q,**qa_gold(q["gold_evidence"][0]["value"],"RMB" if doc_id=="catl2024" else "USD","2024")})
    write_json(ROOT/"data/corpus_manifest.json",docs)
    write_jsonl(ROOT/"data/eval/retrieval_gold.jsonl",qs)
    write_jsonl(ROOT/"data/eval/qa_gold.jsonl",qa)
    write_json(ROOT/"runs/original_scan_checks.json",scanchecks)
    write_json(ROOT/"data/eval/original_independent_verification.json",records)
    print(f"完整语料{len(docs)}份，检索题{len(qs)}，问答题{len(qa)}，原始扫描{len(scanchecks)}页")
    print('来源/split数量', {(s,sp):sum(q['source_type']==s and q['split']==sp for q in qs) for s in ['original','reconstructed','synthetic'] for sp in ['dev','test']})
    assert len(qs)==128 and len(qa)==64


if __name__=="__main__":main()
