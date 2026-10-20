# 检索层：将查询变成分数与排序，再返回视图及其证据。
# BM25、Qwen余弦和融合共用候选视图；这里不生成答案，也不验证金额真实性。
import gc
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
import jieba
import numpy as np
from .paths import ROOT
from .store import read_json, write_json, sha256
from .numbers import numeric_tokens

# 词典在模块导入时就加入jieba：基础/增强两组都受这些财务词影响。
# 所以增强开关不是“有没有金融词典”，而是下面是否追加同义词与数值锚点。
LEXICON=read_json(ROOT/"configs/financial_lexicon.json")
jieba.dt.tmp_dir=str(ROOT/".cache")
for term in LEXICON["terms"]:jieba.add_word(term)
STOP={"的","是","多少","什么","请","年","公司","单位","为","and","the","in","of","a","to"}


# 输入普通文字，输出BM25词项列表，保留重复以便Counter统计tf/qtf。
# 这里的token是检索词项，不是Qwen tokenizer产生的模型整数ID。
def tokenize(text,enhanced=True):
    # 先统一部分全角/兼容字符并小写，使词面统计少受排版和大小写影响。
    # HMM=False关闭jieba的隐马尔可夫新词推断；正则留下文字词项，STOP去部分功能词。
    text=unicodedata.normalize("NFKC",text).lower()
    tokens=[t for t in jieba.lcut(text,HMM=False) if re.fullmatch(r"[\w\u4e00-\u9fff]+",t) and t not in STOP]
    # 同义追加只做子串匹配，不理解财务语义，也不去重；重复会改变词频。
    # numeric_tokens只根据文字中紧邻数字的单位生成锚点，不继承远处表格的单位。
    if enhanced:
        for key,values in LEXICON["synonyms"].items():
            if key in text:tokens+=values.split()
        tokens+=numeric_tokens(text)
    return tokens


# 构造时准备倒排与语料统计；score时沿查询词的倒排列表累计分。
# 本项目候选是视图，因此N数视图而非PDF文件数；改变分块/模板也会改变统计。
class BM25:
    def __init__(self,texts,k1=1.5,b=.75,enhanced=True):
        self.k1,self.b,self.enhanced=k1,b,enhanced
        self.lengths=[];self.postings=defaultdict(dict)
        # postings[词项][视图索引]=该词在此视图的出现次数tf。
        # Counter将重复合并；len(postings[词项])就是df，不是全语料出现次数之和。
        for i,text in enumerate(texts):
            ts=tokenize(text,enhanced);self.lengths.append(len(ts))
            for token,tf in Counter(ts).items():self.postings[token][i]=tf
        self.lengths=np.array(self.lengths,dtype=np.float64)
        self.n=len(texts);self.average=float(self.lengths.mean()) if self.n else 1
        # 采用正值IDF：ln(1+(N-df+0.5)/(df+0.5))。
        # 越常见的词权重越小；log1p(x)对应ln(1+x)，与可能为负的其他IDF版本不同。
        self.idf={token:math.log1p((self.n-len(post)+.5)/(len(post)+.5)) for token,post in self.postings.items()}

    # 输入query，返回长度N的分数数组；第i项始终对应第i个视图，尚未排序。
    # 分数是匹配线索的合成量，不是答案正确概率；没有词命中的候选保留0。
    def score(self,query):
        scores=np.zeros(self.n)
        # 这是每个候选的长度项L=k1*(1-b+b*dl/avgdl)。
        # b=0取消长度修正；max(avgdl,1)保护极短/空语料，此时严格公式有所调整。
        norm=self.k1*(1-self.b+self.b*self.lengths/max(self.average,1))
        for token,qtf in Counter(tokenize(query,self.enhanced)).items():
            for i,tf in self.postings.get(token,{}).items():
                # 每项贡献=qtf*IDF*tf*(k1+1)/(tf+L)，最后相加。
                # tf的收益逐渐饱和；k1不是统一放大按钮，qtf则在此实现中线性相乘。
                # 未知查询词的倒排为空，不贡献分。
                scores[i]+=qtf*self.idf[token]*tf*(self.k1+1)/(tf+norm[i])
        return scores


# 预训练文字双编码器，本项目只做推理，没有训练或微调它。
# 文档离线编码，查询在线编码；两侧使用同一权重，但查询额外带检索任务指令。
class Qwen:
    def __init__(self,device=None,batch_size=8):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size=batch_size
        # 仅加载项目内权重，禁止这里临时联网。GPU半精度节省显存，CPU用float32；
        # 模型输入上限2048是模型token数，不是2048字符，也不是输出向量维度。
        self.model=SentenceTransformer(str(ROOT/"models/qwen3-embedding-0.6b"),device=self.device,
            model_kwargs={"torch_dtype":torch.float16 if self.device=="cuda" else torch.float32,"attn_implementation":"eager"},local_files_only=True)
        self.model.max_seq_length=2048

    # 输入一批文字，输出“文本条数×1024”的float32数组；每条非零向量归一化。
    # batch_size控制并行条数，不把多条文本合成一条表示。
    def encode(self,texts,query=False):
        import torch
        # 这段指令只加在查询前，文档不加。它表达“找能回答财务问题的材料”的任务。
        # 接口约定变化时，查询和文档向量空间都要核对，不能仅检查维度相同。
        prompt="Instruct: Given a financial question, retrieve relevant passages that answer the query\nQuery:" if query else ""
        batch=self.batch_size
        while True:
            try:
                output=self.model.encode(texts,prompt=prompt,batch_size=batch,normalize_embeddings=True,
                    convert_to_numpy=True,show_progress_bar=len(texts)>20)
                if output.shape[-1]!=1024:raise ValueError("Qwen维度不是1024")
                self.batch_size=batch
                # 转float32后再归一化，减少保存精度/舍入带来的范数偏差。
                # 两侧单位长度时，矩阵点积就是余弦；1e-12防除零，但零向量仍没有方向。
                output=output.astype(np.float32)
                return output / np.maximum(np.linalg.norm(output,axis=1,keepdims=True),1e-12)
            # 显存降级先减半批量，直到1；若仍显存不足，再迁CPU。
            # 这是明确的资源回退路径，可能改变速度和浮点末位，不是改用假模型。
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch>1:batch=max(1,batch//2);continue
                self.model.to("cpu");self.device="cpu";self.model.float();batch=1


# 离线写索引入口，会生成视图与两套文档矩阵；只读查看时不需重建。
# 矩阵第i行必须与views.json第i个对象保持对应，shape相等并不能保护顺序。
def build_index(variant="mixed"):
    from .views import build_views
    views=build_views(variant)
    qwen=Qwen()
    out=ROOT/f"data/indexes/{variant}"
    for style in ("natural","structural"):
        texts=[v.structural_text if style=="structural" and v.structural_text else v.text for v in views]
        # 超長行保留稀疏检索；不静默截断财务单元格。
        # 超长行用占位输入编码后写零，保留原全文给BM25。
        # 实际search未严格排除这些零向量：若其他分为负，零分仍可能排前。
        texts=[t if not v.sparse_only else "超长行仅稀疏检索" for t,v in zip(texts,views)]
        embeddings=qwen.encode(texts)
        for i,v in enumerate(views):
            if v.sparse_only:embeddings[i]=0
        np.save(out/f"embeddings_{style}.npy",embeddings)
    # 记录真实编码模式、权重revision和视图文件指纹，便于追溯哪份矩阵对应哪份文本。
    # 这不是运行时自动检查所有配置一致性的保证。
    write_json(out/"manifest.json",{"mode":"actual_qwen_encoding","model":"Qwen/Qwen3-Embedding-0.6B",
        "revision":read_json(ROOT/"models/manifest.json")["qwen_revision"],"dimension":1024,"normalized":True,
        "device":qwen.device,"batch_size":qwen.batch_size,"max_length":2048,"variant":variant,
        "view_sha256":sha256(out/"views.json"),"views":len(views),"row_styles":["natural","structural"]})
    del qwen;gc.collect()
    import torch
    if torch.cuda.is_available():torch.cuda.empty_cache()


# 统一排序入口：只在允许的ids中排序，先分降序，再视图数组索引升序。
# 索引是确定性平局规则，不是语义判断，也不是按view_id字符串排。
# 全0仍可返回列表；融合后的名次也受这一平局规则影响。
def ranked(scores,ids,k):
    return sorted(ids,key=lambda i:(-float(scores[i]),i))[:k]


# 两数组的同一i必须代表同一视图；ids是两路前20的并集。
# 在同一个并集内分别取极值，使用每项真实原分，即使它缺席另一路前20。
# weight是dense权重，BM25用1-weight；输出仍是全视图长度数组，排序另做。
def minmax(sparse,dense,ids,weight=.5):
    ids=list(ids);result=np.zeros(len(sparse))
    for scores,w in ((sparse,1-weight),(dense,weight)):
        lo,hi=min(scores[ids]),max(scores[ids])
        # 某路全同分就没有区分信息，贡献保持0；不会把其权重自动转给另一路。
        # 归一化的1只意味着本次范围最高，不是事实正确或跨题可比置信度。
        if hi>lo:
            result[ids]+=w*(scores[ids]-lo)/(hi-lo)
    return result


# 倒数排名融合只接收截断名次列表，不接收原分，因此放弃分差幅度。
# 名次从1开始，每次出现加1/(K+rank)；某路缺席就没有那项贡献。
# k是平滑常数，不是返回数量top_k；对称1/2与2/1贡献相等。
def rrf(sparse_ranks,dense_ranks,size,k=30):
    result=np.zeros(size)
    for ranks in (sparse_ranks,dense_ranks):
        for rank,i in enumerate(ranks,1):result[i]+=1/(k+rank)
    return result


# 把磁盘视图、证据与两套矩阵装成查询入口，BM25统计在本进程建立。
# 只提供查找与证据关联，不自动展开命中行的完整父表，也不读取gold来帮助搜索。
class Retriever:
    def __init__(self,variant="mixed",config=None,load_dense=True):
        self.config=config or read_json(ROOT/"configs/runtime.json")
        self.variant=variant;self.path=ROOT/f"data/indexes/{variant}"
        self.views=read_json(self.path/"views.json")
        self.evidence=read_json(self.path/"evidence.json")
        self.figures=read_json(self.path/"figures.json")
        self.bm25={};self.embeddings={}
        for style in ("natural","structural"):
            texts=[v["structural_text"] if style=="structural" and v["structural_text"] else v["text"] for v in self.views]
            for enhanced in (False,True):self.bm25[style,enhanced]=BM25(texts,enhanced=enhanced)
            # 只读内存映射直接访问历史矩阵，不重新编码全部文档。
            # load_dense=False只延迟Qwen模型加载，这两份矩阵仍会打开。
            self.embeddings[style]=np.load(self.path/f"embeddings_{style}.npy",mmap_mode="r")
        self.qwen=Qwen() if load_dense else None
        self.query_cache={}

    # 缓存仅属于当前Retriever实例，key是完全相同的查询字符串。
    # 改标点或重启进程可能重新编码；它不缓存DeepSeek答案，也不做语义等价识别。
    def encode_query(self,query):
        if query not in self.query_cache:
            if self.qwen is None:self.qwen=Qwen()
            self.query_cache[query]=self.qwen.encode([query],query=True)[0]
        return self.query_cache[query]

    # 在线检索入口：查询/过滤/方法→候选视图、分数、关联Evidence。
    # kind/doc_id控制可返回范围；只有用户题面限定文档时才应过滤，不能拿gold文档作弊。
    def search(self,query,kind=None,doc_id=None,top_k=5,method=None,style=None,enhanced=True,weight=None,rrf_k=None):
        start=time.monotonic()
        if doc_id and doc_id not in {v["doc_id"] for v in self.views}:raise ValueError("文档ID未登记")
        method=method or self.config["retrieval_method"];style=style or self.config["row_style"]
        ids=[i for i,v in enumerate(self.views) if (not kind or v["kind"] in kind) and (not doc_id or v["doc_id"]==doc_id)]
        if not ids:return []
        # 先算全体BM25；dense也做全矩阵Dq，随后才在允许ids内排。
        # 所以过滤限制候选，却没有将矩阵乘缩为只计算某几行；当前是精确全量检索。
        sparse=self.bm25[style,enhanced].score(query)
        dense=self.embeddings[style]@self.encode_query(query) if method!="bm25" else np.zeros(len(self.views))
        # 每路前20用来建融合池，最后top_k控制返回数量，两个数量不能混淆。
        # 这里的20直接写在执行语句里，不由某个同名配置字段自动控制。
        sparse_ranks=ranked(sparse,ids,20);dense_ranks=ranked(dense,ids,20)
        if method=="bm25":scores=sparse; candidates=ids
        elif method=="qwen":scores=dense;candidates=ids
        else:
            # 先去重并集并固定索引顺序，再依融合分排序。
            # 正确行若两路前20都未进入，融合无法救回；重排池内候选也同样无法救回池外材料。
            candidates=sorted(set(sparse_ranks)|set(dense_ranks))
            if method=="minmax":scores=minmax(sparse,dense,candidates,weight if weight is not None else self.config["dense_weight"])
            elif method=="rrf":scores=rrf(sparse_ranks,dense_ranks,len(self.views),rrf_k or self.config["rrf_k"])
            else:raise ValueError("未知检索方法")
        result=[]
        # 命中一个视图后，只取它自己列出的evidence_ids。
        # 例如微软收入行关联行证据、行名格及三年金额格；表摘要只关联摘要，不自动拿全表。
        for i in ranked(scores,candidates,top_k):
            v=self.views[i]
            result.append({"view":v,"score":float(scores[i]),"bm25_score":float(sparse[i]),"dense_score":float(dense[i]),
                "evidence":[self.evidence[e] for e in v["evidence_ids"]],"elapsed":time.monotonic()-start})
        return result
