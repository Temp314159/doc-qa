# CLI只负责参数分发；search是检索，ask才进入模型工具循环。
# evaluate会写报告并可能调用API，不是只读查看报告的入口。
import argparse
import json
from .paths import ROOT
from .store import read_json


def main():
    p=argparse.ArgumentParser(description='报告与文档多模态RAG')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare-generated',help='生成合成和FinQA重建PDF')
    a=sub.add_parser('parse',help='实际解析或使用真实解析缓存');a.add_argument('--doc-id');a.add_argument('--variant',choices=['mixed','native'],default='mixed');a.add_argument('--force',action='store_true')
    a=sub.add_parser('index',help='Qwen实际建索引');a.add_argument('--variant',default='mixed')
    a=sub.add_parser('search');a.add_argument('query');a.add_argument('--method',choices=['bm25','qwen','minmax','rrf']);a.add_argument('--top-k',type=int,default=5)
    a=sub.add_parser('ask');a.add_argument('question');a.add_argument('--run-id')
    a=sub.add_parser('evaluate-retrieval');a.add_argument('--split',choices=['dev','test'],default='dev')
    a=sub.add_parser('evaluate-qa');a.add_argument('--suite',choices=['demo','online','remaining','remaining-test','refusal'],default='demo');a.add_argument('--attempt',help='可选运行标签；省略时生成独立编号')
    sub.add_parser('budget');sub.add_parser('validate-data')
    args=p.parse_args()
    # 按需导入重依赖；只跑validate-data不需要启动PDF解析或加载外部模型密钥。
    if args.command=='prepare-generated':
        from .preparation import prepare_generated
        prepare_generated()
    elif args.command=='parse':
        from .ingestion import parse_all
        parse_all(args.doc_id,args.variant,args.force)
    elif args.command=='index':
        from .retrieval import build_index
        build_index(args.variant)
    # 二者共用Retriever；ask额外构造Agent并发真实请求，返回JSON包含过程与来源。
    elif args.command in ('search','ask'):
        from .retrieval import Retriever
        r=Retriever()
        if args.command=='search':result=r.search(args.query,method=args.method,top_k=args.top_k)
        else:
            from .agent import Agent
            result=Agent(r).run(args.question,args.run_id)
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    elif args.command=='evaluate-retrieval':
        from .evaluation import evaluate_retrieval
        evaluate_retrieval(args.split)
    elif args.command=='evaluate-qa':
        from .evaluation import evaluate_qa
        evaluate_qa(args.suite,attempt=args.attempt)
    # 这里只读账本保守占用，不向平台查实扣账单。
    elif args.command=='budget':
        from .budget import Ledger
        print('累计保守费用（元）：'+str(Ledger().total()))
    elif args.command=='validate-data':
        from .evaluation import validate_data
        validate_data()


if __name__=='__main__':main()
