# 下载已知URL与网络搜索是不同动作；下载脚本不能替代来源可信度核验。
"""下载已知来源，不执行搜索；保存原文件和来源哈希。"""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "microsoft2024.pdf": "https://www.annualreports.com/HostedData/AnnualReportArchive/m/NASDAQ_MSFT_2024.pdf",
    "apple2024.pdf": "https://s2.q4cdn.com/470004039/files/doc_earnings/2024/q4/filing/10-Q4-2024-As-Filed.pdf",
    "catl2024.pdf": "https://www.catl.com/uploads/1/file/public/202503/20250317094543_6ig9e0mwng.pdf",
    "finqa_train.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "finqa_test.json": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}


def main():
    # 新克隆项目不带原始资料目录；只在项目内建立下载目标。
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, url in SOURCES.items():
        path = ROOT / "data/raw" / name
        # 已有文件复用并记录哈希；新响应先查PDF头或JSON可解析，不把错误网页保存成PDF。
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    content = r.read()
                if name.endswith(".pdf") and not content.startswith(b"%PDF"):
                    raise ValueError("响应并非PDF")
                if name.endswith(".json"):
                    json.loads(content)
                path.write_bytes(content)
            except Exception as exc:
                manifest.append({"file": name, "url": url, "status": "failed", "error": type(exc).__name__})
                print(name + "：下载失败 " + type(exc).__name__)
                continue
        content = path.read_bytes()
        item = {"file": name, "url": url, "status": "downloaded", "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()}
        manifest.append(item)
        print(name + "：已下载，" + str(len(content)) + " 字节")
    (ROOT / "data/source_downloads.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any(x["status"] == "failed" for x in manifest))


if __name__ == "__main__":
    raise SystemExit(main())
