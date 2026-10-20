# 已知模型源下载入口：权重/缓存放项目，记录revision与文件哈希，不执行网页搜索。
from finrag.paths import ROOT
from finrag.store import write_json, sha256
from docling.utils.model_downloader import download_models
from docling.models.easyocr_model import EasyOcrModel
from huggingface_hub import snapshot_download, HfApi


def main():
    download_models(ROOT / "models/docling", with_code_formula=False,
                    with_picture_classifier=False, with_rapidocr=False, with_easyocr=False)
    print("Docling版面与TableFormer权重已保存到项目")
    EasyOcrModel.download_models(recognition_models=["english_g2", "zh_sim_g2"],
                                 local_dir=ROOT / "models/easyocr")
    print("EasyOCR英文/简中权重已保存到项目")
    # 先取确定revision再下载该提交，减少下载过程中模型更新导致文件版本混杂。
    info = HfApi().model_info("Qwen/Qwen3-Embedding-0.6B")
    snapshot_download("Qwen/Qwen3-Embedding-0.6B", revision=info.sha,
        local_dir=ROOT / "models/qwen3-embedding-0.6b",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.md"])
    write_json(ROOT / "models/manifest.json", {"qwen_revision": info.sha,
        "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
                  for p in (ROOT / "models").rglob("*") if p.is_file() and ".cache" not in p.parts]})
    print("Qwen权重与revision已保存到项目")


if __name__ == "__main__":
    main()
