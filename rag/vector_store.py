import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if VectorStoreService._initialized:
            return
        
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        
        VectorStoreService._initialized = True

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _md5_store_path(self) -> str:
        return get_abs_path(chroma_conf["md5_hex_store"])

    def _check_md5_hex(self, md5_for_check: str) -> bool:
        md5_store_path = self._md5_store_path()
        if not os.path.exists(md5_store_path):
            open(md5_store_path, "w", encoding="utf-8").close()
            return False

        with open(md5_store_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line.strip() == md5_for_check:
                    return True

        return False

    def _save_md5_hex(self, md5_for_check: str) -> None:
        with open(self._md5_store_path(), "a", encoding="utf-8") as f:
            f.write(md5_for_check + "\n")

    def _get_file_documents(self, read_path: str) -> list[Document]:
        lower_path = read_path.lower()
        if lower_path.endswith(".txt"):
            return txt_loader(read_path)
        if lower_path.endswith(".pdf"):
            return pdf_loader(read_path)
        return []

    def ingest_file(self, path: str, force_reload: bool = False) -> dict[str, object]:
        md5_hex = get_file_md5_hex(path)
        if not md5_hex:
            raise ValueError(f"无法计算文件 MD5: {path}")

        if not force_reload and self._check_md5_hex(md5_hex):
            logger.info(f"[加载知识库]{path} 内容已存在知识库中，跳过")
            return {"ingested": False, "reason": "duplicate_md5"}

        documents = self._get_file_documents(path)
        if not documents:
            logger.warning(f"[加载知识库]{path} 中没有有效文本内容，跳过")
            return {"ingested": False, "reason": "empty_document"}

        split_document = self.spliter.split_documents(documents)
        if not split_document:
            logger.warning(f"[加载知识库]{path} 切片后没有有效文本内容，跳过")
            return {"ingested": False, "reason": "empty_chunk"}

        self.vector_store.add_documents(split_document)
        self._save_md5_hex(md5_hex)
        logger.info(f"[加载知识库]{path} 内容加载成功")
        return {"ingested": True, "reason": "success", "chunks": len(split_document)}

    def load_document(self):
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            try:
                self.ingest_file(path)
            except Exception as e:
                logger.error(f"[加载知识库]{path} 失败: {str(e)}", exc_info=True)
                continue


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()

    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
