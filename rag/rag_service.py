
"""
优化后的总结服务类：实现多路召回（向量+BM25关键字）+ BGE重排机制
支持异步流式返回（astream），完美适配 Agent 架构
"""
import asyncio
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from utils.logger_handler import logger

# 尝试加载重排模型，如果失败则降级
try:
    from sentence_transformers import CrossEncoder
    RERANK_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers 未安装，将不使用重排功能")
    RERANK_AVAILABLE = False


class RagSummarizeService(object):
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RagSummarizeService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if RagSummarizeService._initialized:
            return
        
        # 1. 初始化底层的向量数据库检索器（第一路：语义召回）
        self.vector_store = VectorStoreService()
        self.dense_retriever = self.vector_store.get_retriever()
        
        # 2. 初始化 BM25 检索器（第二路：关键字召回）
        self.sparse_retriever = None 
        self._init_sparse_retriever()

        # 3. 初始化本地重排模型（Reranker）
        self.rerank_model = None
        if RERANK_AVAILABLE:
            try:
                self.rerank_model = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
                logger.info("成功加载 BGE 重排模型")
            except Exception as e:
                logger.error(f"加载重排模型失败: {e}，将降级为无重排模式。")
                self.rerank_model = None

        # 4. 初始化大模型与 Prompt 链
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        
        RagSummarizeService._initialized = True

    def _init_sparse_retriever(self):
        """从知识库文件直接加载文档初始化 BM25 检索器"""
        from utils.config_handler import chroma_conf
        from utils.path_tool import get_abs_path
        from utils.file_handler import listdir_with_allowed_type, txt_loader, pdf_loader
        
        docs = []
        try:
            # 直接从 data/ 目录读取原始文件来初始化 BM25（最可靠的方式）
            data_path = get_abs_path(chroma_conf["data_path"])
            allowed_files = listdir_with_allowed_type(
                data_path, tuple(chroma_conf["allow_knowledge_file_type"])
            )
            
            for file_path in allowed_files:
                try:
                    file_docs = []
                    if file_path.lower().endswith('.txt'):
                        file_docs = txt_loader(file_path)
                    elif file_path.lower().endswith('.pdf'):
                        file_docs = pdf_loader(file_path)
                    
                    if file_docs:
                        docs.extend(file_docs)
                        logger.debug(f"已加载 {file_path} 到 BM25 检索器")
                except Exception as e:
                    logger.warning(f"加载文件 {file_path} 到 BM25 失败: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"从目录加载文档到 BM25 失败: {e}")
        
        if docs:
            self.sparse_retriever = BM25Retriever.from_documents(docs)
            self.sparse_retriever.k = 3
            logger.info(f"BM25 检索器初始化成功，共加载 {len(docs)} 个文档块")
        else:
            logger.warning("未能加载任何文档，BM25 检索器未初始化")

    def _init_chain(self):
        return self.prompt_template | self.model | StrOutputParser()

    async def _hybrid_retrieve(self, query: str, top_k_recall: int = 4) -> list[Document]:
        """
        核心步骤 1：多路并行召回
        使用 asyncio.gather 同时并发调用向量检索和文本检索，最大化压榨 I/O 效率
        """
        try:
            # 动态调整两路的召回数量
            self.sparse_retriever.k = top_k_recall
            
            # 并发执行两路检索任务
            dense_task = asyncio.create_task(self.dense_retriever.ainvoke(query))
            sparse_task = asyncio.create_task(self.sparse_retriever.ainvoke(query))
            
            dense_docs, sparse_docs = await asyncio.gather(dense_task, sparse_task)
            logger.info(f"向量检索返回 {len(dense_docs)} 条，BM25 检索返回 {len(sparse_docs)} 条")
        except Exception as e:
            logger.error(f"多路召回发生异常: {e}，降级为单路向量检索")
            dense_docs = await self.dense_retriever.ainvoke(query)
            sparse_docs = []

        # 合并两路结果并根据 page_content 进行去重
        seen_contents = set()
        combined_docs = []
        
        for doc in dense_docs + sparse_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                combined_docs.append(doc)
        
        logger.info(f"合并去重后共有 {len(combined_docs)} 条候选文档")
        return combined_docs

    def _rerank_docs(self, query: str, docs: list[Document], top_k_final: int = 2) -> list[Document]:
        """
        核心步骤 2：使用 Cross-Encoder 进行精准重排
        """
        if not docs or not self.rerank_model:
            return docs[:top_k_final]

        # 1. 组装重排模型的输入对: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in docs]
        
        # 2. 预测打分
        scores = self.rerank_model.predict(pairs)
        
        # 3. 将分数绑定回 Document 的 metadata 中
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)
            
        # 4. 按照重排分数从大到小降序排列
        reranked_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
        
        logger.info(f"重排完成，分数前 {top_k_final} 名: {[d.metadata.get('rerank_score', 0) for d in reranked_docs[:top_k_final]]}")
        
        # 5. 截取最终真正精准的前 N 条喂给大模型
        return reranked_docs[:top_k_final]

    async def rag_summarize_stream(self, query: str):
        """
        供上层 Agent 调用的异步流式生成接口
        """
        # 1. 多路并发召回候选文档（向量 + 关键字）
        candidate_docs = await self._hybrid_retrieve(query, top_k_recall=4)
        
        # 2. 本地 Reranker 模型重排打分，筛选出最终最优的 2 条
        final_docs = self._rerank_docs(query, candidate_docs, top_k_final=2)

        # 3. 规范化拼装上下文格式
        context = ""
        for counter, doc in enumerate(final_docs, 1):
            score_info = f" | 重排得分: {doc.metadata.get('rerank_score', 'N/A')}"
            context += f"【参考资料{counter}】: {doc.page_content} (元数据: {doc.metadata}{score_info})\n"

        logger.info(f"构建完成上下文，准备调用 LLM")

        # 4. 使用 astream 异步流式向外吐出 Token
        async for chunk in self.chain.astream({"input": query, "context": context}):
            yield chunk

    async def rag_summarize(self, query: str) -> str:
        """
        保持同步接口兼容性，供旧代码调用
        """
        full_response = ""
        async for chunk in self.rag_summarize_stream(query):
            full_response += chunk
        return full_response


if __name__ == '__main__':
    # 异步运行测试小样
    async def test_main():
        rag = RagSummarizeService()
        print("开始多路召回与重排测试...")
        async for token in rag.rag_summarize_stream("小户型适合哪些扫地机器人"):
            print(token, end="", flush=True)
            
    asyncio.run(test_main())
