import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Not: GOOGLE_API_KEY'i Vercel'in Environment Variables kısmına ekleyeceğiz.
# Bu kod, Vercel'de çalışırken anahtarı oradan otomatik olarak alacaktır.

# --- RAG Pipeline Kurulumu (Uygulama Başladığında Bir Kez Çalışır) ---
try:
    # 1. Bilgi Kaynağını Yükle ve Parçalara Ayır
    with open("bilgi_kaynagi.txt", "r", encoding="utf-8") as f:
        knowledge_base_text = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(knowledge_base_text)

    # 2. Metinleri Google Embeddings ile Vektörlere Dönüştür ve Veritabanı Oluştur
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts, embeddings)

    # 3. LLM olarak Gemini Modelini ve Prompt Şablonunu Tanımla
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, convert_system_message_to_human=True)

    # Prompt Mühendisliği: Modelin nasıl davranacağını burada belirliyoruz.
    prompt_template = """Verilen metinleri kullanarak kullanıcı sorusuna cevap ver. Cevapların kısa, net ve samimi olsun. Eğer bilgi metinlerde yoksa, 'Bu konuda bilgim bulunmuyor, size yardımcı olması için destek ekibimize ulaşabilirsiniz.' de. Kesinlikle bilgi uydurma.

    Metinler:
    {context}

    Soru: {question}
    Cevap:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 4. RAG Zincirini (QA Chain) Oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    RAG_SETUP_SUCCESS = True
except Exception as e:
    RAG_SETUP_SUCCESS = False
    RAG_ERROR_MESSAGE = str(e)


# --- FastAPI Uygulaması ---
app = FastAPI()

class Query(BaseModel):
    question: str

# Ana sayfa (index.html)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Chatbot API endpoint'i
@app.post("/ask")
async def ask_question(query: Query):
    if not RAG_SETUP_SUCCESS:
        return {"answer": f"Uygulama başlatılırken bir hata oluştu: {RAG_ERROR_MESSAGE}. Lütfen Vercel loglarını kontrol edin. API anahtarı doğru ayarlanmış mı?"}
    
    try:
        result = qa_chain.invoke({"query": query.question})
        return {"answer": result['result']}
    except Exception as e:
        if "response was blocked" in str(e).lower():
            return {"answer": "Üzgünüm, bu soruya yanıt veremiyorum. Lütfen farklı bir şekilde sormayı deneyin."}
        return {"answer": f"Cevap üretilirken bir hata oluştu: {str(e)}"}