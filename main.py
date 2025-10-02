import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --- Yeni ve Daha Basit Mantık ---

# 1. Bilgi Kaynağını Uygulama Başlarken Sadece Bir Kez Oku
try:
    with open("bilgi_kaynagi.txt", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = f.read()
    SETUP_SUCCESS = True
except Exception as e:
    SETUP_SUCCESS = False
    ERROR_MESSAGE = str(e)

# 2. LLM olarak Sadece Gemini Modelini Tanımla
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, convert_system_message_to_human=True)

# 3. Prompt Mühendisliği: Prompt'u bilgiyi ve soruyu alacak şekilde güncelle
prompt_template = """Sen BulutSantral A.Ş. için çalışan bir müşteri temsilcisisin. Sadece ve sadece aşağıda verilen 'Bilgi Kaynağı' metnini kullanarak kullanıcının sorusunu cevapla. Cevapların kısa, net ve samimi olsun. Eğer cevap metinde yoksa, 'Bu konuda bilgim bulunmuyor, size yardımcı olması için destek ekibimize ulaşabilirsiniz.' de. Kesinlikle bilgi uydurma.

---
Bilgi Kaynağı:
{knowledge_base}
---

Kullanıcının Sorusu: {question}

Cevap:
"""

# Prompt şablonunu oluştur
prompt = PromptTemplate(
    template=prompt_template, input_variables=["knowledge_base", "question"]
)

# --- FastAPI Uygulaması ---
app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/ask")
async def ask_question(query: Query):
    if not SETUP_SUCCESS:
        return {"answer": f"Uygulama başlatılırken bir hata oluştu: {ERROR_MESSAGE}."}
    
    try:
        # 4. Final Prompt'u oluştur ve LLM'i doğrudan çağır
        final_prompt = prompt.format(knowledge_base=KNOWLEDGE_BASE, question=query.question)
        
        # Karmaşık chain'ler yerine doğrudan invoke kullanıyoruz
        result = llm.invoke(final_prompt)
        
        return {"answer": result.content}
    except Exception as e:
        return {"answer": f"Cevap üretilirken bir hata oluştu: {str(e)}"}