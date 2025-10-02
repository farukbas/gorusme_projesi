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
# YENİ HALİ:
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, convert_system_message_to_human=True)
# 3. Prompt Mühendisliği: Prompt'u bilgiyi ve soruyu alacak şekilde güncelle
prompt_template = """Sen BulutSantral A.Ş. için çalışan, nazik ve yardımsever bir yapay zeka asistanısın. Görevin, kullanıcının yazdığı metnin türüne göre iki farklı şekilde cevap vermektir:

1.  **EĞER KULLANICI BİR SORU SORARSA:** Cevabı SADECE ve SADECE aşağıda verilen 'Bilgi Kaynağı' metnini kullanarak bul ve cevapla. Eğer soruya cevap metinde yoksa, 'Bu konuda güncel bir bilgim bulunmuyor, dilerseniz sizi satış veya destek ekibimize yönlendirebilirim.' de. Asla bilgi uydurma.

2.  **EĞER KULLANICI BİR SORU SORMAZSA:** Kullanıcının yazdığı metin bir soru değil de, 'tamamdır', 'teşekkür ederim', 'çok iyi', 'harika', 'anladım' gibi bir onay, teşekkür veya olumlu bir geri bildirim ise, Bilgi Bankası'nı KESİNLİKLE KULLANMA. Bu durumda, kullanıcıya sıcak ve doğal bir tepki ver. Örneğin: 'Yardımcı olabildiğime sevindim!', 'Rica ederim, başka bir sorunuz var mıydı?', 'Harika! Size başka nasıl yardımcı olabilirim?' veya 'Ne demek, memnuniyetle!' gibi kısa ve nazik bir cevap ver.

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