import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --- YENİ: Gelen Konuşma Geçmişi için Modeller ---
class Turn(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    # Artık tek bir 'question' yerine 'history' listesi alıyoruz
    history: List[Turn]

# --- Bilgi Kaynağını Oku ---
try:
    with open("bilgi_kaynagi.txt", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = f.read()
    SETUP_SUCCESS = True
except Exception as e:
    SETUP_SUCCESS = False
    ERROR_MESSAGE = str(e)

# --- LLM'i ve Prompt Şablonunu Ayarla ---
# Not: Model adını "gemini-1.5-flash-latest" olarak değiştirdim, çünkü bu en güncel ve stabil versiyondur.
# "gemini-2.5-flash" bazen hata verebiliyor.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, convert_system_message_to_human=True)

# YENİ: Prompt şablonu artık 'chat_history' değişkenini de içeriyor
prompt_template = """Sen BulutSantral A.Ş. için çalışan, nazik ve yardımsever bir yapay zeka asistanısın. Görevin, sana verilen Bilgi Kaynağı'nı ve önceki Konuşma Geçmişi'ni dikkate alarak kullanıcının SON mesajına cevap vermektir.

-   EĞER KULLANICI BİR SORU SORARSA: Cevabı SADECE Bilgi Kaynağı'ndan bul. Cevap orada yoksa "Bu konuda bilgim bulunmuyor" de.
-   EĞER KULLANICI TEŞEKKÜR EDER VEYA ONAYLARSA: ("tamam", "güzel", "teşekkürler" gibi), ona göre nazik bir cevap ver.
-   Cevaplarında önceki konuşmayı dikkate al. Örneğin, kullanıcı bir paket sorduktan sonra "fiyatı ne kadar?" derse, o paketin fiyatını söylemelisin.

---
BİLGİ KAYNAĞI:
{knowledge_base}
---

---
KONUŞMA GEÇMİŞİ:
{chat_history}
---

Kullanıcının SON MESAJI: {question}

Asistanın Cevabı:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["knowledge_base", "chat_history", "question"]
)

# --- FastAPI Uygulaması ---
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/ask")
async def ask_question(query: Query):
    if not SETUP_SUCCESS:
        return {"answer": f"Uygulama başlatılırken bir hata oluştu: {ERROR_MESSAGE}."}
    
    try:
        # Konuşma geçmişini okunabilir bir metne çeviriyoruz
        formatted_history = ""
        for turn in query.history[:-1]: # Son kullanıcı mesajı hariç
            role = "Asistan" if turn.role == 'model' else "Kullanıcı"
            formatted_history += f"{role}: {turn.content}\n"

        last_user_message = query.history[-1].content

        # Final Prompt'u oluştur ve LLM'i çağır
        final_prompt = prompt.format(
            knowledge_base=KNOWLEDGE_BASE, 
            chat_history=formatted_history, 
            question=last_user_message
        )
        
        result = llm.invoke(final_prompt)
        
        return {"answer": result.content}
    except Exception as e:
        return {"answer": f"Cevap üretilirken bir hata oluştu: {str(e)}"}