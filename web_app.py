"""Web UI for the knowledge assistant - NO LLM and NO API key needed.

Run:   py -3.14 web_app.py        then open http://127.0.0.1:8000
It shows the passages (PDF pages / sheet rows) closest in meaning to your question,
with file name, page/row and a similarity score. Chat memory is kept, so short
follow-up questions are searched together with your previous question.
"""
import json
import threading
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")

from langchain_core.messages import HumanMessage  # noqa: E402

from src.config import Settings  # noqa: E402
from src.indexer import get_embeddings, load_or_build_index  # noqa: E402
from src.memory import BufferMemory  # noqa: E402
from src.rag_chain import _location  # noqa: E402

PORT = 8000

settings = Settings.from_env()
print("Loading knowledge base ...")
store, info = load_or_build_index(settings, get_embeddings(settings))
print(f"Ready: {info['chunks']} chunks | embeddings = {settings.embedding_provider}")
memory = BufferMemory()
lock = threading.Lock()


def build_query(question: str) -> str:
    """Short follow-ups ("what are its prerequisites?") get the previous question added."""
    previous = [m.content for m in memory.get_messages() if isinstance(m, HumanMessage)]
    if previous and len(question.split()) <= 6:
        return previous[-1] + " " + question
    return question


def ask(question: str) -> dict:
    with lock:
        query = build_query(question)
        hits = store.similarity_search_with_score(query, k=settings.top_k)
        results = [
            {
                "source": d.metadata.get("source"),
                "location": _location(d.metadata),
                # FAISS returns squared L2 distance of unit vectors -> convert to cosine
                "score": round(max(0.0, 1 - float(dist) / 2), 3),
                "text": d.page_content,
            }
            for d, dist in hits
        ]
        top = results[0] if results else None
        memory.add_turn(question, f"{top['source']} ({top['location']})" if top else "-")
        return {"query": query, "results": results, "turns": memory.turn_count}


PAGE = """<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Delta Assistant</title>
<style>
:root{--bg:#f4f5f8;--card:#fff;--ink:#1b2130;--muted:#677084;--line:#e2e5ec;--accent:#2457d6;--onaccent:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#11141b;--card:#1a1f2a;--ink:#e8ebf2;--muted:#98a1b3;--line:#2a3140;--accent:#6b93ff;--onaccent:#0b1020}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,"Segoe UI",Tahoma,sans-serif;display:flex;flex-direction:column;height:100vh}
header{display:flex;align-items:center;gap:12px;padding:12px 16px;background:var(--card);border-bottom:1px solid var(--line)}
header h1{font-size:17px;margin:0;flex:1}
.pill{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:99px;padding:2px 10px}
button{font:inherit;cursor:pointer;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px}
button.primary{background:var(--accent);color:var(--onaccent);border-color:var(--accent)}
#chat{flex:1;overflow:auto;padding:16px;display:flex;flex-direction:column;gap:14px;max-width:860px;width:100%;margin:0 auto}
.me{align-self:flex-start;background:var(--accent);color:var(--onaccent);padding:8px 14px;border-radius:14px;max-width:85%;white-space:pre-wrap}
.bot{align-self:stretch;display:flex;flex-direction:column;gap:8px}
.note{font-size:13px;color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.card.best{border-color:var(--accent)}
.meta{display:flex;gap:8px;align-items:center;font-size:13px;color:var(--muted);margin-bottom:4px;direction:ltr;justify-content:flex-end}
.meta b{color:var(--ink)}
.score{margin-inline-start:auto}
.txt{white-space:pre-wrap;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer;unicode-bidi:plaintext}
.txt.open{-webkit-line-clamp:unset}
form{display:flex;gap:8px;padding:12px 16px;background:var(--card);border-top:1px solid var(--line)}
form .wrap{display:flex;gap:8px;max-width:860px;width:100%;margin:0 auto}
input{flex:1;font:inherit;padding:10px 14px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.hint{color:var(--muted);text-align:center;margin-top:30px}
</style></head><body>
<header><h1>Delta Assistant</h1><span class="pill" id="mem">الذاكرة: 0 أسئلة</span>
<button id="reset">مسح المحادثة</button></header>
<div id="chat"><div class="hint">اكتب سؤالك عن اللائحة أو الخدمات أو المقررات أو الرسوم.<br>
مثال: How many credit hours are required to graduate؟ ثم: And what CGPA do I need؟</div></div>
<form id="f"><div class="wrap"><input id="q" autocomplete="off" dir="auto" placeholder="اكتب سؤالك هنا..." autofocus>
<button class="primary" id="send">بحث</button></div></form>
<script>
const chat=document.getElementById('chat'),q=document.getElementById('q'),mem=document.getElementById('mem');
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function scroll(){chat.scrollTop=chat.scrollHeight}
document.getElementById('f').addEventListener('submit',async ev=>{
  ev.preventDefault();const question=q.value.trim();if(!question)return;
  chat.querySelector('.hint')?.remove();q.value='';
  const me=el('div','me',question);me.dir='auto';chat.append(me);
  const bot=el('div','bot');const wait=el('div','note','جاري البحث...');bot.append(wait);chat.append(bot);scroll();
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
    const d=await r.json();wait.remove();
    if(d.error){bot.append(el('div','note',d.error));return}
    mem.textContent='الذاكرة: '+d.turns+' أسئلة';
    const note=el('div','note','أقرب الفقرات من ملفاتك'+(d.query!==question?' — بحثت عن: '+d.query:''));note.dir='auto';bot.append(note);
    d.results.forEach((x,i)=>{
      const c=el('div','card'+(i===0?' best':''));const m=el('div','meta');
      m.append(el('b','',x.source),el('span','',x.location),el('span','score',Math.round(x.score*100)+'%'));
      const t=el('div','txt',x.text);t.title='اضغط للتوسيع';t.onclick=()=>t.classList.toggle('open');
      c.append(m,t);bot.append(c)});
  }catch(e){wait.textContent='حصل خطأ في الاتصال بالسيرفر'}
  scroll();
});
document.getElementById('reset').onclick=async()=>{
  await fetch('/api/reset',{method:'POST'});chat.innerHTML='';mem.textContent='الذاكرة: 0 أسئلة'};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype: str = "application/json; charset=utf-8") -> None:
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            payload = {}
        if self.path == "/api/ask":
            question = str(payload.get("question", "")).strip()
            if not question:
                self._send(400, json.dumps({"error": "اكتب سؤالًا أولًا"}, ensure_ascii=False))
                return
            self._send(200, json.dumps(ask(question), ensure_ascii=False))
        elif self.path == "/api/reset":
            with lock:
                memory.clear()
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, *args) -> None:  # keep the terminal clean
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Open {url} in your browser (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")