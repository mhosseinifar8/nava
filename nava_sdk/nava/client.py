from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from websockets.asyncio.client import connect

class NavaServiceError(RuntimeError): pass
@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: list[dict]
    language_prob: float
@dataclass(frozen=True)
class SpeechResult:
    pcm: bytes
    sample_rate: int
    engine: str
def check(response: httpx.Response) -> None:
    if response.is_success: return
    try: detail=response.json().get("detail",response.text)
    except ValueError: detail=response.text
    raise NavaServiceError(f"Nava returned HTTP {response.status_code}: {detail}")
def ws_url(url: str, path: str) -> str:
    return ("wss://"+url[8:] if url.startswith("https://") else "ws://"+url[7:] if url.startswith("http://") else url)+path
class NavaClient:
    def __init__(self,stt_url="http://127.0.0.1:8001",tts_url="http://127.0.0.1:8002",*,timeout=120.0): self.stt_url=stt_url.rstrip("/"); self.tts_url=tts_url.rstrip("/"); self.http=httpx.Client(timeout=timeout)
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
    def close(self): self.http.close()
    def transcribe(self,pcm:bytes,*,hotwords:str|None=None,initial_prompt:str|None=None)->TranscriptionResult:
        r=self.http.post(self.stt_url+"/api/v1/transcriptions",content=pcm,headers={"Content-Type":"application/octet-stream"},params={k:v for k,v in {"hotwords":hotwords,"initial_prompt":initial_prompt}.items() if v}); check(r); b=r.json(); return TranscriptionResult(b["text"],b["segments"],b["language_prob"])
    def synthesize(self,text:str,*,engine="pocket",voice_prompt="default",speed=1.0)->SpeechResult:
        r=self.http.post(self.tts_url+"/api/v1/speech",json={"text":text,"engine":engine,"voice_prompt":voice_prompt,"speed":speed}); check(r); return SpeechResult(r.content,int(r.headers["X-Nava-Sample-Rate"]),r.headers["X-Nava-Engine"])
class AsyncNavaClient:
    def __init__(self,stt_url="http://127.0.0.1:8001",tts_url="http://127.0.0.1:8002",*,timeout=120.0): self.stt_url=stt_url.rstrip("/"); self.tts_url=tts_url.rstrip("/"); self.http=httpx.AsyncClient(timeout=timeout)
    async def __aenter__(self): return self
    async def __aexit__(self,*_): await self.aclose()
    async def aclose(self): await self.http.aclose()
    async def transcribe(self,pcm:bytes,*,hotwords:str|None=None,initial_prompt:str|None=None)->TranscriptionResult:
        r=await self.http.post(self.stt_url+"/api/v1/transcriptions",content=pcm,headers={"Content-Type":"application/octet-stream"},params={k:v for k,v in {"hotwords":hotwords,"initial_prompt":initial_prompt}.items() if v}); check(r); b=r.json(); return TranscriptionResult(b["text"],b["segments"],b["language_prob"])
    async def synthesize(self,text:str,*,engine="pocket",voice_prompt="default",speed=1.0)->SpeechResult:
        r=await self.http.post(self.tts_url+"/api/v1/speech",json={"text":text,"engine":engine,"voice_prompt":voice_prompt,"speed":speed}); check(r); return SpeechResult(r.content,int(r.headers["X-Nava-Sample-Rate"]),r.headers["X-Nava-Engine"])
    async def transcribe_ws(self,pcm:bytes,*,hotwords:str|None=None,initial_prompt:str|None=None)->TranscriptionResult:
        async with connect(ws_url(self.stt_url,"/ws/stt"),max_size=None) as socket:
            config={k:v for k,v in {"hotwords":hotwords,"initial_prompt":initial_prompt}.items() if v}
            if config: await socket.send(json.dumps(config,ensure_ascii=False))
            await socket.send(pcm); await socket.send('{"action":"eos"}')
            while True:
                frame=await socket.recv()
                if isinstance(frame,bytes): continue
                body=json.loads(frame)
                if body.get("status")=="error": raise NavaServiceError(body["error"])
                if "text" in body: return TranscriptionResult(body["text"],body["segments"],body["language_prob"])
    async def stream_speech_ws(self,text:str,*,engine="pocket",voice_prompt="default",speed=1.0)->AsyncIterator[SpeechResult]:
        async with connect(ws_url(self.tts_url,"/ws/tts"),max_size=None) as socket:
            await socket.send(json.dumps({"text":text,"engine":engine,"voice_prompt":voice_prompt,"speed":speed},ensure_ascii=False)); rate=None
            while True:
                frame=await socket.recv()
                if isinstance(frame,bytes):
                    if rate is None: raise NavaServiceError("TTS sent audio before metadata")
                    yield SpeechResult(frame,rate,engine); continue
                body=json.loads(frame)
                if body.get("status")=="error": raise NavaServiceError(body["error"])
                if body.get("status")=="streaming": rate=int(body["sample_rate"]); engine=body["engine"]
                if body.get("status")=="completed": return
