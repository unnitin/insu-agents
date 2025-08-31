
from __future__ import annotations
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.responses import PlainTextResponse
import uuid, os

app = FastAPI(title="Call Server", version="0.2.0")
CALLS: Dict[str, Dict[str, Any]] = {}

class StartBody(BaseModel):
    to: str
    lead_name: str
    script: Dict[str, Any]
    callbacks: Dict[str, str]

def _twilio_available() -> bool:
    try:
        from .twilio_client import TwilioClient  # noqa
        return True
    except Exception:
        return False

@app.post('/api/call/start')
async def start_call(payload: StartBody):
    sid = 'C' + uuid.uuid4().hex[:12]
    CALLS[sid] = {'status':'queued', 'payload': payload.dict(), 'events': []}
    if _twilio_available() and os.getenv('TWILIO_ACCOUNT_SID'):
        from .twilio_client import TwilioClient
        tc = TwilioClient()
        public_base = os.getenv('PUBLIC_BASE_URL', 'http://localhost:8790')
        voice_webhook = f"{public_base}/twilio/voice?cid={sid}"
        status_cb = f"{public_base}/twilio/status?cid={sid}"
        resp = tc.place_call(to=payload.to, webhook_url=voice_webhook, status_callback_url=status_cb)
        CALLS[sid]['twilio'] = resp; CALLS[sid]['status'] = 'dialing'
    return {'ok': True, 'call_id': sid, 'twilio': CALLS[sid].get('twilio')}

@app.post('/twilio/status')
async def twilio_status(request: Request):
    form = await request.form()
    cid = request.query_params.get('cid')
    if cid and cid in CALLS:
        CALLS[cid]['twilio_status'] = dict(form)
    return {'ok': True}

@app.post('/twilio/voice')
async def twilio_voice(request: Request):
    cid = request.query_params.get('cid')
    if not cid or cid not in CALLS:
        return PlainTextResponse('<Response><Say>No active script found.</Say></Response>', media_type='application/xml')
    script = CALLS[cid]['payload']['script']
    opening = script.get('opening','Hello'); disclosure = script.get('disclosure','')
    xml = f'''
<Response>
  <Say>{opening}</Say>
  <Pause length="1"/>
  <Say>{disclosure}</Say>
  <Gather input="speech dtmf" timeout="5" numDigits="1">
    <Say>Please press any key or say yes if you can provide a quote.</Say>
  </Gather>
  <Say>Thank you. We will follow up by email.</Say>
  <Hangup/>
</Response>'''.strip()
    return PlainTextResponse(xml, media_type='application/xml')

@app.get('/api/call/{call_id}')
async def get_call(call_id: str):
    return CALLS.get(call_id, {'error':'not_found'})
