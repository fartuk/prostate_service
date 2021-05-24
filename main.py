from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uuid import uuid4
import uvicorn
import json
import os

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)
os.makedirs("data/in", exist_ok=True)
os.makedirs("data/out", exist_ok=True)

queue = []

@app.get("/check")
async def check(key:str):
    return {'status':'ok'}



@app.post("/post_file/")
def post_file(file: bytes = File(...)):

    global queue
    request_id = str(uuid4())
   
    path = 'data/in/{}.tiff'.format(request_id)
    with open(path, 'wb') as f:
        f.write(file)
    
    queue.append(request_id)

    return {"request_id": request_id}


@app.get("/get_result")
async def check(request_id: str):
    path = "data/out/{}".format(request_id)
    if not os.path.exists(path):
        return {'status':'request is not processed'}

    with open(path, "r") as f:
        data = json.load(f)
    
    return data

#from starlette.responses import StreamingResponse
#import io
from fastapi.responses import StreamingResponse

@app.get("/worker/get_task")
def get_task():
    global queue
    if len(queue) == 0:
        return {'status': 'no tasks'}

    request_id = queue[0]
    queue = queue[1:]

    #return StreamingResponse(open("conc.png", 'rb'), media_type="image/png")

    #return StreamingResponse(open("018eabc8f4503ab89d0725b430e4808f.tiff", 'rb'), media_type='image/png')

    return StreamingResponse(open("data/in/{}.tiff".format(request_id), 'rb'),
                             media_type='image/png',
                             headers={'Content-Disposition': request_id})


    #return FileResponse("data/in/{}".format(request_id))


@app.post("/worker/post_result")
async def post_result(request: Request):
    data = await request.json()
    request_id = data['request_id']
    path = 'data/out/{}'.format(request_id)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
        
    return data



