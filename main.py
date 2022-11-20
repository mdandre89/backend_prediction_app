from io import BytesIO
import requests

from fastai.vision import *
import model

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loadedmodels = {}

# TODO solve startup issue
@app.on_event("startup")
async def startup_event():
    learn = await model.setup_learner()
    loadedmodels['learn'] = learn

@app.get("/healthcheck")
def healthcheck():
    return 'OK'

@app.get('/local')
async def local():
    img = open('static/file.jpg', 'rb').read()
    imgraw = BytesIO(img)
    img = open_image((imgraw))
    
    prediction = loadedmodels['learn'].predict(img)[2]
    bests = model.sorted_prob(prediction)

    return {"result": str(bests[0][-1])}

@app.get('/randoms')
async def randoms():
    response = requests.get('https://source.unsplash.com/random/500x500')
    imgraw = BytesIO(response.content)
    img = open_image(imgraw)

    prediction = loadedmodels['learn'].predict(img)[2]
    bests = model.sorted_prob(prediction)

    return {"result": str(bests[0][-1]), 'url': response.url}

@app.post('/predict')
async def analyze(file: UploadFile = File(...)):
    img_bytes = file.file.read()
    img = open_image(BytesIO(img_bytes))

    prediction = loadedmodels['learn'].predict(img)[2]
    bests = model.sorted_prob(prediction)

    return {"result": str(bests[0][-1])}