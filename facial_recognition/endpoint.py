from fastapi import FastAPI, UploadFile
from PIL import Image
from io import BytesIO
import face_recognition
from classification import classification

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.state.classification = classification.Classification()

def validate_type(image: UploadFile):
    if not image.content_type.startswith("image/"):
        raise ValueError("File is not an image, as far as I can tell from inspecting the MIME type")

@app.post("/image/")
async def create_file(image: UploadFile):
    # validate image type and convert to appropriate type
    validate_type(image)
    image = face_recognition.load_image_file(BytesIO(await image.read()))

    # perform classification
    names = app.state.classification.classify(image)

    return {"names": names}

