from fastapi import FastAPI, File, UploadFile
import io
import torch
from torchsummary import summary
import uvicorn
from model import UNET, prepare_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from fastapi.responses import Response, FileResponse, StreamingResponse
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET(3,3)
model.load_state_dict(torch.load("./model_params/best.pth", map_location = device))
model.eval()

@app.post("/uploadfile/")
def create_upload_file(img: UploadFile, mask: UploadFile) :
    image = prepare_image(img, mask)

    pil_image = transforms.ToPILImage()(image[0])
    pil_image.save("before.png")

    output = model(image)

    # Step 1: Convert the PyTorch tensor to a PIL Image
    image_tensor = torch.clamp(output[0], 0,1)
    image = transforms.ToPILImage()(image_tensor)
    # fn = f"{img.filename}.png"
    # image.save(fn)

    # Step 2: Convert the PIL Image to bytes
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="PNG")
    # img_bytes = img_byte_array.getvalue()
    img_byte_array.seek(0)

    # Step 3: Return the image bytes as a StreamingResponse
    return StreamingResponse(img_byte_array, media_type="image/png")


# if __name__ == "__main__":
#     uvicorn.run("main:app", port=8000, log_level="info", reload=True)
