from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from detection_model import VideoClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
video_classifier = VideoClassifier('detection_model.h5')

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Video for Prediction</title>
    </head>
    <body>
        <h1>Upload a Video for Prediction</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="video/*" required>
            <button type="submit">Upload and Predict</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4", "video/avi", "video/mov"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
        temp_file.write(await file.read())
        temp_file.flush()

        try:
            final_class, max_probability = video_classifier.process_video(temp_file.name)
            
            # Base response content
            response = {
                # "For this video, model computed with a probability of": max_probability,
                "training_accuracy": "96.15%",
                # "The validation accuracy of the model in % is": "91.67%"
            }
            
            # Add dynamic message based on the final class
            if final_class == 'compressed':
                response["remark"] = "Part is defective."
            elif final_class == 'extended':
                response["remark"] = "Part is in working condition."
            elif final_class == 'uncertain':
                response["remark"] = "Part may be defective."
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    return JSONResponse(content=response)
