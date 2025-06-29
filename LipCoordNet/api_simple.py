from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import logging
from gtts import gTTS
import uuid
import time
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def health_check():
    return {"status": "ok", "message": "Server is running on Railway"}

@app.get("/")
async def root():
    return {"message": "Lipreading API is running", "version": "1.0.0"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('video/'):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Only video files are accepted")

    unique_id = str(uuid.uuid4())
    
    try:
        # For now, return a simple prediction without complex processing
        prediction = "HELLO WORLD"
        
        # Generate audio file
        audio_filename = f"audio_{unique_id}.mp3"
        audio_path = f"outputs/{audio_filename}"
        
        try:
            tts = gTTS(text=str(prediction), lang='en')
            tts.save(audio_path)
            logger.info(f"Generated audio file: {audio_path}")
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

        # Get Railway URL from environment
        railway_url = os.environ.get("RAILWAY_STATIC_URL", "http://localhost:8080")
        audio_uri = f"{railway_url}/outputs/{audio_filename}"

        return JSONResponse(content={
            "prediction": str(prediction),
            "audioUri": audio_uri,
            "videoUri": None,  # Skip video processing for now
            "success": True
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if filename.endswith('.mp3'):
        return FileResponse(
            file_path, 
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
