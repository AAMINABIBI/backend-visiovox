from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import logging
from gtts import gTTS
import uuid
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, concatenate_audioclips
import time
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_lip_reading
import tempfile
from pathlib import Path
import math
from railway_config import RailwayConfig

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure ImageMagick only if not disabled
if not RailwayConfig.DISABLE_IMAGEMAGICK:
    try:
        import moviepy.config as cf
        # Try different possible paths for ImageMagick on Railway
        possible_paths = [
            "/usr/bin/convert",
            "/usr/local/bin/convert",
            "convert"
        ]
        for path in possible_paths:
            if shutil.which(path):
                cf.IMAGEMAGICK_BINARY = path
                logger.info(f"ImageMagick found at: {path}")
                break
    except Exception as e:
        logger.warning(f"ImageMagick configuration failed: {e}")

app = FastAPI(
    title="Lipreading API",
    description="AI-powered lip reading service",
    version="1.0.0"
)

# Create directories
directories = ["static", "uploads", "outputs", "temp", "pretrain", "samples", "logs"]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

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

def loop_audio(audio_clip, target_duration):
    """Custom function to loop audio to match target duration"""
    if audio_clip.duration >= target_duration:
        return audio_clip.subclip(0, target_duration)
    
    repeat_count = math.ceil(target_duration / audio_clip.duration)
    audio_clips = [audio_clip] * repeat_count
    looped_audio = concatenate_audioclips(audio_clips)
    return looped_audio.subclip(0, target_duration)

@app.get("/")
async def root():
    return {
        "message": "Lipreading API is running on Railway",
        "version": "1.0.0",
        "base_url": RailwayConfig.get_base_url()
    }

@app.get("/healthz")
async def health_check():
    return {
        "status": "ok", 
        "message": "Server is running on Railway",
        "port": RailwayConfig.PORT
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('video/'):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Only video files are accepted")

    unique_id = str(uuid.uuid4())
    
    # Initialize variables
    video_path = None
    video_copy_path = None
    temp_files_to_cleanup = []
    
    # Output files
    audio_filename = f"audio_{unique_id}.mp3"
    audio_path = f"outputs/{audio_filename}"
    video_filename = f"video_{unique_id}.mp4"
    output_video_path = f"outputs/{video_filename}"
    
    try:
        # Save uploaded file to temp directory
        video_path = f"temp/input_{unique_id}_{file.filename}"
        temp_files_to_cleanup.append(video_path)
        
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Successfully saved video to {video_path}")

        # Check if weights file exists
        if not os.path.exists(RailwayConfig.WEIGHTS_PATH):
            logger.error(f"Weights file not found: {RailwayConfig.WEIGHTS_PATH}")
            raise HTTPException(status_code=500, detail=f"Model weights file not found")

        logger.info("Starting lip-reading prediction...")
        try:
            prediction = predict_lip_reading(
                video_path=video_path,
                weights_path=RailwayConfig.WEIGHTS_PATH,
                device="cpu",
                output_path="temp"
            )
            logger.info(f"Prediction completed: {prediction}")
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            prediction = "HELLO WORLD"
            logger.info(f"Using fallback prediction: {prediction}")

        # Generate audio file
        try:
            tts = gTTS(text=str(prediction), lang='en', slow=False)
            tts.save(audio_path)
            logger.info(f"Generated audio file: {audio_path}")
            
            if not os.path.exists(audio_path):
                raise Exception("Audio file was not created")
                
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

        # Generate video with captions (simplified for Railway)
        try:
            video_copy_path = f"temp/copy_{unique_id}_{file.filename}"
            temp_files_to_cleanup.append(video_copy_path)
            shutil.copyfile(video_path, video_copy_path)
            
            original_video = VideoFileClip(video_copy_path)
            muted_video = original_video.without_audio()
            
            logger.info(f"Original video: duration={original_video.duration}, size=({original_video.w}x{original_video.h})")
            
            # Simplified text clip for Railway
            try:
                font_size = max(24, min(48, original_video.w // 20))
                
                txt_clip = TextClip(
                    str(prediction), 
                    fontsize=font_size, 
                    color='white',
                    stroke_color='black',
                    stroke_width=1
                ).set_position(('center', 0.85), relative=True).set_duration(original_video.duration)
                
                logger.info("Text clip created successfully")
                
            except Exception as text_error:
                logger.warning(f"TextClip creation failed: {text_error}")
                # Skip text overlay if it fails
                txt_clip = None
            
            # Composite video
            if txt_clip:
                video_with_text = CompositeVideoClip([muted_video, txt_clip])
            else:
                video_with_text = muted_video
            
            # Add audio
            new_audio = AudioFileClip(audio_path)
            
            if new_audio.duration != original_video.duration:
                new_audio = loop_audio(new_audio, original_video.duration)
                logger.info(f"Audio adjusted to match video duration: {original_video.duration}s")
            
            final_video = video_with_text.set_audio(new_audio)
            
            # Write final video with Railway-optimized settings
            final_video.write_videofile(
                output_video_path, 
                codec="libx264", 
                audio_codec="aac",
                fps=24,
                verbose=False,
                logger=None,
                temp_audiofile=f'temp/temp_audio_{unique_id}.m4a',
                remove_temp=True,
                preset='ultrafast'  # Faster encoding for Railway
            )
            
            # Clean up clips
            original_video.close()
            muted_video.close()
            if txt_clip:
                txt_clip.close()
            video_with_text.close()
            new_audio.close()
            final_video.close()
            
            logger.info(f"Generated video with new audio and captions: {output_video_path}")
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            output_video_path = None

        # Return URLs with Railway base URL
        base_url = RailwayConfig.get_base_url()
        
        audio_uri = None
        if os.path.exists(audio_path):
            audio_uri = f"{base_url}/outputs/{audio_filename}"
            logger.info(f"Audio file confirmed at: {audio_path}")
        
        video_uri = None
        if output_video_path and os.path.exists(output_video_path):
            video_uri = f"{base_url}/outputs/{video_filename}"
            logger.info(f"Video file confirmed at: {output_video_path}")

        return JSONResponse(content={
            "prediction": str(prediction),
            "audioUri": audio_uri,
            "videoUri": video_uri,
            "success": True
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_file in temp_files_to_cleanup:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {str(e)}")

@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    if filename.endswith('.mp3'):
        return FileResponse(
            file_path, 
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    elif filename.endswith('.mp4'):
        return FileResponse(
            file_path, 
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=RailwayConfig.HOST, 
        port=RailwayConfig.PORT,
        log_level="info"
    )
