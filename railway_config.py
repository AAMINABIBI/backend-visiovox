"""
Railway-specific configuration
"""
import os

class RailwayConfig:
    # Railway provides these environment variables
    PORT = int(os.environ.get("PORT", 8080))
    HOST = "0.0.0.0"
    
    # Railway public URL (will be set after deployment)
    RAILWAY_STATIC_URL = os.environ.get("RAILWAY_STATIC_URL", "")
    BASE_URL = RAILWAY_STATIC_URL or f"http://localhost:{PORT}"
    
    # File paths
    WEIGHTS_PATH = os.environ.get(
        "WEIGHTS_PATH",
        "pretrain/LipCoordNet_coords_loss_0.025581153109669685_wer_0.01746208431890914_cer_0.006488426950253695.pt"
    )
    
    # Disable ImageMagick for Railway (use fallback)
    DISABLE_IMAGEMAGICK = os.environ.get("DISABLE_IMAGEMAGICK", "true").lower() == "true"
    
    @classmethod
    def get_base_url(cls):
        """Get the correct base URL for Railway deployment"""
        if cls.RAILWAY_STATIC_URL:
            return cls.RAILWAY_STATIC_URL
        return f"http://localhost:{cls.PORT}"
