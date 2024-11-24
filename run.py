from app.main import app
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting Flask application")
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False  # Set to False in production
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
