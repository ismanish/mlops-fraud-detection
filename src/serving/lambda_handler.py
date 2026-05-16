"""AWS Lambda handler wrapping the FastAPI app via Mangum."""

from mangum import Mangum

from src.serving.app import app

handler = Mangum(app, lifespan="on")
