from fastapi import FastAPI
import segmentation, objectDetection, cropping
import languageDetection, frenchExtraction, arabicExtraction
import pipeline_arabic, pipeline_french
import fullPipeline
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(segmentation.router, prefix="/segment")
app.include_router(objectDetection.router, prefix="/detect")
app.include_router(cropping.router, prefix="/crop_ordered")
app.include_router(languageDetection.router, prefix="/language-detection")
app.include_router(frenchExtraction.router, prefix="/extract-french-text")
app.include_router(arabicExtraction.router, prefix="/extract-arabic-text")
app.include_router(pipeline_arabic.router, prefix="/pipeline-ar")
app.include_router(pipeline_french.router, prefix="/pipeline-fr")
app.include_router(fullPipeline.router, prefix="/full-pipeline")

