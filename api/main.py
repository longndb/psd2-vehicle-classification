"""FastAPI application for Vehicle Classification API."""

import io
import logging
from typing import Union

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    PredictionResponse,
    ErrorResponse,
    HealthResponse,
)
from api.predictor import get_predictor, VehiclePredictor, WINDOW_SECONDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("vehicle-api")

# API Version
API_VERSION = "1.0.0"

# FastAPI app with OpenAPI configuration
app = FastAPI(
    title="Vehicle Classification API",
    description="""
## Vehicle Transportation Mode Classification API

This API predicts the mode of transportation (bike or car) from accelerometer sensor data.

### Features
- **CSV Upload**: Upload sensor data in CSV format
- **Segment Analysis**: Get predictions for each time segment (configurable window size)
- **Overall Prediction**: Majority voting across all segments
- **Distribution Analysis**: See the breakdown of predictions across modes

### Configuration
- **WINDOW_SECONDS** (env variable): Duration of each analysis window in seconds (default: 2.0)

### CSV Format Requirements
The uploaded CSV must contain the following columns:
- `time`: Timestamp in seconds
- `ax`: Accelerometer X-axis reading
- `ay`: Accelerometer Y-axis reading  
- `az`: Accelerometer Z-axis reading

Additional columns are allowed but will be ignored.

### Model
Uses a Convolutional Neural Network (CNN) trained on accelerometer data to classify transportation modes.
""",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "Prediction",
            "description": "Endpoints for transportation mode prediction"
        },
        {
            "name": "Health",
            "description": "Service health and status endpoints"
        }
    ]
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: VehiclePredictor = None


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global predictor
    logger.info("Starting Vehicle Classification API...")
    logger.info(f"Window size: {WINDOW_SECONDS} seconds")
    try:
        logger.info("Loading model...")
        predictor = get_predictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check if the service is running and the model is loaded."
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor.is_loaded(),
        version=API_VERSION,
        window_seconds=WINDOW_SECONDS
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse, "description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    tags=["Prediction"],
    summary="Predict Transportation Mode",
    description="""
Upload a CSV file containing accelerometer sensor data to predict the transportation mode.

The CSV must contain columns: `time`, `ax`, `ay`, `az`

Returns:
- Overall prediction based on majority voting
- Per-segment predictions with probability distributions
- Distribution of predictions across all modes (bike, car)
"""
)
async def predict(
    file: UploadFile = File(
        ...,
        description="CSV file with accelerometer data (columns: time, ax, ay, az)"
    )
) -> Union[PredictionResponse, JSONResponse]:
    """Predict transportation mode from CSV accelerometer data."""
    global predictor
    
    logger.info(f"Received file upload: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        logger.warning(f"Invalid file type rejected: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Invalid file type",
                "details": "Only CSV files are accepted"
            }
        )
    
    # Ensure predictor is loaded
    if predictor is None:
        try:
            logger.info("Lazy loading model...")
            predictor = get_predictor()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "Model not available",
                    "details": str(e)
                }
            )
    
    try:
        # Read CSV content
        logger.info(f"Reading CSV file: {file.filename}")
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        logger.info(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
        
        # Run prediction
        logger.info("Starting prediction...")
        result = predictor.predict(df)
        logger.info(f"Prediction complete: {result['total_segments']} segments, "
                   f"overall={result['overall_prediction']['label']} "
                   f"(confidence={result['overall_prediction']['confidence']:.2%})")
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.error(f"Invalid CSV data: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Invalid CSV data",
                "details": str(e)
            }
        )
    except pd.errors.EmptyDataError:
        logger.error("Empty CSV file uploaded")
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Empty CSV file",
                "details": "The uploaded CSV file is empty"
            }
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Prediction failed",
                "details": str(e)
            }
        )


@app.get(
    "/",
    tags=["Health"],
    summary="Root",
    description="Root endpoint with API information."
)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Vehicle Classification API",
        "version": API_VERSION,
        "window_seconds": WINDOW_SECONDS,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

