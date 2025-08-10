# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import sys

import google.auth
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.cloud import logging as google_cloud_logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, export

from app.utils.gcs import create_bucket_if_not_exists
from app.utils.tracing import CloudTraceLoggingSpanExporter
from app.utils.typing import Feedback, TranslateRequest
from app.translation_utils import translate_text

_, project_id = google.auth.default()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# logging_client = google_cloud_logging.Client()
# logging_client.setup_logging(log_level="INFO")
# logger = logging_client.logger(__name__)
logger = logging.getLogger(__name__)
# Define the frontend URL that needs to be allowed
frontend_url = "https://tiny-tastes-adk-prod-ui-44603466838.us-central1.run.app"

# Get allowed origins from environment variable and add the frontend URL
allow_origins_env = os.getenv("ALLOW_ORIGINS", "")
allow_origins = set(filter(None, allow_origins_env.split(',')))
allow_origins.add(frontend_url)

bucket_name = f"gs://{project_id}-tiny-tastes-adk-prod-logs-data"
create_bucket_if_not_exists(
    bucket_name=bucket_name, project=project_id, location="us-central1"
)

provider = TracerProvider()
processor = export.BatchSpanProcessor(CloudTraceLoggingSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=bucket_name,
    allow_origins=allow_origins,
)
app.title = "tiny-tastes-adk-prod"
app.description = "API for interacting with the Agent tiny-tastes-adk-prod"


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    # logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


@app.post("/translate")
def translate(request: TranslateRequest) -> dict[str, str]:
    """Translate text to Brazilian Portuguese.

    Args:
        request: The request data with text to translate.

    Returns:
        The translated text.
    """
    translated_text = translate_text(request.text, "pt-BR")
    return {"translated_text": translated_text}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
