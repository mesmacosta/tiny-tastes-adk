frontend:
	PROJECT_ID=$$(gcloud config get-value project) && \
	gcloud beta run deploy tiny-tastes-adk-prod-ui \
		--source . \
		--memory "4Gi" \
		--project $$PROJECT_ID \
		--region "us-central1" \
		--no-allow-unauthenticated \
		--labels "created-by=adk" \
		--set-env-vars \
		"COMMIT_SHA=$(shell git rev-parse HEAD)" \
		$(if $(IAP),--iap) \
		$(if $(PORT),--port=$(PORT))
