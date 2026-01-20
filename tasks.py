import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "group04"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain", target: str = "all") -> None:
    """Build docker images (target: 'train', 'api', 'frontend', 'drift', 'all')."""
    if target in ("train", "all"):
        ctx.run(
            f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
            echo=True,
            pty=not WINDOWS,
        )
    if target in ("api", "all"):
        ctx.run(
            f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
            echo=True,
            pty=not WINDOWS,
        )
    if target in ("frontend", "all"):
        ctx.run(
            f"docker build -t frontend:latest frontend -f frontend/frontend.dockerfile --progress={progress}",
            echo=True,
            pty=not WINDOWS,
        )
    if target in ("drift", "all"):
        ctx.run(
            f"docker build -t drift:latest . -f dockerfiles/drift.dockerfile --progress={progress}",
            echo=True,
            pty=not WINDOWS,
        )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# GCP Deployment commands
@task
def build_and_push_gcp(
    ctx: Context,
    service: str = "all",
    region: str = "europe-north1",
    ar_repo: str = "ml-images",
    progress: str = "plain",
) -> None:
    project_id = ctx.run("gcloud config get-value project", hide=True).stdout.strip()

    if service in ("api", "all"):
        api_image = f"{region}-docker.pkg.dev/{project_id}/{ar_repo}/api-image:latest"
        ctx.run(
            f"docker build -t {api_image} . -f dockerfiles/api.dockerfile --progress={progress}",
            echo=True,
            pty=not WINDOWS,
        )
        ctx.run(f"docker push {api_image}", echo=True, pty=not WINDOWS)

    if service in ("frontend", "all"):
        frontend_image = f"{region}-docker.pkg.dev/{project_id}/{ar_repo}/frontend-image:latest"
        ctx.run(
            f"docker build -t {frontend_image} frontend -f frontend/frontend.dockerfile --progress={progress}",
            echo=True,
            pty=not WINDOWS,
        )
        ctx.run(f"docker push {frontend_image}", echo=True, pty=not WINDOWS)


@task
def deploy_gcp(
    ctx: Context,
    service: str = "all",
    region: str = "europe-north1",
    ar_repo: str = "ml-images",
) -> None:
    """Deploy services to GCP Cloud Run.

    Args:
        service: 'api', 'frontend', or 'all'
        region: GCP region (default: europe-north1)
        ar_repo: Artifact Registry repository name (default: ml-images)
    """
    project_id = ctx.run("gcloud config get-value project", hide=True).stdout.strip()

    if service in ("api", "all"):
        api_image = f"{region}-docker.pkg.dev/{project_id}/{ar_repo}/api-image:latest"
        ctx.run(
            f"gcloud run deploy production-api --image {api_image} --region {region} "
            "--allow-unauthenticated --set-env-vars PORT=8000",
            echo=True,
            pty=not WINDOWS,
        )

    if service in ("frontend", "all"):
        frontend_image = f"{region}-docker.pkg.dev/{project_id}/{ar_repo}/frontend-image:latest"
        ctx.run(
            f"gcloud run deploy production-frontend --image {frontend_image} --region {region} "
            "--allow-unauthenticated --set-env-vars PORT=8501",
            echo=True,
            pty=not WINDOWS,
        )


# Drift Detection commands
@task
def check_drift(ctx: Context) -> None:
    """Run drift detection locally."""
    ctx.run("uv run python -m src.drift_detection", echo=True, pty=not WINDOWS)


@task
def deploy_drift_detection(ctx: Context) -> None:
    """Deploy drift detection service to GCP."""
    ctx.run("gcloud builds submit --config=drift_cloudbuild.yaml", echo=True, pty=not WINDOWS)


@task
def setup_drift_scheduler(ctx: Context) -> None:
    """Setup Cloud Scheduler for automated drift detection."""
    project_id = ctx.run("gcloud config get-value project", hide=True).stdout.strip()
    ctx.run(f"uv run python scripts/setup_drift_scheduler.py {project_id}", echo=True, pty=not WINDOWS)
