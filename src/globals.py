import os

import supervisely as sly
from dotenv import load_dotenv

from src.utils import get_app_host

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

internal_address = os.getenv("SUPERVISELY_API_SERVER_ADDRESS", None)
sly.logger.debug("Internal Supervisely API server address: %s", internal_address)
if internal_address == "":
    internal_address = None
    del os.environ["SUPERVISELY_API_SERVER_ADDRESS"]
    sly.logger.debug("Removed empty SUPERVISELY_API_SERVER_ADDRESS from environment")

api_token = os.getenv("API_TOKEN", None)
sly.logger.debug("API token from environment: %s", api_token)
if api_token == "":
    api_token = None
    del os.environ["API_TOKEN"]
    sly.logger.debug("Removed empty API_TOKEN from environment")

access_token = os.getenv("SUPERVISELY_API_ACCESS_TOKEN", None)
sly.logger.debug("Access token from environment: %s", access_token)
if access_token == "":
    access_token = None

if access_token is None and internal_address is not None and api_token is None:
    message = (
        f"You are trying to connect to the internal Supervisely API server: {internal_address}. "
        "Access token or API token is required to connect to the internal Supervisely API server, "
        "but it is not provided in the environment variables."
    )
    sly.logger.error(message)
    raise RuntimeError(message)

if internal_address is not None and api_token is None:
    temp_api = sly.Api(ignore_task_id=True)
    response = temp_api.post("instance.admin-info", data={"accessToken": access_token})
    token = response.json()["apiToken"]
    sly.logger.debug("Using Supervisely API token: %s", token)
elif api_token is not None:
    token = api_token
    sly.logger.debug("Using Supervisely API token from environment")
else:
    token = None
    sly.logger.debug("No internal Supervisely API server address found, using public API")

api = sly.Api(ignore_task_id=True, token=token)
sly.logger.debug("Connected to Supervisely API: %s", api.server_address)
clip_slug = "supervisely-ecosystem/deploy-clip-as-service"

# region envvars
qdrant_host = os.getenv("modal.state.qdrantHost") or os.getenv("QDRANT_HOST")
clip_host = os.getenv("modal.state.clipHost", None) or os.getenv("CLIP_HOST", None)

sly.logger.debug("CLIP host from environment: %s", clip_host)
if clip_host is None or clip_host == "":
    clip_host = get_app_host(api, clip_slug)

try:
    clip_host = int(clip_host)
    task_info = api.task.get_info_by_id(clip_host)
    try:
        clip_host = api.server_address + task_info["settings"]["message"]["appInfo"]["baseUrl"]
    except KeyError:
        sly.logger.warning("Cannot get CLIP URL from task settings")
        raise RuntimeError("Cannot connect to CLIP Service")
except ValueError:
    if clip_host[:4] not in ["http", "ws:/", "grpc"]:
        clip_host = "grpc://" + clip_host
# endregion

if not qdrant_host:
    raise ValueError("QDRANT_HOST is not set in the environment variables")
if not clip_host:
    raise ValueError("CLIP_HOST is not set in the environment variables")

sly.logger.info("Qdrant host: %s", qdrant_host)
sly.logger.info("CLIP host: %s", clip_host)

# region constants
IMAGE_SIZE_FOR_CLIP = 224
# endregion

sly.logger.debug("Image size for CLIP: %s", IMAGE_SIZE_FOR_CLIP)

background_tasks = {}
