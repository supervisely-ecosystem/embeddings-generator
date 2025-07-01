import os

import supervisely as sly
from dotenv import load_dotenv

from src.utils import get_app_host

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api.from_env(ignore_task_id=True)
sly.logger.debug("Connected to Supervisely API: %s", api.server_address)
api.file.load_dotenv_from_teamfiles(override=True)
clip_slug = "bb17e33346af1da8b807ab86b714cf9a/clip-service"  #! replace after release

# region envvars
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
sly.logger.debug("Team ID: %s, Workspace ID: %s", team_id, workspace_id)


qdrant_host = os.getenv("modal.state.qdrantHost") or os.getenv("QDRANT_HOST")
clip_host = os.getenv("modal.state.clipHost", None) or os.getenv("CLIP_HOST", None)
if clip_host is None:
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
