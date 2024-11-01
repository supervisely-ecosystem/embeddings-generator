import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api.from_env()
sly.logger.debug(f"Connected to Supervisely API: {api.server_address}.")
api.file.load_dotenv_from_teamfiles(override=True)

# region envvars
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
sly.logger.debug(f"Team ID: {team_id}, Workspace ID: {workspace_id}")

# The first option (modal.state) comes from the Modal window, the second one (QDRANT_HOST)
# comes from the .env file.
qdrant_host = os.getenv("modal.state.qdrantHost") or os.getenv("QDRANT_HOST")
cas_host = os.getenv("modal.state.casHost") or os.getenv("CAS_HOST")
# endregion

if not qdrant_host:
    raise ValueError("QDRANT_HOST is not set in the environment variables")
if not cas_host:
    raise ValueError("CAS_HOST is not set in the environment variables")

sly.logger.info(f"Qdrant host: {qdrant_host}")
sly.logger.info(f"CAS host: {cas_host}")

# region constants
IMAGE_SIZE_FOR_CAS = 224
# endregion

sly.logger.debug(f"Image size for CAS: {IMAGE_SIZE_FOR_CAS}")
