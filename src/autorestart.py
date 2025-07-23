import datetime
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import supervisely as sly
from supervisely import logger

from src.utils import CustomDataFields, to_thread, set_embeddings_in_progress, clear_update_flag

STATE_DIR: str = "./state/in_progress"
PROJECT_FILE_PATTERN = "project_{project_id}.json"


@dataclass
class ProjectTaskInfo:
    project_id: int
    timestamp: str
    custom_data_flag: str

    def to_dict(self) -> Dict:
        return {
            "project_id": self.project_id,
            "timestamp": self.timestamp,
            "custom_data_flag": self.custom_data_flag,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProjectTaskInfo":
        return cls(
            project_id=data["project_id"],
            timestamp=data["timestamp"],
            custom_data_flag=data["custom_data_flag"],
        )


class EmbeddingsTaskManager:
    """Manages tracking of embeddings creation tasks for crash recovery."""

    @staticmethod
    def _get_project_file_path(project_id: int) -> str:
        """Get file path for a specific project."""
        filename = PROJECT_FILE_PATTERN.format(project_id=project_id)
        return os.path.join(STATE_DIR, filename)

    @staticmethod
    def _ensure_state_dir() -> None:
        """Ensure state directory exists."""
        os.makedirs(STATE_DIR, exist_ok=True)

    @staticmethod
    @to_thread
    def create_task_file(project_id: int) -> None:
        """Create a file to track embeddings task start."""
        try:
            EmbeddingsTaskManager._ensure_state_dir()

            # Create timestamp and custom data flag
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            custom_data_flag = CustomDataFields.EMBEDDINGS_UPDATE_STARTED_AT

            task_info = ProjectTaskInfo(
                project_id=project_id, timestamp=timestamp, custom_data_flag=custom_data_flag
            )

            file_path = EmbeddingsTaskManager._get_project_file_path(project_id)

            with open(file_path, "w") as f:
                json.dump(task_info.to_dict(), f, indent=2)

            logger.debug(
                f"Created embeddings task file for project {project_id}",
                extra={"file_path": file_path, "timestamp": timestamp},
            )

        except Exception as e:
            logger.error(f"Failed to create task file for project {project_id}: {e}", exc_info=True)

    @staticmethod
    @to_thread
    def remove_task_file(project_id: int) -> None:
        """Remove task file when embeddings creation is complete."""
        try:
            file_path = EmbeddingsTaskManager._get_project_file_path(project_id)

            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(
                    f"Removed embeddings task file for project {project_id}",
                    extra={"file_path": file_path},
                )
            else:
                logger.debug(
                    f"Task file for project {project_id} does not exist, nothing to remove."
                )

        except Exception as e:
            logger.error(f"Failed to remove task file for project {project_id}: {e}", exc_info=True)

    @staticmethod
    def get_stuck_projects() -> List[ProjectTaskInfo]:
        """Get list of projects that were in progress when service crashed."""
        try:
            EmbeddingsTaskManager._ensure_state_dir()

            stuck_projects = []
            pattern = os.path.join(STATE_DIR, "project_*.json")

            for file_path in glob.glob(pattern):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    task_info = ProjectTaskInfo.from_dict(data)
                    stuck_projects.append(task_info)

                except Exception as e:
                    logger.error(f"Failed to load task file {file_path}: {e}", exc_info=True)
                    # Try to remove corrupted file
                    try:
                        os.remove(file_path)
                        logger.debug(f"Removed corrupted task file: {file_path}")
                    except:
                        pass

            if stuck_projects:
                logger.info(f"Found {len(stuck_projects)} stuck projects from previous session")
            else:
                logger.debug("No stuck projects found")

            return stuck_projects

        except Exception as e:
            logger.error(f"Failed to get stuck projects: {e}", exc_info=True)
            return []

    @staticmethod
    async def cleanup_stuck_projects(api: sly.Api) -> None:
        """Clean up stuck projects after service restart."""
        try:
            stuck_projects = EmbeddingsTaskManager.get_stuck_projects()

            if not stuck_projects:
                logger.debug("No stuck projects to clean up")
                return

            logger.info(f"Cleaning up {len(stuck_projects)} stuck projects...")

            for task_info in stuck_projects:
                try:
                    project_id = task_info.project_id
                    logger.info(f"Cleaning up stuck project {project_id}")

                    # Reset embeddings in progress flag and clear custom data flag
                    await set_embeddings_in_progress(api, project_id, False)
                    await clear_update_flag(api, project_id)
                    logger.debug(f"Reset flags for project {project_id}")

                    # Remove task file
                    EmbeddingsTaskManager.remove_task_file(project_id)

                    logger.info(f"Successfully cleaned up stuck project {project_id}")

                except Exception as e:
                    logger.error(
                        f"Failed to cleanup project {task_info.project_id}: {e}", exc_info=True
                    )
                    # Still try to remove the file even if API calls failed
                    try:
                        EmbeddingsTaskManager.remove_task_file(task_info.project_id)
                    except:
                        pass

            logger.info("Finished cleaning up stuck projects")

        except Exception as e:
            logger.error(f"Failed to cleanup stuck projects: {e}", exc_info=True)

    @staticmethod
    def clear_all_task_files() -> None:
        """Clear all task files (for testing or emergency cleanup)."""
        try:
            if not os.path.exists(STATE_DIR):
                logger.debug("State directory does not exist, nothing to clear")
                return

            pattern = os.path.join(STATE_DIR, "project_*.json")
            files = glob.glob(pattern)

            for file_path in files:
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed task file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove file {file_path}: {e}")

            logger.info(f"Cleared {len(files)} task files")

        except Exception as e:
            logger.error(f"Failed to clear task files: {e}", exc_info=True)
