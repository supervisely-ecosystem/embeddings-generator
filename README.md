<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/poster.jpg">

# Embeddings Generator

**Microservice for generating embeddings using CLIP as Service and storing them in Qdrant vector database**

<p align="center">
  <a href="#Overview">Overview</a> •
    <a href="#How-To-Run">How To Run</a> •
    <a href="#How-To-Use">How To Use</a> •
    <a href="#API-Endpoints">API Endpoints</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/embeddings-generator)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/embeddings-generator)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/embeddings-generator.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/embeddings-generator.png)](https://supervisely.com)

</div>

## Overview

Embeddings Generator is a headless Supervisely microservice that provides high-performance vector embeddings generation for images using CLIP (Contrastive Language-Image Pre-Training) technology. The service automatically generates embeddings for project images and stores them in Qdrant vector database, enabling powerful semantic search, similarity analysis, and diverse image selection capabilities.

The service operates as a background microservice and integrates seamlessly with Supervisely ecosystem, providing RESTful API endpoints for embeddings generation, semantic search, and advanced image analysis workflows.

### Key Features

- **Text-to-Image Search**: Find images using natural language descriptions.
- **Image-to-Image Search**: Discover visually similar images in your dataset.
- **Hybrid Search**: Combine text prompts and reference images for precise results.
- **Diverse Selection**: Use clustering algorithms to select diverse image subsets.

### Core Technologies

- **CLIP as Service**: Utilizes state-of-the-art CLIP models for generating high-quality image embeddings.
- **Qdrant Vector Database**: Efficiently stores and manages embeddings for high-performance retrieval.
- **Supervisely Ecosystem**: Integrates with Supervisely platform for seamless project management
- **RESTful API**: Provides simple HTTP endpoints for easy integration with external systems and workflows.
- **Background Processing**: Runs as a headless service with automatic updating of embeddings as new images are added to projects.

## How To Run

**Prerequisites:**

- Supervisely instance with admin access
- Running CLIP as Service instance (task ID)
- Qdrant vector database instance (URL)
- Recommended: Run the Embeddings Auto-Updater app to keep embeddings up-to-date automatically.

When launching the service, configure these settings in the modal dialog:

1. **Qdrant DB**: Full URL including protocol (https/http) and port.
2. **CLIP Service**: Task ID for CLIP as Service session.
3. **Change App availability** to "Whole Instance" to make it accessible for all projects. If you forget to do this, you can change it later in the app session settings.

After configuration, click "Run" to deploy the service. The application will start in headless mode and will be available for all projects in your Supervisely instance.

![How To Run](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/how_to_run.jpg)

## How To Use

For each project, you want to use the AI Search feature you need to enable this feature:

|                                                                                                                            |                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| ![Enable AI Search 1](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/enable_1.jpg) | ![Enable AI Search 2](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/enable_2.jpg) |

After enabling the AI Search feature, embeddings will be generated automatically for all images in the project, it may take some time depending on the number of images.

|                                                                                                                                   |                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| ![Preparing Embeddings 1](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/preparing_1.jpg) | ![Preparing Embeddings 2](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/preparing_2.jpg) |

Once embeddings are generated, you can use the semantic search and diverse selection features:

**Semantic Search**

- Use text prompts to find similar images in your project:

![Prompt](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/prompt.jpg)

- Use reference images to find visually similar images:

| Select Reference Images                                                                                                      | Results                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ![Reference Image 1](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/reference_1.jpg) | ![Reference Image 2](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/reference_2.jpg) |

When results are returned, you can see the confidence scores for each image, indicating how similar they are to the search query. You can adjust the slider to filter results based on confidence:

![Confidence Adjustment](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/confidence.gif)

**Diverse Selection**

Use clustering algorithms to select diverse images from your project:

![Diverse Selection](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/diverse.jpg)

## API Endpoints

The service provides three main API endpoints:

### `/embeddings` - Generate Embeddings

Generate or update embeddings for project images.

**Examples:**

```python
# Generate embeddings for all images in project
data = {"project_id": 12345, "team_id": 67890}
api.task.send_request(task_id, "embeddings", data, skip_response=True)

# Generate embeddings for specific images
data = {"image_ids": [101, 102, 103], "team_id": 67890}
api.task.send_request(task_id, "embeddings", data, skip_response=True)

# Force regeneration of all embeddings
data = {"project_id": 12345, "team_id": 67890, "force": True}
api.task.send_request(task_id, "embeddings", data, skip_response=True)
```

### `/search` - Semantic Search

Search for similar images using text prompts or reference images.

**Examples:**

```python
# Text-to-image search
data = {"project_id": 12345, "limit": 50, "prompt": "red car on highway"}
response = api.task.send_request(task_id, "search", data)

# Image-to-image search
data = {"project_id": 12345, "limit": 20, "image_ids": [101, 102]}
response = api.task.send_request(task_id, "search", data)

# Hybrid search (text + images)
data = {
    "project_id": 12345,
    "limit": 30,
    "prompt": "sports car",
    "image_ids": [101]
}
response = api.task.send_request(task_id, "search", data)
```

### `/diverse` - Diverse Selection

Select diverse images using clustering algorithms.

**Examples:**

```python
# K-means diverse selection
data = {"project_id": 12345, "method": "kmeans", "limit": 100}
response = api.task.send_request(task_id, "diverse", data)

# Random diverse selection
data = {"project_id": 12345, "method": "random", "limit": 50}
response = api.task.send_request(task_id, "diverse", data)
```

---

For technical support and questions, please join our [Supervisely Ecosystem Slack community](https://supervisely.com/slack).
