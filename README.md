<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/poster.jpg" alt="Embeddings Generator Poster"/>

# Embeddings Generator

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#api-endpoints">API Endpoints</a>  •
  <a href="#using-ai-search">Using AI Search</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/embeddings-generator)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)

</div>

# Overview

🧩 **Embeddings Generator** is a **system-level microservice** designed to power the **AI Search** feature in Supervisely by providing high-performance vector embeddings generation for images using CLIP (Contrastive Language-Image Pre-Training) technology.

Key features:

-   **Instance-level service**: Runs as a system container for the entire Supervisely instance.
-   **RESTful API**: Provides HTTP endpoints for embeddings generation and semantic search.
-   **CLIP Service integration**: High-quality image embeddings using state-of-the-art CLIP models.
-   **Qdrant integration**: Efficient vector database for embedding storage and retrieval.
-   **Semantic search capabilities**: Text-to-image and image-to-image search functionality.
-   **Diverse selection**: Advanced clustering algorithms for selecting diverse image subsets.
-   **Zero-downtime operation**: Runs continuously in the background as a headless service.

The service enables powerful AI-driven image analysis workflows:

1. **Text-to-Image Search**: Find images using natural language descriptions.
2. **Image-to-Image Search**: Discover visually similar images in datasets.
3. **Hybrid Search**: Combine text prompts and reference images for precise results.
4. **Diverse Selection**: Use clustering algorithms to select diverse image subsets.

## Architecture

The application uses a containerized microservice architecture with RESTful API endpoints:

-   **Containerized Service**: Runs as a Docker container at the instance level.
-   **CLIP Service**: Generates high-quality embeddings using CLIP models.
-   **Qdrant Integration**: Efficiently stores and manages vector embeddings.
-   **RESTful API**: Simple HTTP endpoints for easy integration with external systems.
-   **Background Processing**: Headless service with automatic embedding management.
-   **Multi-project Support**: Handles multiple projects concurrently.

## Deployment

### Prerequisites

-   Supervisely instance with admin access.
-   Docker environment for container deployment.
-   Running CLIP as Service instance (task ID or service endpoint).
-   Qdrant vector database instance (URL).

### Environment Variables

Configure the service using the environment variables in `docker-compose.yml`.

### Configuration

-   **Qdrant DB**: Full URL including protocol (https/http) and port (e.g., `https://192.168.1.1:6333`).
-   **CLIP Service**: Task ID for CLIP as Service session or its host including port (e.g., `1234` or `https://192.168.1.1:51000`).

The service starts automatically on instance startup and provides API endpoints for all projects in the Supervisely instance.

**Recommended**: Deploy alongside the [Embeddings Auto-Updater](https://github.com/supervisely-ecosystem/embeddings-auto-updater) service to keep embeddings up-to-date automatically.

## API Endpoints

The service provides three main API endpoints:

1. `/embeddings` - Generate or update embeddings for project images.

2. `/search` - Semantic search for similar images using text prompts or reference images.

3. `/diverse` - Select diverse images using clustering algorithms.

## Using AI Search

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

-   Use text prompts to find similar images in your project:

![Prompt](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/prompt.jpg)

-   Use reference images to find visually similar images:

| Select Reference Images                                                                                                      | Results                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ![Reference Image 1](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/reference_1.jpg) | ![Reference Image 2](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/reference_2.jpg) |

When results are returned, you can see the confidence scores for each image, indicating how similar they are to the search query. You can adjust the slider to filter results based on confidence:

![Confidence Adjustment](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/confidence.gif)

**Diverse Selection**

Use clustering algorithms to select diverse images from your project:

![Diverse Selection](https://github.com/supervisely-ecosystem/embeddings-generator/releases/download/v0.1.0/diverse.jpg)

---

For technical support and questions, please join our [Supervisely Ecosystem Slack community](https://supervisely.com/slack).
