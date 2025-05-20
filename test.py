import os
from collections import defaultdict

import dotenv
import supervisely as sly
from supervisely.api.entities_collection_api import CollectionTypeFilter
from supervisely.app.widgets import (
    Bokeh,
    Button,
    Card,
    Checkbox,
    Container,
    GridGallery,
    IFrame,
    Input,
    InputNumber,
    RadioTabs,
    Select,
    SelectDataset,
    SelectProject,
    Text,
)

dotenv.load_dotenv(os.path.expanduser("~/supervisely.env"))
dotenv.load_dotenv("local.env")

api = sly.Api()


embeddings_generator_task_id = os.getenv("TASK_ID")
project_id = os.getenv("PROJECT_ID")
workspace_id = os.getenv("WORKSPACE_ID")
team_id = os.getenv("TEAM_ID")

current_items = []

bokeh = Bokeh([])
bokeh_iframe = IFrame()
bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="100%")
select_project = SelectProject(default_id=project_id, compact=False, workspace_id=workspace_id)
select_dataset = SelectDataset(default_id=None, compact=True, project_id=project_id)
prompt_search = Input(placeholder="Search query")
image_search = Input(placeholder="Image IDs")
gallery = GridGallery(5)
prompt_search_button = Button("Prompt Search")
image_search_button = Button("Image Search")
load_project = Button("Generate")
update_payload = Button("Update Payload")
force_checkbox = Checkbox(content="Force update")
tabs = RadioTabs(
    titles=["Gallery", "Plot"],
    contents=[gallery, bokeh_iframe],
)
clusters_button = Button("Clusters")
limit_text = Text("<b>Limit</b>")
limit_size_input = InputNumber(value=20, size="medium")
limit_container = Container(widgets=[limit_text, limit_size_input])

diverse_button = Button("Diverse")

clustering_text = Text("<b>Clustering Method</b>")
clustering_method_selector = Select(
    items=[Select.Item("kmeans", "KMeans"), Select.Item("dbscan", "DBSCAN")],
    placeholder="Clustering method",
)
clustering_container = Container(widgets=[clustering_text, clustering_method_selector])
sampling_text = Text("<b>Sampling Method</b>")
sampling_method_selector = Select(
    items=[Select.Item("random", "Random"), Select.Item("centroids", "Centroids")],
    placeholder="Sampling method",
)
sampling_container = Container(widgets=[sampling_text, sampling_method_selector])
methods_container = Container(
    widgets=[clustering_container, sampling_container, limit_container],
    direction="horizontal",
)


def sort_by_score(image_infos):
    """
    Sorts image_infos by score in descending order.
    """
    return sorted(image_infos, key=lambda x: x.ai_search_meta.get("score", 0), reverse=True)


def draw_all_projections(project_id, force):
    global current_items

    bokeh_iframe.loading = True
    print("=====================================")
    print("Embeddings Generator Task ID and Project ID")
    print("task_id:", embeddings_generator_task_id)
    print("project_id:", project_id)
    print("=====================================")
    print("Sending request to update_embeddings...")

    try:
        r = api.task.send_request(
            embeddings_generator_task_id,
            "embeddings",
            data={"project_id": project_id, "force": force, "return_vectors": True},
        )
        image_ids = r.get("image_ids")
    except:
        pass

    # print("=====================================")
    # print("Sending request to get projections...")

    # r = api.task.send_request(
    #     embeddings_generator_task_id, "projections", data={"project_id": project_id}
    # )
    # infos, projections = r
    # print(f"Got {len(projections)} projections")

    # print("=====================================")
    # print("Init bokeh widget")
    # bokeh.clear()
    # plot = Bokeh.Circle(
    #     x_coordinates=[projection[1] for projection in projections],
    #     y_coordinates=[projection[0] for projection in projections],
    #     colors=["#222222"] * len(projections),
    #     legend_label="Images",
    #     plot_id=1,
    #     radii=[0.05] * len(projections),
    # )
    # bokeh.add_plots([plot])
    # current_items = list(zip(infos, projections))
    # bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="100%")
    # bokeh_iframe.loading = False


def draw_projections_per_prompt(project_id, prompt, limit=20, dataset_id=None):
    gallery.loading = True
    bokeh_iframe.loading = True

    print("=====================================")
    print("Sending request to search...")
    r = api.task.send_request(
        embeddings_generator_task_id,
        "search",
        data={
            "project_id": project_id,
            "prompt": prompt,
            "limit": limit,
            "dataset_id": dataset_id,
        },
    )
    image_collection_id = r.get("collection_id")
    image_infos = api.entities_collection.get_items(
        image_collection_id, CollectionTypeFilter.AI_SEARCH
    )
    image_infos = sort_by_score(image_infos)
    print(f"Got {len(image_infos)} images")
    print("=====================================")

    # print("Init bokeh widget")
    # this_ids = set([info.id for info in image_infos])
    # bokeh.clear()
    # plot = Bokeh.Circle(
    #     x_coordinates=[item[1][1] for item in current_items if item[0]["id"] not in this_ids],
    #     y_coordinates=[item[1][0] for item in current_items if item[0]["id"] not in this_ids],
    #     colors=["#222222" for item in current_items if item[0]["id"] not in this_ids],
    #     legend_label="Images",
    #     plot_id=1,
    #     radii=[0.05 for item in current_items if item[0]["id"] not in this_ids],
    # )
    # new_plot = Bokeh.Circle(
    #     x_coordinates=[item[1][1] for item in current_items if item[0]["id"] in this_ids],
    #     y_coordinates=[item[1][0] for item in current_items if item[0]["id"] in this_ids],
    #     colors=["#006400" for item in current_items if item[0]["id"] in this_ids],
    #     legend_label=prompt,
    #     radii=[0.05 for item in current_items if item[0]["id"] in this_ids],
    # )
    # bokeh.add_plots([plot, new_plot])
    # bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="600px")
    # bokeh_iframe.loading = False

    print("=====================================")
    print("init Gallery")
    gallery.clean_up()
    # project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    for i, info in enumerate(image_infos, 1):
        # ann_info = api.annotation.download(info.id)
        score = info.ai_search_meta.get("score")
        gallery.append(
            info.full_storage_url,
            # ann_info,
            # project_meta,
            title=f"ID: {info.id}, Score: {score}",
            # call_update=False,
        )
        print(f"image {i}/{len(image_infos)} added to gallery")
    gallery._update()
    gallery.loading = False


def draw_projections_per_ids(project_id, ids, limit=20):
    gallery.loading = True
    bokeh_iframe.loading = True

    print("=====================================")
    print("Sending request to search...")
    r = api.task.send_request(
        embeddings_generator_task_id,
        "search",
        data={
            "project_id": project_id,
            "by_image_ids": ids,
            "limit": limit,
            "dataset_id": 964,
        },
    )
    image_collection_id = r.get("collection_id")
    image_infos = api.entities_collection.get_items(
        image_collection_id, CollectionTypeFilter.AI_SEARCH
    )
    image_infos = sort_by_score(image_infos)
    print(f"Got {len(image_infos)} images")
    print("=====================================")

    # print("Init bokeh widget")
    # this_ids = set([info.id for info in image_infos])
    # bokeh.clear()
    # plot = Bokeh.Circle(
    #     x_coordinates=[item[1][1] for item in current_items if item[0]["id"] not in this_ids],
    #     y_coordinates=[item[1][0] for item in current_items if item[0]["id"] not in this_ids],
    #     colors=["#222222" for item in current_items if item[0]["id"] not in this_ids],
    #     legend_label="Images",
    #     plot_id=1,
    #     radii=[0.05 for item in current_items if item[0]["id"] not in this_ids],
    # )
    # new_plot = Bokeh.Circle(
    #     x_coordinates=[item[1][1] for item in current_items if item[0]["id"] in this_ids],
    #     y_coordinates=[item[1][0] for item in current_items if item[0]["id"] in this_ids],
    #     colors=["#006400" for item in current_items if item[0]["id"] in this_ids],
    #     legend_label=ids,
    #     radii=[0.05 for item in current_items if item[0]["id"] in this_ids],
    # )
    # bokeh.add_plots([plot, new_plot])
    # bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="600px")
    # bokeh_iframe.loading = False

    print("=====================================")
    print("init Gallery")
    gallery.clean_up()
    # project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    for i, info in enumerate(image_infos, 1):
        # ann_info = api.annotation.download(info.id)
        score = info.ai_search_meta.get("score")
        gallery.append(
            info.full_storage_url,
            # ann_info,
            title=f"ID: {info.id}, Score: {score}",
            # project_meta,
            # call_update=False,
        )
        print(f"image {i}/{len(image_infos)} added to gallery")
    gallery._update()
    gallery.loading = False


def draw_clusters(project_id):
    gallery.loading = True
    bokeh_iframe.loading = True

    print("=====================================")
    print("Sending request to clusters...")
    r = api.task.send_request(
        embeddings_generator_task_id,
        "clusters",
        data={"project_id": project_id, "reduce": True},
    )
    image_infos, labels = r
    print(f"Got {len(image_infos)} images")
    print("=====================================")

    print("Clusters:")
    for label, count in defaultdict(int, {l: labels.count(l) for l in set(labels)}).items():
        print(f"Cluster #{label}: {count} images")
    print("=====================================")

    print("Init bokeh widget")

    unique_labels = set(labels)
    plots = []
    predefined_colors = sly.color.get_predefined_colors(len(unique_labels) - 1)
    predefined_hex_colors = [sly.color.rgb2hex(color) for color in predefined_colors]
    colors = ["#222222", *predefined_hex_colors]
    for label, color in zip(sorted(unique_labels), colors):
        this_ids = [info["id"] for info, l in zip(image_infos, labels) if l == label]
        n = len(this_ids)
        plot = Bokeh.Circle(
            x_coordinates=[item[1][1] for item in current_items if item[0]["id"] in this_ids],
            y_coordinates=[item[1][0] for item in current_items if item[0]["id"] in this_ids],
            colors=[color] * n,
            legend_label="Outliers" if label == -1 else "Cluster #" + str(label + 1),
            plot_id=label,
            radii=[0.05] * n,
        )
        plots.append(plot)

    gallery.clean_up()
    bokeh.clear()
    bokeh.add_plots(plots)
    bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="600px")
    bokeh_iframe.loading = False
    gallery._update()
    gallery.loading = False


def draw_diverse(project_id, sample_size, clustering_method, sampling_method):
    gallery.loading = True
    bokeh_iframe.loading = True
    print("=====================================")
    print("Sending request to diverse...")
    r = api.task.send_request(
        embeddings_generator_task_id,
        "diverse",
        data={
            "project_id": project_id,
            "sampling_method": sampling_method,
            "sample_size": sample_size,
            "clustering_method": clustering_method,
            # "image_ids": [22862],
        },
    )
    image_collection_id = r.get("collection_id")
    image_infos = api.entities_collection.get_items(
        image_collection_id, CollectionTypeFilter.AI_SEARCH
    )
    print(f"Got {len(image_infos)} images")
    print("=====================================")

    # print("Init bokeh widget")

    # this_ids = set([info.id for info in image_infos])
    # bokeh.clear()
    # plot = Bokeh.Circle(
    #     x_coordinates=[item[1][1] for item in current_items if item[0]["id"] not in this_ids],
    #     y_coordinates=[item[1][0] for item in current_items if item[0]["id"] not in this_ids],
    #     colors=["#222222" for item in current_items if item[0]["id"] not in this_ids],
    #     legend_label="Images",
    #     plot_id=1,
    #     radii=[0.05 for item in current_items if item[0]["id"] not in this_ids],
    # )
    # new_plot = Bokeh.Circle(
    #     x_coordinates=[item[1][1] for item in current_items if item[0]["id"] in this_ids],
    #     y_coordinates=[item[1][0] for item in current_items if item[0]["id"] in this_ids],
    #     colors=["#006400" for item in current_items if item[0]["id"] in this_ids],
    #     legend_label="Sample",
    #     radii=[0.05 for item in current_items if item[0]["id"] in this_ids],
    # )
    # bokeh.add_plots([plot, new_plot])
    # bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="600px")
    # bokeh_iframe.loading = False

    print("=====================================")
    print("init Gallery")
    gallery.clean_up()
    # project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    for i, info in enumerate(image_infos, 1):
        # ann_info = api.annotation.download(info.id)
        score = info.ai_search_meta.get("score")
        gallery.append(
            info.full_storage_url,
            # ann_info,
            # project_meta,
            title=f"ID: {info.id}, Score: {score}",
            # call_update=False,
        )
        print(f"image {i}/{len(image_infos)} added to gallery")
    gallery._update()
    gallery.loading = False


def update_project_payload(project_id):
    print("=====================================")
    print("Sending request to update_payload...")
    r = api.task.send_request(
        embeddings_generator_task_id,
        "update_embeddings_payload",
        data={"project_id": project_id},
    )
    print(f"Got {len(r)} images")
    print("=====================================")


@load_project.click
def load_project_click():
    project_id = select_project.get_selected_id()
    force = force_checkbox.is_checked()
    draw_all_projections(project_id, force=force)


@update_payload.click
def update_payload_click():
    project_id = select_project.get_selected_id()
    update_project_payload(project_id)


@prompt_search_button.click
def search_click():
    prompt = prompt_search.get_value()
    project_id = select_project.get_selected_id()
    dataset_id = select_dataset.get_selected_id()
    limit = limit_size_input.get_value()
    draw_projections_per_prompt(project_id, prompt, dataset_id=dataset_id, limit=limit)


@image_search_button.click
def image_search_click():
    ids = image_search.get_value()
    ids = [int(i) for i in ids.split(", ")]
    project_id = select_project.get_selected_id()
    limit = limit_size_input.get_value()
    draw_projections_per_ids(project_id, ids, limit=limit)


@clusters_button.click
def clusters_click():
    project_id = select_project.get_selected_id()
    draw_clusters(project_id)


@diverse_button.click
def diverse_click():
    project_id = select_project.get_selected_id()
    sample_size = limit_size_input.get_value()
    clustering_method = clustering_method_selector.get_value()
    sampling_method = sampling_method_selector.get_value()
    draw_diverse(project_id, sample_size, clustering_method, sampling_method)


@select_project.value_changed
def select_project_change(project_id):
    select_dataset.set_project_id(project_id)


card_0 = Card(
    title="0️⃣ Utility",
    content=Container(widgets=[update_payload]),
    description="This section is for utility functions",
)

card_1 = Card(
    title="1️⃣ Generate Embeddings",
    content=Container(widgets=[load_project, force_checkbox]),
    description="You must generate embeddings for project before searching",
)

container_card_0_1 = Container(widgets=[card_0, card_1], direction="horizontal")

card_2 = Card(
    title="2️⃣ Search",
    content=Container(
        widgets=[
            Container(
                widgets=[prompt_search, limit_size_input, select_dataset, prompt_search_button],
                style="flex: 1 1 auto;/* display: flex; */",
            ),
            Container(
                widgets=[image_search, limit_size_input, image_search_button],
                style="flex: 1 1 auto;/* display: flex; */",
            ),
        ],
        direction="horizontal",
    ),
)
card_3 = Card(
    title="3️⃣ Diverse",
    content=Container(widgets=[methods_container, diverse_button]),
)
card_4 = Card(title="4️⃣ Clusters", content=Container(widgets=[clusters_button]))
container_cards_3_4 = Container(
    widgets=[card_3, card_4],
    direction="horizontal",
)
layout = Container(widgets=[select_project, container_card_0_1, card_2, container_cards_3_4, tabs])
app = sly.Application(layout=layout)
