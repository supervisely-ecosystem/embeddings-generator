import os
from collections import defaultdict, namedtuple

import dotenv
import supervisely as sly
from supervisely.app.widgets import (
    Bokeh,
    Button,
    Container,
    Empty,
    Flexbox,
    GridGalleryV2,
    IFrame,
    Input,
    InputNumber,
    OneOf,
    RadioTabs,
    Select,
    SelectDataset,
    SelectItem,
    SelectProject,
)

dotenv.load_dotenv(os.path.expanduser("~/supervisely.env"))
dotenv.load_dotenv("local.env")


class TupleFields:
    """Fields of the named tuples used in the project."""

    ID = "id"
    DATASET_ID = "dataset_id"
    FULL_URL = "full_url"
    CAS_URL = "cas_url"
    HDF5_URL = "hdf5_url"
    UPDATED_AT = "updated_at"
    UNIT_SIZE = "unitSize"
    URL = "url"
    THUMBNAIL = "thumbnail"
    ATLAS_ID = "atlasId"
    ATLAS_INDEX = "atlasIndex"
    VECTOR = "vector"
    IMAGES = "images"


_ImageInfoLite = namedtuple(
    "_ImageInfoLite",
    [
        TupleFields.ID,
        TupleFields.DATASET_ID,
        TupleFields.FULL_URL,
        TupleFields.CAS_URL,
        TupleFields.UPDATED_AT,
    ],
)


class ImageInfoLite(_ImageInfoLite):
    def to_json(self):
        return {
            TupleFields.ID: self.id,
            TupleFields.DATASET_ID: self.dataset_id,
            TupleFields.FULL_URL: self.full_url,
            TupleFields.CAS_URL: self.cas_url,
            TupleFields.UPDATED_AT: self.updated_at,
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            id=data[TupleFields.ID],
            dataset_id=data[TupleFields.DATASET_ID],
            full_url=data[TupleFields.FULL_URL],
            cas_url=data[TupleFields.CAS_URL],
            updated_at=data[TupleFields.UPDATED_AT],
        )


api = sly.Api()


embeddings_generator_task_id = os.getenv("TASK_ID")
project_id = os.getenv("PROJECT_ID")

current_items = []

bokeh = Bokeh([])
bokeh_iframe = IFrame()
bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="100%")
select_project = SelectProject(compact=False)
input_search = Input(placeholder="Search query")
gallery = GridGalleryV2(5)
search_button = Button("Search")
load_project = Button("Load project")
tabs = RadioTabs(titles=["Plot", "Gallery"], contents=[bokeh_iframe, gallery])
clusters_button = Button("Clusters")
input_sample_size = InputNumber(value=20)
diverse_button = Button("Diverse")


def draw_all_projections(project_id):
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
            data={"project_id": project_id, "force": False},
        )
    except:
        pass

    print("=====================================")
    print("Sending request to get projections...")

    r = api.task.send_request(
        embeddings_generator_task_id, "projections", data={"project_id": project_id}
    )
    infos, projections = r
    print(f"Got {len(projections)} projections")

    print("=====================================")
    print("Init bokeh widget")
    bokeh.clear()
    plot = Bokeh.Circle(
        x_coordinates=[projection[1] for projection in projections],
        y_coordinates=[projection[0] for projection in projections],
        colors=["#222222"] * len(projections),
        legend_label="Images",
        plot_id=1,
        radii=[0.05] * len(projections),
    )
    bokeh.add_plots([plot])
    current_items = list(zip(infos, projections))
    bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="100%")
    bokeh_iframe.loading = False


def draw_projections_per_prompt(project_id, prompt, limit=20):
    gallery.loading = True
    bokeh_iframe.loading = True

    print("=====================================")
    print("Sending request to search...")
    r = api.task.send_request(
        embeddings_generator_task_id,
        "search",
        data={"project_id": project_id, "prompt": prompt, "limit": limit},
    )
    image_infos = r[0]
    print(f"Got {len(image_infos)} images")
    print("=====================================")

    print("Init bokeh widget")

    this_ids = set([info["id"] for info in image_infos])
    bokeh.clear()
    plot = Bokeh.Circle(
        x_coordinates=[item[1][1] for item in current_items if item[0]["id"] not in this_ids],
        y_coordinates=[item[1][0] for item in current_items if item[0]["id"] not in this_ids],
        colors=["#222222" for item in current_items if item[0]["id"] not in this_ids],
        legend_label="Images",
        plot_id=1,
        radii=[0.05 for item in current_items if item[0]["id"] not in this_ids],
    )
    new_plot = Bokeh.Circle(
        x_coordinates=[item[1][1] for item in current_items if item[0]["id"] in this_ids],
        y_coordinates=[item[1][0] for item in current_items if item[0]["id"] in this_ids],
        colors=["#006400" for item in current_items if item[0]["id"] in this_ids],
        legend_label=prompt,
        radii=[0.05 for item in current_items if item[0]["id"] in this_ids],
    )
    bokeh.add_plots([plot, new_plot])
    # bokeh._load_chart()
    bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="600px")
    bokeh_iframe.loading = False

    print("=====================================")
    print("init Gallery")
    gallery.clean_up()
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    for i, info in enumerate(image_infos, 1):
        ann_info = api.annotation.download(info["id"])
        gallery.append(info["full_url"], ann_info, project_meta, call_update=False)
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
    colors = ["#222222", *sly.color.get_predefined_colors(len(unique_labels) - 1)]
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
    # bokeh._load_chart()
    bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="600px")
    bokeh_iframe.loading = False
    gallery._update()
    gallery.loading = False


def draw_diverse(project_id, sample_size):
    gallery.loading = True
    bokeh_iframe.loading = True
    print("=====================================")
    print("Sending request to diverse...")
    r = api.task.send_request(
        embeddings_generator_task_id,
        "diverse",
        data={"project_id": project_id, "method": "random", "sample_size": sample_size},
    )
    image_infos = r
    print(f"Got {len(image_infos)} images")
    print("=====================================")

    print("Init bokeh widget")

    this_ids = set([info["id"] for info in image_infos])
    bokeh.clear()
    plot = Bokeh.Circle(
        x_coordinates=[item[1][1] for item in current_items if item[0]["id"] not in this_ids],
        y_coordinates=[item[1][0] for item in current_items if item[0]["id"] not in this_ids],
        colors=["#222222" for item in current_items if item[0]["id"] not in this_ids],
        legend_label="Images",
        plot_id=1,
        radii=[0.05 for item in current_items if item[0]["id"] not in this_ids],
    )
    new_plot = Bokeh.Circle(
        x_coordinates=[item[1][1] for item in current_items if item[0]["id"] in this_ids],
        y_coordinates=[item[1][0] for item in current_items if item[0]["id"] in this_ids],
        colors=["#006400" for item in current_items if item[0]["id"] in this_ids],
        legend_label="Sample",
        radii=[0.05 for item in current_items if item[0]["id"] in this_ids],
    )
    bokeh.add_plots([plot, new_plot])
    # bokeh._load_chart()
    bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="600px")
    bokeh_iframe.loading = False

    print("=====================================")
    print("init Gallery")
    gallery.clean_up()
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    for i, info in enumerate(image_infos, 1):
        ann_info = api.annotation.download(info["id"])
        gallery.append(info["full_url"], ann_info, project_meta, call_update=False)
        print(f"image {i}/{len(image_infos)} added to gallery")
    gallery._update()
    gallery.loading = False


@load_project.click
def load_project_click():
    project_id = select_project.get_selected_id()
    draw_all_projections(project_id)


@search_button.click
def search_click():

    prompt = input_search.get_value()
    project_id = select_project.get_selected_id()
    draw_projections_per_prompt(project_id, prompt)


@clusters_button.click
def clusters_click():
    project_id = select_project.get_selected_id()
    draw_clusters(project_id)


@diverse_button.click
def diverse_click():
    project_id = select_project.get_selected_id()
    sample_size = input_sample_size.get_value()
    draw_diverse(project_id, sample_size)


layout = Container(
    widgets=[
        select_project,
        load_project,
        input_search,
        search_button,
        clusters_button,
        input_sample_size,
        diverse_button,
        tabs,
    ]
)
app = sly.Application(layout=layout)
