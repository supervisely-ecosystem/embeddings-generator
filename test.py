import os
from collections import defaultdict, namedtuple

import dotenv
import supervisely as sly
from supervisely.app.widgets import (Bokeh, Button, Container, Empty, Flexbox,
                                     GridGalleryV2, IFrame, Input, OneOf,
                                     RadioTabs, Select, SelectDataset,
                                     SelectItem, SelectProject)

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
input_search = Input()
# select = Select(items=[Select.Item("all", "All", content=Empty()), Select.Item("search", "Search", content=input_search), Select.Item("image", "Image", content=Empty())])

gallery = GridGalleryV2(5)

# oneof = OneOf(select)
search_button = Button("Search")
load_project = Button("Load project")

tabs = RadioTabs(titles=["Plot", "Gallery"], contents=[bokeh_iframe, gallery])

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
        r = api.task.send_request(embeddings_generator_task_id, "embeddings", data={"project_id": project_id, "force": False})
    except:
        pass

    print("=====================================")
    print("Sending request to get projections...")

    r = api.task.send_request(embeddings_generator_task_id, "projections", data={"project_id": project_id})
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
    print("Embeddings Generator Task ID and Project ID")
    print("task_id:", embeddings_generator_task_id)
    print("project_id:", project_id)
    print("=====================================")
    print("Sending request to update_embeddings...")

    try:
        r = api.task.send_request(embeddings_generator_task_id, "embeddings", data={"project_id": project_id, "force": False})
    except:
        pass


    print ("=====================================")
    print("Sending request to search...")
    r = api.task.send_request(embeddings_generator_task_id, "search", data={"project_id": project_id, "prompt": prompt, "limit": limit})
    image_infos = [ImageInfoLite.from_json(info) for info in r]
    print(f"Got {len(image_infos)} images")

    print("=====================================")
    print("Sending request to get projections...")

    r = api.task.send_request(embeddings_generator_task_id, "projections", data={"project_id": project_id, "image_ids": [info.id for info in image_infos]})
    infos, projections = r
    print(f"Got {len(projections)} projections")

    print("=====================================")
    print("Init bokeh widget")
    

    plot: Bokeh.Circle = bokeh._plots[0]
    this_ids = set([info["id"] for info in infos])
    plot._x_coordinates = [item[1][1] for item in current_items if item[0]["id"] not in this_ids]
    plot._y_coordinates = [item[1][0] for item in current_items if item[0]["id"] not in this_ids]
    plot._radii = [0.05 for item in current_items if item[0]["id"] not in this_ids]
    plot._colors = ["#222222" for item in current_items if item[0]["id"] not in this_ids]

    new_plot = Bokeh.Circle(
        x_coordinates=[projection[1] for projection in projections],
        y_coordinates=[projection[0] for projection in projections],
        colors=["#006400"] * len(projections),
        legend_label=prompt,
        radii=[0.05] * len(projections),
    )
    if len(bokeh._plots) > 1:
        bokeh._plots[1] = new_plot
    else:
        bokeh.add_plots([new_plot])
    bokeh._load_chart()
    bokeh_iframe.set(bokeh.html_route_with_timestamp, height="650px", width="600px")
    bokeh_iframe.loading = False

    print("=====================================")
    print("init Gallery")
    gallery.clean_up()
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    for i, info in enumerate(infos, 1):
        ann_info = api.annotation.download(info["id"])
        gallery.append(info["full_url"], ann_info, project_meta, call_update=False)
        print(f"image {i}/{len(infos)} added to gallery")
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
    

layout = Container(widgets=[select_project, load_project ,input_search, search_button, tabs])
app = sly.Application(layout=layout)
