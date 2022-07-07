import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import time
import segm.utils.torch as ptu

from segm.data.utils import STATS
from segm.data.biqdataset import BIQUANTS_DATASET_CATS_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb

from segm.model.factory import load_model
from segm.model.utils import inference


@click.command()
@click.option("--model-path", type=str)
@click.option("--input-dir", "-i", type=str, help="folder with input images")
@click.option("--output-dir", "-o", type=str, help="folder with output images")
@click.option("--gpu/--cpu", default=True, is_flag=True)
def main(model_path, input_dir, output_dir, gpu):
    ptu.set_gpu_mode(gpu)

    model_dir = Path(model_path).parent
    model, variant = load_model(model_path)
    normalization_name = variant["dataset_kwargs"]["normalization"]
    normalization = STATS[normalization_name]
    model = model.to(ptu.device)
    model.eval()

    cat_names, cat_colors = dataset_cat_description(BIQUANTS_DATASET_CATS_PATH)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    list_dir = list(input_dir.iterdir())
    for filename in tqdm(list_dir, ncols=80):
        pil_im = image = Image.open(filename).convert('RGB')
        h, w = pil_im.size
        pil_im = pil_im.resize((512, 512)).copy()
        image = image.resize((h // 3, w // 3)).copy()
        im = F.pil_to_tensor(pil_im).float() / 255
        im = F.normalize(im, normalization["mean"], normalization["std"])
        im = im.to(ptu.device).unsqueeze(0)

        im_meta = dict(flip=False)
        logits = inference(
            model,
            [im],
            [im_meta],
            ori_shape=im.shape[2:4],
            window_size=variant["inference_kwargs"]["window_size"],
            window_stride=variant["inference_kwargs"]["window_stride"],
            batch_size=1,
        )
        seg_map = logits.argmax(0, keepdim=True).unsqueeze(0)
        seg_map = torch.nn.functional.interpolate(seg_map.float(), size=(w // 3, h // 3), mode='nearest')[0, 0]

        visualizer = Visualizer(np.array(image), cat_names, cat_colors)
        vis_output = visualizer.draw_sem_seg(seg_map.to('cpu'))
        vis_output.save(output_dir / filename.name)

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05


class Visualizer:
    def __init__(self, img_rgb, our_classes, our_color, scale=1, instance_mode=0):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.our_classes = our_classes
        self.our_color = our_color
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self._instance_mode = instance_mode

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.5):
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]

        for label in filter(lambda l: l < len(self.our_classes), labels):
            try:
                mask_color = [x / 255 for x in self.our_color[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = self.our_classes[int(label)]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output

    def draw_binary_mask(
            self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5, area_threshold=0
    ):
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if not mask.has_holes:
            # draw polygons for regular masks
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
        else:
            #  Use Path/PathPatch to draw vector graphics:
            # https://stackoverflow.com/questions/8919719/how-to-plot-a-complex-polygon
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            has_valid_segment = True
            self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))

        if text is not None and has_valid_segment:
            # sometimes drawn on wrong objects. the heuristics here can improve.
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
            largest_component_id = np.argmax(stats[1:, -1]) + 1

            # draw text on the largest component, as well as other very large components.
            for cid in range(1, _num_cc):
                if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                    # median is more stable than centroid
                    # center = centroids[largest_component_id]
                    center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                    self.draw_text(text, center, color=lighter_color)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    def _change_color_brightness(self, color, brightness_factor):
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def draw_text(
            self,
            text,
            position,
            *,
            font_size=None,
            color="g",
            horizontal_alignment="center",
            rotation=0
    ):
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output


class GenericMask:

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (height, width), m.shape
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()


class VisImage:
    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


if __name__ == "__main__":
    main()

