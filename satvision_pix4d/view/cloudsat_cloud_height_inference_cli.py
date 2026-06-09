"""Run cloud-height binary segmentation inference from a trained Lightning checkpoint."""

import argparse
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

from satvision_pix4d.dataloaders.cloudsat_inference_loader import (
    DEFAULT_CLOUDSAT_INFERENCE_DIR,
    build_cloudsat_inference_dataloader,
)

ALTITUDE_VALUES_KM = np.arange(0.0, 20.0, 0.5, dtype=np.float32)
FILENAME_RE = re.compile(
    r"^abi_chip_(?P<year>\d{4})(?P<doy>\d{3})_(?P<hhmm>\d{4})_r(?P<row>\d{5})_c(?P<col>\d{5})\.npz$"
)


class CustomUNET(nn.Module):
    """Custom U-Net architecture matching the cloud-height baseline notebook."""

    def __init__(
        self,
        in_channels: int = 16,
        decoder_classes: int = 64,
        num_classes: int = 1,
        head_dropout: float = 0.2,
        output_shape: Tuple[int, int] = (91, 40),
    ) -> None:
        super().__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]

        self.double_conv_downs = nn.ModuleList(
            [self._double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])]
        )
        self.up_trans = nn.ModuleList(
            [
                nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
                for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])
            ]
        )
        self.double_conv_ups = nn.ModuleList(
            [self._double_conv(layer, layer // 2) for layer in self.layers[::-1][:-2]]
        )
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, decoder_classes, kernel_size=1)
        self.prediction_head = nn.Sequential(
            nn.Conv2d(decoder_classes, num_classes, kernel_size=3, stride=1, padding=1),
            nn.Dropout(head_dropout),
            nn.Upsample(size=output_shape, mode="bilinear", align_corners=False),
        )

    @staticmethod
    def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concat_layers = []

        for down in self.double_conv_downs:
            x = down(x)
            if down is not self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)

        concat_layers = concat_layers[::-1]

        for up_trans, double_conv_up, concat_layer in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = F.interpolate(x, size=concat_layer.shape[2:], mode="bilinear", align_corners=False)
            x = double_conv_up(torch.cat((concat_layer, x), dim=1))

        x = self.final_conv(x)
        x = self.prediction_head(x)
        return x


def _strip_lightning_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            stripped[key[len("model."):]] = value
    return stripped


def load_model_from_lightning_ckpt(
    checkpoint_path: str,
    device: torch.device,
    in_channels: int,
    num_classes: int,
    target_height: int,
    target_width: int,
) -> CustomUNET:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = _strip_lightning_prefixes(state_dict)
    if not model_state:
        raise KeyError(
            "Could not find keys with 'model.' prefix in checkpoint state_dict. "
            "This script expects a Lightning checkpoint from CustomUNETLightning."
        )

    model = CustomUNET(
        in_channels=in_channels,
        num_classes=num_classes,
        decoder_classes=64,
        head_dropout=0.2,
        output_shape=(target_height, target_width),
    )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model


def _default_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_chip_filename(path: str) -> Dict[str, object]:
    name = os.path.basename(path)
    match = FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {name}")

    year = int(match.group("year"))
    doy = int(match.group("doy"))
    hhmm = match.group("hhmm")
    row = int(match.group("row"))
    col = int(match.group("col"))
    dt = datetime.strptime(f"{year}-{doy:03d} {hhmm}", "%Y-%j %H%M")
    utc_np = np.datetime64(dt.replace(second=0, microsecond=0), "m")

    return {
        "time": utc_np,
        "row": row,
        "col": col,
        "id": col,
        "row_col_id": f"r{row:05d}_c{col:05d}",
    }


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _build_output_dataset(records: List[Dict[str, object]], altitude_km: np.ndarray) -> xr.Dataset:
    if not records:
        raise ValueError("No inference records to write.")

    records = sorted(records, key=lambda r: (r["time"], r["id"], r["row"], r["path"]))

    times = np.array(sorted({r["time"] for r in records}), dtype="datetime64[m]")
    ids = np.array(sorted({int(r["id"]) for r in records}), dtype=np.int32)

    time_to_idx = {times[i]: i for i in range(len(times))}
    id_to_idx = {int(v): i for i, v in enumerate(ids.tolist())}

    n_time = len(times)
    n_id = len(ids)
    n_point = int(records[0]["lat_transect"].shape[0])
    n_alt = int(altitude_km.shape[0])

    cloud_mask = np.full((n_time, n_id, n_point, n_alt), np.nan, dtype=np.float32)
    lat_transect = np.full((n_time, n_id, n_point), np.nan, dtype=np.float32)
    lon_transect = np.full((n_time, n_id, n_point), np.nan, dtype=np.float32)
    center_row = np.full((n_time, n_id), -1, dtype=np.int32)
    center_col = np.full((n_time, n_id), -1, dtype=np.int32)
    center_lat = np.full((n_time, n_id), np.nan, dtype=np.float32)
    center_lon = np.full((n_time, n_id), np.nan, dtype=np.float32)
    row_col_id = np.full((n_time, n_id), "", dtype="<U13")

    seen_pairs = set()
    for r in records:
        lat_arr = np.asarray(r["lat_transect"], dtype=np.float32)
        lon_arr = np.asarray(r["lon_transect"], dtype=np.float32)
        pred_arr = np.asarray(r["prediction"], dtype=np.float32)

        if lat_arr.shape[0] != n_point or lon_arr.shape[0] != n_point:
            raise ValueError("Inconsistent transect length across chips.")
        if pred_arr.shape != (n_point, n_alt):
            raise ValueError(
                f"Prediction shape {pred_arr.shape} does not match expected {(n_point, n_alt)}. "
                "Check target-height/target-width and altitude definition."
            )

        key = (r["time"], int(r["id"]))
        if key in seen_pairs:
            raise ValueError(
                f"Duplicate (time, col-id) detected for time={r['time']} col={r['id']}. "
                "Using col as id requires col to be unique within each UTC time."
            )
        seen_pairs.add(key)

        ti = time_to_idx[r["time"]]
        ii = id_to_idx[int(r["id"])]

        cloud_mask[ti, ii, :, :] = pred_arr
        lat_transect[ti, ii, :] = lat_arr
        lon_transect[ti, ii, :] = lon_arr
        center_row[ti, ii] = int(r["row"])
        center_col[ti, ii] = int(r["col"])
        center_lat[ti, ii] = float(r["center_lat"])
        center_lon[ti, ii] = float(r["center_lon"])
        row_col_id[ti, ii] = str(r["row_col_id"])

    ds = xr.Dataset(
        data_vars={
            "cloud_mask": (
                ("time", "id", "transect_point", "altitude"),
                cloud_mask,
                {
                    "long_name": "Binary cloud mask prediction",
                    "description": "0/1 predictions from cloud-height inference model",
                },
            ),
            "center_row": (("time", "id"), center_row),
            "center_col": (("time", "id"), center_col),
            "center_lat": (("time", "id"), center_lat),
            "center_lon": (("time", "id"), center_lon),
            "lat_transect": (("time", "id", "transect_point"), lat_transect),
            "lon_transect": (("time", "id", "transect_point"), lon_transect),
            "row_col_id": (("time", "id"), row_col_id),
        },
        coords={
            "time": times,
            "id": ids,
            "transect_point": np.arange(n_point, dtype=np.int32),
            "altitude": altitude_km,
        },
        attrs={
            "title": "Cloud-height inference transects",
            "source": "satvision_pix4d cloud-height inference CLI",
            "id_definition": "Column index parsed from chip filename (_cXXXXX)",
            "centroid_metadata": "center_row, center_col, center_lat, center_lon stored as dataset variables indexed by (time, id)",
            "time_parsing": "Parsed from chip filename pattern abi_chip_YYYYDOY_HHMM_rRRRRR_cCCCCC.npz",
        },
    )
    ds["altitude"].attrs["units"] = "km"
    ds["altitude"].attrs["long_name"] = "Altitude above reference level"
    ds["lat_transect"].attrs["units"] = "degrees_north"
    ds["lon_transect"].attrs["units"] = "degrees_east"
    return ds


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud-height binary segmentation inference to a single NetCDF file")
    parser.add_argument("--checkpoint-path", required=True, help="Path to Lightning .ckpt file")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_CLOUDSAT_INFERENCE_DIR,
        help=f"Directory with inference .npz files (default: {DEFAULT_CLOUDSAT_INFERENCE_DIR})",
    )
    parser.add_argument("--output-path", required=True, help="Output NetCDF path (e.g., preds.nc)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--glob-pattern", default="*.npz")
    parser.add_argument("--strict-loader", action="store_true", help="Raise if a sample fails to load")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")

    parser.add_argument("--in-channels", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--target-height", type=int, default=91)
    parser.add_argument("--target-width", type=int, default=40)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.target_width != len(ALTITUDE_VALUES_KM):
        raise ValueError(
            f"target-width ({args.target_width}) must match altitude bins ({len(ALTITUDE_VALUES_KM)}) "
            "for altitude=np.arange(0,20,0.5)."
        )

    device = _default_device(args.device)
    loader = build_cloudsat_inference_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        glob_pattern=args.glob_pattern,
        strict=args.strict_loader,
    )
    model = load_model_from_lightning_ckpt(
        checkpoint_path=args.checkpoint_path,
        device=device,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        target_height=args.target_height,
        target_width=args.target_width,
    )

    print("Starting inference (sorted input order)")
    print(f"  checkpoint: {args.checkpoint_path}")
    print(f"  data_dir: {args.data_dir}")
    print(f"  output_path: {args.output_path}")
    print(f"  samples: {len(loader.dataset)}")
    print(f"  device: {device}")

    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            chips = batch["chip"].float().to(device)
            paths = list(batch["path"])
            lats = _to_numpy(batch["lat_transect"])
            lons = _to_numpy(batch["lon_transect"])
            center_lats = _to_numpy(batch["center_lat"])
            center_lons = _to_numpy(batch["center_lon"])
            coords = _to_numpy(batch["coords"]) if "coords" in batch else None

            logits = model(chips)
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > args.threshold).to(torch.uint8)
            preds_np = preds.cpu().numpy()

            for idx, path in enumerate(paths):
                meta = _parse_chip_filename(path)
                row = int(meta["row"])
                col = int(meta["col"])
                if coords is not None:
                    row = int(coords[idx][0])
                    col = int(coords[idx][1])
                    meta["row"] = row
                    meta["col"] = col
                    meta["id"] = col
                    meta["row_col_id"] = f"r{row:05d}_c{col:05d}"

                records.append(
                    {
                        "path": path,
                        "time": meta["time"],
                        "id": meta["id"],
                        "row": meta["row"],
                        "col": meta["col"],
                        "row_col_id": meta["row_col_id"],
                        "lat_transect": lats[idx],
                        "lon_transect": lons[idx],
                        "center_lat": float(center_lats[idx]),
                        "center_lon": float(center_lons[idx]),
                        "prediction": preds_np[idx],
                    }
                )

    ds = _build_output_dataset(records, ALTITUDE_VALUES_KM)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    ds.to_netcdf(args.output_path)
    print(f"Inference complete. Wrote NetCDF: {args.output_path}")
    print(f"Dataset dims: {dict(ds.sizes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
