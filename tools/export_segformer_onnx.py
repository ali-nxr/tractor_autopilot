"""
tools/export_segformer_onnx.py — one-time export of a Hugging Face SegFormer
checkpoint to the ONNX model segformer_onnx.py runs with onnxruntime.

Only this script needs torch + transformers; the running app needs
onnxruntime alone.

Run (from the project root):
    python tools/export_segformer_onnx.py
    python tools/export_segformer_onnx.py --model nvidia/segformer-b2-finetuned-ade-512-512 \
        --out models/segformer-b2-ade-512x512.onnx

What is baked INTO the graph, and why:
  - Preprocessing. Input is the camera's own uint8 BGR frame (NHWC), already
    resized to the model's input size. BGR->RGB, /255, ImageNet mean/std and
    NHWC->NCHW all run inside the graph (on the GPU when one is used) — the
    only per-frame CPU work left is one cv2.resize.
  - Softmax + class reduction. A second input, `class_weights` (num_labels,),
    selects which classes count as "driveable" (1.0 = driveable, 0.0 = not).
    The graph returns sum(softmax * weights) — one (1, H/4, W/4) probability
    map instead of the full 150-channel logits, so the GPU->CPU copy is tiny.
    Because the class choice is an INPUT, editing the class list in
    config_segformer.py takes effect without re-exporting.
  - Argmax class map. A second output, `class_ids` (1, H/4, W/4) int64,
    holds each pixel's most likely ADE20K class. Only the debug view
    (config.DEBUG) fetches it — onnxruntime skips nothing else either way.

The label names (id2label), input size and source checkpoint are stored in
the ONNX file's own metadata, so the model is a single self-describing file
— no sidecar that can go missing or drift out of sync.
"""

import argparse
import json
import os
import sys

import torch
from transformers import SegformerForSemanticSegmentation

# ImageNet statistics — what every nvidia/segformer-* checkpoint was trained
# with (see the checkpoint's preprocessor_config.json).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _DriveableProbabilityModel(torch.nn.Module):
    def __init__(self, segformer):
        super().__init__()
        self.segformer = segformer
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, image_bgr_u8, class_weights):
        x = image_bgr_u8.permute(0, 3, 1, 2).float()   # NHWC uint8 -> NCHW float
        x = x.flip(1)                                    # BGR -> RGB
        x = (x - self.mean) / self.std
        logits = self.segformer(pixel_values=x).logits   # (1, C, H/4, W/4)
        probs = torch.softmax(logits, dim=1)
        driveable = torch.einsum("nchw,c->nhw", probs, class_weights)
        return driveable, torch.argmax(logits, dim=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512",
                        help="Hugging Face checkpoint id or local folder")
    parser.add_argument("--out", default=os.path.join("models", "segformer-b0-ade-512x512.onnx"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    if args.width % 32 or args.height % 32:
        parser.error("--width and --height must be multiples of 32 (SegFormer's total stride)")

    print(f"[export] loading {args.model}")
    segformer = SegformerForSemanticSegmentation.from_pretrained(args.model).eval()
    id2label = {int(k): v.strip() for k, v in segformer.config.id2label.items()}
    num_labels = len(id2label)

    model = _DriveableProbabilityModel(segformer).eval()
    dummy_image = torch.zeros((1, args.height, args.width, 3), dtype=torch.uint8)
    dummy_weights = torch.zeros((num_labels,), dtype=torch.float32)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[export] exporting {args.width}x{args.height}, opset {args.opset} -> {args.out}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_image, dummy_weights),
            args.out,
            input_names=["image_bgr", "class_weights"],
            output_names=["driveable_prob", "class_ids"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,   # the TorchScript exporter: fixed shapes, widest EP support (incl. TensorRT)
        )

    import onnx
    onnx_model = onnx.load(args.out)
    metadata = {
        "source_model": args.model,
        "input_width": str(args.width),
        "input_height": str(args.height),
        "id2label": json.dumps(id2label),
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, args.out)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[export] done: {args.out} ({size_mb:.1f} MB, {num_labels} classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
