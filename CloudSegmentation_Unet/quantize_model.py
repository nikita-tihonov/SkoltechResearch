import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp
from functools import partial

import re
import subprocess
from rich.progress import track
from sklearn.metrics import accuracy_score

import nncf
import openvino as ov


def pixel_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def validate(model: ov.Model, val_loader: torch.utils.data.DataLoader) -> float:

    accs = []

    compiled_model = ov.compile_model(model, device_name="CPU")
    output = compiled_model.outputs[0]

    for images, target in track(val_loader, description="Validating"):
        pred = compiled_model(images)[output]
        pred = np.argmax(pred, axis=1)
        acc = pixel_accuracy(pred, target)
        accs.append(acc)

    return sum(accs)/len(accs)


def run_benchmark(model_path: Path, shape: list[int]) -> float:
    command = [
        "benchmark_app",
        "-m", model_path.as_posix(),
        "-d", "CPU",
        "-api", "async",
        "-t", "15",
        "-shape", str(shape),
    ]  # fmt: skip
    cmd_output = subprocess.check_output(command, text=True)  # nosec
    print(*cmd_output.splitlines()[-8:], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", cmd_output)
    return float(match.group(1))


def get_model_size(ir_path: Path, m_type: str = "Mb") -> float:
    xml_size = ir_path.stat().st_size
    bin_size = ir_path.with_suffix(".bin").stat().st_size
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    print(f"Model graph (xml):   {xml_size:.3f} {m_type}")
    print(f"Model weights (bin): {bin_size:.3f} {m_type}")
    print(f"Model size:          {model_size:.3f} {m_type}")
    return model_size



class CalibrationDataset(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
        super().__init__()

        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()][:300]

        self.pytorch = pytorch

    def combine_files(self, r_file: Path, g_dir, b_dir,nir_dir, gt_dir):

        files = {'red': r_file,
                 'green':g_dir/r_file.name.replace('red', 'green'),
                 'blue': b_dir/r_file.name.replace('red', 'blue'),
                 'nir': nir_dir/r_file.name.replace('red', 'nir'),
                 'gt': gt_dir/r_file.name.replace('red', 'gt')}

        return files

    def __len__(self):

        return len(self.files)

    def open_as_array(self, idx, invert=False):

        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                            np.array(Image.open(self.files[idx]['nir'])),
                           ], axis=2)

        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))

        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    
    def open_mask(self, idx, add_dims=False):

        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)

        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):

        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)

        return x, y

    def __repr__(self):

        s = 'Dataset class with {} files'.format(self.__len__())

        return s
    

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(device)


base_path = Path('/root/.cache/kagglehub/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images/versions/4/38-Cloud_training')
dataset = CalibrationDataset(base_path/'train_red',
                    base_path/'train_green',
                    base_path/'train_blue',
                    base_path/'train_nir',
                    base_path/'train_gt'
                    )

batch_size = 32
calibration_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


torch_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=2
).to(device)

torch_model.load_state_dict(torch.load("weights/unet_cloud_25epochs.pth", map_location=device))
torch_model.to(device)
torch_model.eval()
print("Модель загружена и готова к инференсу")


def transform_fn(data_item: tuple[torch.Tensor, int], device: torch.device) -> torch.Tensor:
    images, _ = data_item
    return images.to(device)


subset_size = 300 // batch_size
calibration_dataset = nncf.Dataset(calibration_loader, partial(transform_fn, device=device))
torch_quantized_model = nncf.quantize(
    torch_model, 
    calibration_dataset, 
    subset_size=subset_size,
    preset=nncf.QuantizationPreset.MIXED,  # или nncf.QuantizationPreset.MIXED
    # fast_bias_correction=False,
    # target_device=nncf.TargetDevice.CPU,  # или nncf.TargetDevice.VPU, GPU
    )


dummy_input = torch.randn(1, 4, 384, 384)
ov_model = ov.convert_model(torch_model.cpu(), example_input=dummy_input)
ov_quantized_model = ov.convert_model(torch_quantized_model.cpu(), example_input=dummy_input)

fp32_ir_path =  Path("ov_weights/unet_cloud_fp32_mixed.xml")
ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
print(f"[1/7] Save FP32 model: {fp32_ir_path}")
fp32_model_size = get_model_size(fp32_ir_path)

int8_ir_path = Path("ov_weights/unet_cloud_int8_mixed.xml")
ov.save_model(ov_quantized_model, int8_ir_path)
print(f"[2/7] Save INT8 model: {int8_ir_path}")
int8_model_size = get_model_size(int8_ir_path)

print("[3/7] Benchmark FP32 model:")
fp32_fps = run_benchmark(fp32_ir_path, shape=[1, 4, 384, 384])
print("[4/7] Benchmark INT8 model:")
int8_fps = run_benchmark(int8_ir_path, shape=[1, 4, 384, 384])

print("[5/7] Validate OpenVINO FP32 model:")
fp32_top1 = validate(ov_model, calibration_loader)
print(f"Accuracy: {fp32_top1:.3f}")

print("[6/7] Validate OpenVINO INT8 model:")
int8_top1 = validate(ov_quantized_model, calibration_loader)
print(f"Accuracy: {int8_top1:.3f}")

print("[7/7] Report:")
print(f"Accuracy drop: {fp32_top1 - int8_top1:.3f}")
print(f"Model compression rate: {fp32_model_size / int8_model_size:.3f}")
print(f"Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")