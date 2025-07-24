import os
import datetime
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from dateutil import parser  # 用于解析 ISO8601 时间
import pandas as pd
from PIL import Image

# ==============================================
# 0) 读取 zero-shot 格式数据集的辅助函数
# ==============================================
def load_zero_shot_dataset(dataset_dir: str):
    csv_path = os.path.join(dataset_dir, "data.csv")
    imgs_dir = os.path.join(dataset_dir, "imgs")
    df = pd.read_csv(csv_path)
    if not df["time"].is_monotonic_increasing:
        df = df.sort_values("time").reset_index(drop=True)
    img_files = sorted(os.listdir(imgs_dir))
    if len(img_files) != len(df):
        raise ValueError("CSV 文件的行数与 imgs 目录中的图片数量不一致。")
    dataset = []
    for idx, row in df.iterrows():
        img_path = os.path.join(imgs_dir, img_files[idx])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"无法打开图片 {img_path}: {e}")
        sample = {
            "time": row["time"],
            "pv": row["pv"],
            "image": img
        }
        dataset.append(sample)
    return dataset

# ==============================================
# 1) 自定义 Dataset: SkyImageSeqForecastDataset
# ==============================================
class SkyImageSeqForecastDataset(Dataset):
    def __init__(self, hf_dataset, input_size=5, horizons=1, transform=None, time_gap_threshold: float = None, cap=30.0):
        self.hf_dataset = hf_dataset
        self.input_size = input_size
        self.horizons = horizons
        self.transform = transform
        self.time_gap_threshold = time_gap_threshold
        self.cap = cap
        total_samples = len(self.hf_dataset) - (input_size + horizons) + 1
        if total_samples < 1:
            raise ValueError(
                f"Not enough data: dataset_len={len(self.hf_dataset)}, "
                f"input_size={input_size}, horizons={horizons}"
            )
        if self.time_gap_threshold is not None:
            self.valid_indices = []
            for idx in range(total_samples):
                valid = True
                prev_time = self._get_time(self.hf_dataset[idx])
                for j in range(idx + 1, idx + input_size + horizons):
                    curr_time = self._get_time(self.hf_dataset[j])
                    if curr_time - prev_time > self.time_gap_threshold:
                        valid = False
                        break
                    prev_time = curr_time
                if valid:
                    self.valid_indices.append(idx)
            if not self.valid_indices:
                raise ValueError("No valid samples found under the given time_gap_threshold")
            self.num_samples = len(self.valid_indices)
        else:
            self.valid_indices = list(range(total_samples))
            self.num_samples = total_samples

    def _get_time(self, sample):
        time_info = sample["time"]
        if isinstance(time_info, datetime.datetime):
            return time_info.timestamp()
        elif isinstance(time_info, (int, float)):
            return float(time_info)
        elif isinstance(time_info, list):
            return float(time_info[0])
        else:
            try:
                dt = parser.isoparse(time_info)
                return dt.timestamp()
            except Exception:
                return 0.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        in_start = valid_idx
        in_end = valid_idx + self.input_size
        out_start = in_end
        out_end = in_end + self.horizons

        images_in, times_in, pvs_in = [], [], []
        images_out, times_out, pvs_out = [], [], []

        for i in range(in_start, in_end):
            sample = self.hf_dataset[i]
            img, t_info, pv_info = self._process_sample(sample)
            images_in.append(img)
            times_in.append(t_info)
            pvs_in.append(pv_info)

        for i in range(out_start, out_end):
            sample = self.hf_dataset[i]
            img, t_info, pv_info = self._process_sample(sample)
            images_out.append(img)
            times_out.append(t_info)
            pvs_out.append(pv_info)

        images_in = torch.stack(images_in, dim=0)
        times_in  = torch.stack(times_in, dim=0)
        pvs_in    = torch.stack(pvs_in, dim=0)

        images_out = torch.stack(images_out, dim=0)
        times_out  = torch.stack(times_out, dim=0)
        pvs_out    = torch.stack(pvs_out, dim=0)

        return (images_in, times_in, pvs_in), (images_out, times_out, pvs_out)

    def _process_sample(self, sample):
        img = sample["image"]
        time_info = sample["time"]
        pv_info = sample["pv"]

        if self.transform:
            img = self.transform(img)
        else:
            raise ValueError("No transform provided; need to convert PIL to tensor manually")

        if isinstance(time_info, datetime.datetime):
            time_info = time_info.timestamp()
        if isinstance(time_info, (int, float)):
            time_info = torch.tensor([time_info], dtype=torch.float)
        elif isinstance(time_info, list):
            time_info = torch.tensor(time_info, dtype=torch.float)
        else:
            try:
                dt = parser.isoparse(time_info)
                time_info = torch.tensor([dt.timestamp()], dtype=torch.float)
            except Exception:
                time_info = torch.tensor([0.0], dtype=torch.float)

        if isinstance(pv_info, (int, float)):
            pv_info = torch.tensor([pv_info / self.cap], dtype=torch.float)
        elif isinstance(pv_info, list):
            pv_info = torch.tensor(pv_info, dtype=torch.float) / self.cap
        else:
            pv_info = torch.tensor([0.0], dtype=torch.float)

        return img, time_info, pv_info

# ==============================================
# 2) 数据加载器函数: load_dataset_dataloaders
# ==============================================
def load_dataset_dataloaders(
    batch_size=32,
    num_workers=0,
    dataset_dir="skyimagenet/SKIPPD",  # 当 dataset_mode=="hf" 时，传 Hugging Face 数据集名称
    input_size=5,
    horizons=1,
    sampling_interval=15,
    normalize_image=True,  # 是否对图像进行标准化
    dataset_mode="hf"      # "hf" 或 "zero-shot"
):
    # 用于筛选 2019 年数据的函数
    def filter_2019(example):
        try:
            # 如果已经有 timestamp 字段则直接用，否则解析 time 字段
            if "timestamp" in example:
                ts = example["timestamp"]
            else:
                if isinstance(example["time"], datetime.datetime):
                    ts = example["time"].timestamp()
                else:
                    dt = parser.isoparse(example["time"])
                    ts = dt.timestamp()
            return datetime.datetime.fromtimestamp(ts).year == 2019
        except Exception:
            return False

    if dataset_mode == "zero-shot":
        full_zero_dataset = load_zero_shot_dataset(dataset_dir)
        # 筛选出 2019 年的数据
        full_zero_dataset = [ex for ex in full_zero_dataset if filter_2019(ex)]
        print(f"Zero-shot 2019 数据集样本数: {len(full_zero_dataset)}")
        full_zero_dataset = full_zero_dataset[::sampling_interval]
        print(f"采样后样本数: {len(full_zero_dataset)}")
    else:
        raw_dataset = load_dataset(dataset_dir)
        print(f"原始数据集 splits: {list(raw_dataset.keys())}")
        original_train_count = len(raw_dataset['train'])
        original_test_count = len(raw_dataset['test'])
        print(f"原始 train 数量: {original_train_count}")
        print(f"原始 test 数量: {original_test_count}")
        full_dataset = concatenate_datasets([raw_dataset['train'], raw_dataset['test']])
        # 解析时间戳
        def parse_time_to_timestamp(example):
            if isinstance(example["time"], datetime.datetime):
                example["timestamp"] = example["time"].timestamp()
            else:
                dt = parser.isoparse(example["time"])
                example["timestamp"] = dt.timestamp()
            return example
        full_dataset = full_dataset.map(parse_time_to_timestamp)
        # 筛选出 2019 年数据
        full_dataset = full_dataset.filter(filter_2019)
        print("筛选出 2019 年数据完成。")
        full_dataset = full_dataset.sort("timestamp")
        print("数据按照 time 排序完成。")
        # 按照 7:1:2 拆分
        split_1 = full_dataset.train_test_split(test_size=0.2, shuffle=False)
        train_val_dataset = split_1['train']
        test_dataset = split_1['test']
        print(f"重新划分后 test 数量: {len(test_dataset)}")
        split_2 = train_val_dataset.train_test_split(test_size=0.125, shuffle=False)
        train_dataset = split_2['train']
        val_dataset = split_2['test']
        print(f"重新划分后 train 数量: {len(train_dataset)}")
        print(f"重新划分后 validation 数量: {len(val_dataset)}")
        train_dataset = train_dataset.select(range(0, len(train_dataset), sampling_interval))
        val_dataset   = val_dataset.select(range(0, len(val_dataset), sampling_interval))
        test_dataset  = test_dataset.select(range(0, len(test_dataset), sampling_interval))
        print("采样后各数据集样本数量:")
        print("  Train:", len(train_dataset))
        print("  Validation:", len(val_dataset))
        print("  Test:", len(test_dataset))
        full_dataset = {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }
    
    if normalize_image:
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        transform_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    if dataset_mode == "zero-shot":
        if 'UOW' in dataset_dir:
            cap = 31000.0
        elif 'SBRC' in dataset_dir:
            cap = 6300.0
        else:
            cap = 10000.0
        train_ds = SkyImageSeqForecastDataset(full_zero_dataset, input_size=input_size, horizons=horizons, transform=transform_pipeline, cap=cap)
        val_ds   = SkyImageSeqForecastDataset(full_zero_dataset, input_size=input_size, horizons=horizons, transform=transform_pipeline, cap=cap)
        test_ds  = SkyImageSeqForecastDataset(full_zero_dataset, input_size=input_size, horizons=horizons, transform=transform_pipeline, cap=cap)
    else:
        train_ds = SkyImageSeqForecastDataset(full_dataset["train"], input_size=input_size, horizons=horizons, transform=transform_pipeline)
        val_ds   = SkyImageSeqForecastDataset(full_dataset["validation"], input_size=input_size, horizons=horizons, transform=transform_pipeline)
        test_ds  = SkyImageSeqForecastDataset(full_dataset["test"], input_size=input_size, horizons=horizons, transform=transform_pipeline)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 定义统一使用的参数
    batch_size = 2
    sampling_interval = 2
    normalize_image = False
    dataset_dir = "../zero-shot/UOW"  # 或者指定 Hugging Face 数据集名称
    dataset_mode = "hf"         # "hf" 或 "zero-shot"
    
    # 指定不同的 horizon，最后只保存最后一个的结果
    horizons_list = [10, 20, 30]
    # 取最后一个 horizon 值
    horizons = horizons_list[-1]
    input_size = 2 * horizons  # input_size 是 horizon 的两倍
    print(f"\n处理组合: input_size={input_size}, horizons={horizons}")
    
    # 这里只需要测试集，因此取返回的第三个 DataLoader
    _, _, test_loader = load_dataset_dataloaders(
        batch_size=batch_size,
        input_size=input_size,
        horizons=horizons,
        sampling_interval=sampling_interval,
        normalize_image=normalize_image,
    )
    
    # 从 test_loader.dataset 中提取每个样本的时间数据
    # 每个 __getitem__ 返回 ((imgs_in, times_in, pvs_in), (imgs_out, times_out, pvs_out))
    data_records = []


    time_list = []
    targets_list = []
    with torch.no_grad():
        for batch_data in test_loader:
            (imgs_in, times_in, pvs_in), (imgs_out, times_out, pvs_out) = batch_data
            times = times_out.squeeze(-1)               # (B, horizons)
            targets = pvs_out.squeeze(-1)             # (B, horizons)s)
            time_list.append(times)
            targets_list.append(targets)
    targets_all = torch.cat(targets_list, dim=0).float().numpy() # shape: (num_windows, horizons)
    time_all = torch.cat(time_list, dim=0).float().numpy() # shape: (num_windows, horizons)
    # 构造 DataFrame 并保存为 CSV 文件
    time_df = time_all[:, -1]
    # save name with horizons
    time_df = pd.DataFrame(time_df)
    file_name = f"test_time_hor{horizons}.csv"
    time_df.to_csv(file_name, index=False)
    print(f"保存时间数据至 {file_name}")