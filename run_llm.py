import os
# ------------------------- GPU 自动选择 -------------------------
def select_device():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU')
    memory_list = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU').read().split('\n')
    memory_list = [int(x.split()[2]) for x in memory_list if 'Used' in x]
    device = memory_list.index(min(memory_list))
    print(f"Using GPU {device}")
    return device

os.environ["CUDA_VISIBLE_DEVICES"] = str(select_device())

import datetime
import argparse
import json
import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from peft import LoraModel, LoraConfig, get_peft_model       # 若未使用 LoRA，可移除
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs

from dataloader.mydataloader import load_dataset_dataloaders
from models.model import PV_VLM

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd

# ------------------------- 工具函数 -------------------------
def save_predictions_to_csv(preds, targets, file_prefix, log_dir):
    """
    将预测值与真实值展平成一列保存为 CSV。
    """
    flat_preds = preds.flatten()
    flat_targets = targets.flatten()
    df = pd.DataFrame({'pred': flat_preds, 'target': flat_targets})
    file_path = os.path.join(log_dir, f"{file_prefix}_predictions_flat.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {file_prefix} flat predictions and targets to {file_path}")

def train_step(model, optimizer, loss_fn, inputs, targets, accelerator):
    optimizer.zero_grad()
    outputs = model(*inputs)                  # (B, horizons)
    targets = targets.to(outputs.dtype)
    loss = loss_fn(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    return loss.item()

def train_epoch(model, optimizer, loss_fn, train_loader, accelerator, global_step, writer, log_interval=100):
    model.train()
    epoch_loss_sum = 0.0
    epoch_steps = 0
    pbar = tqdm(train_loader, desc="Training", unit="batch")
    for batch_data in pbar:
        (imgs_in, times_in, pvs_in), (imgs_out, times_out, pvs_out) = batch_data
        imgs_in = imgs_in.to(accelerator.device)
        pvs_in = pvs_in.to(accelerator.device)
        pvs_out = pvs_out.to(accelerator.device)   # (B, horizons, 1)
        targets = pvs_out.squeeze(-1)              # (B, horizons)

        loss_val = train_step(model, optimizer, loss_fn, (imgs_in, pvs_in), targets, accelerator)
        epoch_loss_sum += loss_val
        epoch_steps += 1
        global_step += 1

        running_avg = epoch_loss_sum / epoch_steps
        pbar.set_postfix({"avg_loss": f"{running_avg:.4f}"})
        if global_step % log_interval == 0:
            writer.add_scalar("Loss/train_step", loss_val, global_step)
    epoch_avg_loss = epoch_loss_sum / (epoch_steps if epoch_steps > 0 else 1)
    return epoch_avg_loss, global_step

def decimate_predictions(preds, targets, n_points):
    """
    滑窗预测去重：取每个窗口的最后一个预测点，得到不重叠序列，再截取最后 n_points。
    """
    unique_preds = preds[:, -1]
    unique_targets = targets[:, -1]
    total = len(unique_preds)
    if total <= n_points:
        return unique_preds, unique_targets, unique_preds, unique_targets
    start = total - n_points
    return unique_preds[start:], unique_targets[start:], unique_preds, unique_targets

def plot_decimated_predictions(decimated_preds, decimated_targets):
    x = np.arange(len(decimated_preds))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, decimated_targets, marker='o', label='Actual')
    ax.plot(x, decimated_preds, marker='x', label='Predicted')
    ax.set_xlabel("Time Index")
    ax.set_ylabel("PV Output")
    ax.set_title(f"Decimated Predictions vs Actual (Last {len(decimated_preds)} Points)")
    ax.legend()
    return fig

def validate_epoch_full(model, loss_fn, loader, accelerator):
    model.eval()
    loss_sum = 0.0
    steps = 0
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for batch_data in loader:
            (imgs_in, times_in, pvs_in), (imgs_out, times_out, pvs_out) = batch_data
            imgs_in = imgs_in.to(accelerator.device)
            pvs_in = pvs_in.to(accelerator.device)
            pvs_out = pvs_out.to(accelerator.device)  # (B, horizons, 1)
            targets = pvs_out.squeeze(-1)             # (B, horizons)
            outputs = model(imgs_in, pvs_in)          # (B, horizons)
            targets = targets.to(outputs.dtype)
            loss_val = loss_fn(outputs, targets)
            loss_sum += loss_val.item()
            steps += 1
            preds_list.append(outputs.cpu())
            targets_list.append(targets.cpu())
    if steps == 0:
        print("Debug: No data in loader!")
        return 0, None, None
    avg_loss = loss_sum / steps
    preds_all = torch.cat(preds_list, dim=0).float().numpy()
    targets_all = torch.cat(targets_list, dim=0).float().numpy()
    model.train()
    return avg_loss, preds_all, targets_all

def tiny_overfit_test(model, loss_fn, optimizer, accelerator, train_loader, epochs=50):
    """
    极小数据集过拟合测试（可选调试用）。
    """
    print("\n=== Starting tiny dataset overfit test ===")
    tiny_data = []
    for i, batch_data in enumerate(train_loader):
        tiny_data.append(batch_data)
        if i >= 1:
            break

    tiny_loader = [(imgs_in, times_in, pvs_in, imgs_out, times_out, pvs_out)
                   for (imgs_in, times_in, pvs_in), (imgs_out, times_out, pvs_out) in tiny_data]

    total_samples = sum([imgs_in.size(0) for (imgs_in, _, _, _, _, _) in tiny_loader])
    print(f"tiny_loader total samples = {total_samples}")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for (imgs_in, times_in, pvs_in, imgs_out, times_out, pvs_out) in tiny_loader:
            imgs_in = imgs_in.to(accelerator.device)
            pvs_in = pvs_in.to(accelerator.device)
            pvs_out = pvs_out.to(accelerator.device)
            targets = pvs_out.squeeze(-1)
            loss_val = train_step(model, optimizer, loss_fn, (imgs_in, pvs_in), targets, accelerator)
            epoch_loss_sum += loss_val
            epoch_steps += 1
        epoch_avg_loss = epoch_loss_sum / (epoch_steps if epoch_steps > 0 else 1)
        print(f"  Tiny Overfit Epoch {epoch}/{epochs} - loss = {epoch_avg_loss:.6f}")
    print("=== Finished tiny dataset overfit test ===\n")

# ------------------------- 主程序 -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train forecast model with early stopping, single final test and config logging'
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--input_size', type=int, default=20, help='frames in input sequence')
    parser.add_argument('--horizons', type=int, default=10, help='future frames')
    parser.add_argument('--log_dir', type=str, default='./runs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--vision_tower', type=str, default='siglip2-base-patch16-224')
    parser.add_argument('--vision_tower_ckpt', type=str, default='google/siglip2-base-patch16-224')
    parser.add_argument('--model_name', type=str, default='GPT2')
    parser.add_argument('--patch_len', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--num_hidden_layers', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_soft_prompt_text_tokens', type=int, default=20)
    parser.add_argument('--description', type=str, default='PV power alwasy positive')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.0)

    # StepLR
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)

    # plotting
    parser.add_argument('--n_points', type=int, default=int(96*1.5))

    # Ablation
    parser.add_argument('--wio_ts', type=bool, default=False)
    parser.add_argument('--wio_img', type=bool, default=False)
    parser.add_argument('--wio_prompt', type=bool, default=False)
    parser.add_argument('--only_ts', type=bool, default=False)

    # zero-shot
    parser.add_argument('--zs', action='store_true')
    parser.add_argument('--zs_dataset', type=str, default='')

    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    device = accelerator.device

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(args.log_dir, f"{args.vision_tower}_in{args.input_size}_hor{args.horizons}_{timestamp}")
    writer = SummaryWriter(log_dir=run_log_dir)

    config_text = json.dumps(vars(args), indent=4)
    writer.add_text("Config", config_text, global_step=0)

    torch.manual_seed(args.seed)
    loss_fn = torch.nn.MSELoss()

    # DataLoaders
    train_loader, val_loader, test_loader = load_dataset_dataloaders(
        batch_size=args.batch_size,
        input_size=args.input_size,
        horizons=args.horizons,
        num_workers=args.num_workers,
        dataset_dir="skyimagenet/SKIPPD"
    )
    if args.zs:
        print(f"Zero-shot mode. Dataset: {args.zs_dataset}")
        _, _, zs_loader = load_dataset_dataloaders(
            batch_size=args.batch_size,
            input_size=args.input_size,
            horizons=args.horizons,
            num_workers=args.num_workers,
            dataset_dir=args.zs_dataset,
            dataset_mode='zero-shot'
        )

    model = PV_VLM(configs=args)
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trained_parameters, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Accelerator prepare
    objs = [train_loader, val_loader, test_loader, model, optimizer, scheduler]
    if args.zs:
        objs.insert(2, zs_loader)  # 插入在 test_loader 后
    prepared = accelerator.prepare(*objs)
    # 解析返回
    if args.zs:
        train_loader, val_loader, test_loader, zs_loader, model, optimizer, scheduler = prepared
    else:
        train_loader, val_loader, test_loader, model, optimizer, scheduler = prepared

    global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_state = None

    # tiny_overfit_test(model, loss_fn, optimizer, accelerator, train_loader, epochs=300)

    # ----------------------- 训练循环（仅验证，最终再测试） -----------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        train_loss, global_step = train_epoch(model, optimizer, loss_fn, train_loader, accelerator, global_step, writer, log_interval=100)
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)

        val_loss, val_preds, val_targets = validate_epoch_full(model, loss_fn, val_loader, accelerator)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        print(f"Epoch {epoch} Validation Average Loss: {val_loss:.4f}")

        if best_val_loss - val_loss > args.min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print(f"Validation loss improved to {val_loss:.4f}.")
            best_state = copy.deepcopy(accelerator.unwrap_model(model).state_dict())

            # 可视化验证预测（可选）
            if val_preds is not None and val_targets is not None:
                dec_val_preds, dec_val_targets, _, _ = decimate_predictions(val_preds, val_targets, args.n_points)
                fig_val = plot_decimated_predictions(dec_val_preds, dec_val_targets)
                writer.add_figure("Validation_Predictions", fig_val, epoch)
                plt.close(fig_val)
        else:
            epochs_without_improvement += 1
            print(f"No significant improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered!")
                break

        scheduler.step()
        print("Current learning rate:", optimizer.param_groups[0]['lr'])

    # ----------------------- 保存 last/best -----------------------
    accelerator.wait_for_everyone()
    last_path = os.path.join(run_log_dir, f"{args.model_name}_last.pt")
    accelerator.save(accelerator.unwrap_model(model).state_dict(), last_path)
    print(f"Saved LAST model to {last_path}")

    if best_state is not None:
        best_path = os.path.join(run_log_dir, f"{args.model_name}_best.pt")
        accelerator.save(best_state, best_path)
        print(f"Saved BEST model to {best_path}")

    # ----------------------- 训练结束后统一测试 -----------------------
    if not args.zs:
        test_loss, preds, targets = validate_epoch_full(model, loss_fn, test_loader, accelerator)
        print(f"[FINAL] Test Average Loss: {test_loss:.4f}")
        if preds is not None and targets is not None:
            save_predictions_to_csv(preds, targets, "test", run_log_dir)
            dec_preds, dec_targets, all_preds, all_targets = decimate_predictions(preds, targets, args.n_points)
            fig_test = plot_decimated_predictions(dec_preds, dec_targets)
            writer.add_figure("Test_Predictions_Final", fig_test, global_step)
            plt.close(fig_test)
            save_predictions_to_csv(all_preds, all_targets, "test_draw", run_log_dir)
    else:
        zs_loss, zs_preds, zs_targets = validate_epoch_full(model, loss_fn, zs_loader, accelerator)
        print(f"[FINAL] Zero-shot Test Average Loss: {zs_loss:.4f}")
        if zs_preds is not None and zs_targets is not None:
            save_predictions_to_csv(zs_preds, zs_targets, "zs_test", run_log_dir)
            dec_zs_preds, dec_zs_targets, all_zs_preds, all_zs_targets = decimate_predictions(zs_preds, zs_targets, args.n_points)
            fig_zs = plot_decimated_predictions(dec_zs_preds, dec_zs_targets)
            writer.add_figure("Zero-shot_Predictions_Final", fig_zs, global_step)
            plt.close(fig_zs)
            save_predictions_to_csv(all_zs_preds, all_zs_targets, "zs_draw", run_log_dir)

    writer.close()
    print("Training finished!")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared!")
