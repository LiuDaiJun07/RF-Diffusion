import numpy as np
import os
from scipy.io import savemat
from scipy.interpolate import interp1d

# 配置项
SIGNAL_KEY = "bin"
COND_KEY = "index"
DATA_TYPE = np.complex128  # 保真：改回 complex128

def npz_to_mat_rf_diffusion(NPZ_INPUT_PATH, MAT_OUTPUT_PATH):
    try:
        # 1. 读取NPZ
        print(f"[1/7] 读取NPZ：{NPZ_INPUT_PATH}")
        try:
            npz_data = np.load(NPZ_INPUT_PATH, allow_pickle=True)
        except FileNotFoundError:
            raise Exception(f"找不到文件：{NPZ_INPUT_PATH}")
        except Exception as e:
            raise Exception(f"读取失败：{str(e)}")

        if SIGNAL_KEY not in npz_data:
            raise Exception(f"无 '{SIGNAL_KEY}' 键，可用：{list(npz_data.keys())}")
        if COND_KEY not in npz_data:
            raise Exception(f"无 '{COND_KEY}' 键，可用：{list(npz_data.keys())}")
        
        complex_signals = npz_data[SIGNAL_KEY]
        cond_label = npz_data[COND_KEY]
        print(f"NPZ数据：信号={complex_signals.shape}，条件={cond_label.shape}")

        # 2. 去除冗余维度
        complex_signals = complex_signals.squeeze()
        print(f"[2/7] 去冗余后：{complex_signals.shape}")

        # 3. 插值到512 —— 完全保真，无任何 round
        num_samples = complex_signals.shape[0]
        target_time_len = 512
        signals_interpolated = []
        print(f"[3/7] 插值到512维度...")
        
        for i in range(num_samples):
            single_signal = complex_signals[i]
            interp_real = interp1d(
                np.arange(30), single_signal.real, 
                kind="linear", fill_value="extrapolate", bounds_error=False
            )
            interp_imag = interp1d(
                np.arange(30), single_signal.imag, 
                kind="linear", fill_value="extrapolate", bounds_error=False
            )
            # 保真：不做任何 round
            single_interp = interp_real(np.arange(target_time_len)) + 1j * interp_imag(np.arange(target_time_len))
            signals_interpolated.append(single_interp)
        
        complex_signals = np.array(signals_interpolated, dtype=DATA_TYPE)
        print(f"  插值完成：{complex_signals.shape}，类型={DATA_TYPE}")

        # 4. 调整为 (512,128)
        print(f"[4/7] 调整为(512,128)...")
        single_sample = complex_signals[0]
        feature = np.tile(single_sample[:, np.newaxis], (1, 128))
        print(f"  feature：{feature.shape}")

        # 5. cond 变成 (1,6) uint8
        print(f"[5/7] 调整cond为(1,6)...")
        cond_6d = np.pad(cond_label, (0, 6 - len(cond_label)), mode='constant')
        cond = cond_6d[np.newaxis, :].astype(np.uint8)
        print(f"  cond：{cond.shape} uint8")

        # 6. 功率归一化（论文要求，不是失真）
        print(f"[6/7] 归一化...")
        l2_norms = np.linalg.norm(feature, axis=0)
        avg_l2_norm = np.mean(l2_norms) if np.mean(l2_norms) > 1e-8 else 1e-8
        feature = feature / avg_l2_norm * 100
        print(f"  归一化完成：平均L2={avg_l2_norm:.6f}")

        # 7. 保存：无 h5py，不改变 MAT 版本，只开原生无损压缩
        print(f"[7/7] 保存MAT：{MAT_OUTPUT_PATH}")
        mat_data = {
            "feature": feature,
            "cond": cond
        }
        savemat(MAT_OUTPUT_PATH, mat_data, do_compression=True)

        mat_size = os.path.getsize(MAT_OUTPUT_PATH) / 1024
        print(f"  ✅ 保存完成：体积≈{mat_size:.1f}KB")
        print(f"  feature：{feature.shape} | {DATA_TYPE}")
        print(f"  cond：{cond.shape} | uint8")

    except Exception as e:
        print(f"\n❌ 转换失败：{str(e)}")
        return False
    return True

def batch_process_directory(src_root, dst_root):
    if not os.path.isdir(src_root):
        print(f"【错误】源目录不存在：{src_root}")
        return
    
    file_count = 0
    success_count = 0
    for root, dirs, files in os.walk(src_root):
        relative_path = os.path.relpath(root, src_root)
        dst_dir = os.path.join(dst_root, relative_path)
        os.makedirs(dst_dir, exist_ok=True)
        
        for file_name in files:
            if not file_name.endswith('.npz'):
                continue
            
            src_file = os.path.join(root, file_name)
            dst_file = os.path.join(dst_dir, os.path.splitext(file_name)[0] + '.mat')
            
            print(f"\n===== 处理：{src_file} =====")
            try:
                success = npz_to_mat_rf_diffusion(src_file, dst_file)
                if success:
                    success_count += 1
                    print(f"✅ 成功：{dst_file}")
                else:
                    print(f"❌ 失败：{dst_file}")
                file_count += 1
            except Exception as e:
                print(f"❌ 出错：{str(e)}")
    
    print(f"\n===== 批量处理完成 =====")
    print(f"总计文件数：{file_count}")
    print(f"成功数：{success_count}")
    print(f"失败数：{file_count - success_count}")

if __name__ == "__main__":
    SOURCE_DIR = r"/data/coding/npz"
    TARGET_DIR = r"/data/coding/mat"
    batch_process_directory(SOURCE_DIR, TARGET_DIR)