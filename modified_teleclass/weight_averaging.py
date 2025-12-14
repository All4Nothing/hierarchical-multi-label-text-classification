import torch
import os
import copy

def average_checkpoints(model_dir, output_path, epoch_list=[1, 2, 3]):
    """
    Epoch 1, 2, 3의 가중치를 평균내어 새로운 best_model을 만듭니다.
    """
    print(f"Averaging checkpoints: {epoch_list} from {model_dir}...")
    
    avg_state_dict = None
    
    for epoch in epoch_list:
        ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch}", "pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        if avg_state_dict is None:
            avg_state_dict = state_dict
        else:
            for key in state_dict:
                avg_state_dict[key] += state_dict[key]
                
    # 평균 계산
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key] / len(epoch_list)
        
    # 저장
    os.makedirs(output_path, exist_ok=True)
    torch.save(avg_state_dict, os.path.join(output_path, "pytorch_model.bin"))
    print(f"Model Soup saved to {output_path}")

# 사용 예시 (Pipeline run 메서드 마지막이나 별도 실행)
target_dir = "outputs/models"
average_checkpoints(target_dir, "outputs/models/model_soup", epoch_list=[1, 2])