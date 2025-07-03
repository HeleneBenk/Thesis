import os
os.environ["WANDB_API_KEY"] = "NO_WANDB"
os.environ["WANDB_DISABLED"] = "true"
os.environ["RAY_AIR_NEW_LOGGING"] = "0"

from ultralytics import YOLO
from ray import tune
import yaml

model = YOLO("yolo11m.pt")


search_space = {
    "lr0": tune.uniform(1e-4, 1e-1),
    "momentum": tune.uniform(0.7, 0.98),
    "box": tune.uniform(0.02, 0.2),
    "cls": tune.uniform(0.2, 4.0)
}


result_grid = model.tune(
    data="/cfs/earth/scratch/benkehel/YOLO/dataset.yaml",
    space=search_space,
    epochs=50,                
    use_ray=True,
    gpu_per_trial=1,          
    iterations=12,            
    batch=8,                 
    imgsz=640
)

best_result = result_grid.get_best_result()
print("Best config:", best_result.config)
print("Best mAP50-95:", best_result.metrics["metrics/box/map"])

df = result_grid.get_dataframe()
df.to_csv("all_trials.csv", index=False)


with open("best_hyperparameters.yaml", "w") as f:
    yaml.dump(best_result.config, f)
