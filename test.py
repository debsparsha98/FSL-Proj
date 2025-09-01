# First, let's discover what's available
'''
import super_gradients.training.datasets.datasets_conf as datasets_conf
from super_gradients.training import dataloaders
#from super_gradients.training.dataloaders.dataloaders import DATA_LOADER_CONFIGS

print("Available dataloaders:")
for name in dir(dataloaders):
    if "coco" in name.lower() or "yolo" in name.lower() or "detection" in name.lower():
        print(f"  - {name}")

print("Available classes in datasets_conf:")
for name in dir(datasets_conf):
    if 'COCO' in name or 'YOLO' in name or 'DETECTION' in name:
        print(f"  - {name}")

# Also check what's in the main datasets module
import super_gradients.training.datasets as datasets
print("\nAvailable classes in main datasets module:")
for name in dir(datasets):
    if 'COCO' in name or 'YOLO' in name or 'DETECTION' in name:
        print(f"  - {name}")

print("Available config names:")
for config_name in DATA_LOADER_CONFIGS.keys():
    print(f"  - {config_name}")

print("\nSimple config names (try these):")
simple_configs = [name for name in DATA_LOADER_CONFIGS.keys() if not name.startswith('coco2017_')]
for config_name in simple_configs:
    print(f"  - {config_name}")

# discovery.py
import super_gradients.training.dataloaders.dataloaders as dataloaders_module

print("Available items in dataloaders module:")
for name in dir(dataloaders_module):
    if not name.startswith('_'):
        print(f"  - {name}")

# Also check if there are any pre-defined configs
try:
    if hasattr(dataloaders_module, 'DATA_LOADER_CONFIGS'):
        print("\nDATA_LOADER_CONFIGS found:")
        for config_name in dataloaders_module.DATA_LOADER_CONFIGS.keys():
            print(f"  - {config_name}")
    else:
        print("\nNo DATA_LOADER_CONFIGS found in this version")
except:
    print("\nCould not access DATA_LOADER_CONFIGS")

# find_collate_functions.py
import inspect
import super_gradients.training.datasets as datasets

print("Searching for available collate functions in SuperGradients...")

# Check all modules for collate functions
collate_functions = []

# Check datasets_utils
try:
    from super_gradients.training.datasets import datasets_utils
    for name in dir(datasets_utils):
        if 'collate' in name.lower() or 'Collate' in name:
            obj = getattr(datasets_utils, name)
            if inspect.isclass(obj) or callable(obj):
                collate_functions.append((f"datasets_utils.{name}", obj))
except ImportError:
    pass

# Check main datasets module
for name in dir(datasets):
    if 'collate' in name.lower() or 'Collate' in name:
        obj = getattr(datasets, name)
        if inspect.isclass(obj) or callable(obj):
            collate_functions.append((f"datasets.{name}", obj))

# Check dataloaders module
try:
    import super_gradients.training.dataloaders.dataloaders as dataloaders
    for name in dir(dataloaders):
        if 'collate' in name.lower() or 'Collate' in name:
            obj = getattr(dataloaders, name)
            if inspect.isclass(obj) or callable(obj):
                collate_functions.append((f"dataloaders.{name}", obj))
except ImportError:
    pass

print(f"\nFound {len(collate_functions)} collate functions:")
for location, func in collate_functions:
    print(f"  - {location}")

if not collate_functions:
    print("No collate functions found. You may need to create your own.")

# check_recipes.py
from super_gradients.training import training_hyperparams
import super_gradients.training.training_hyperparams as hyperparams_module

print("Available training hyperparams:")
for name in dir(hyperparams_module):
    if not name.startswith('_') and name not in ['get', 'load_recipe', 'training_hyperparams']:
        print(f"  - {name}")

# Also try to see what's available in the recipes directory
import os
recipes_path = "C:/Users/debsp/AppData/Local/Programs/Python/Python310/lib/site-packages/super_gradients/recipes"
if os.path.exists(recipes_path):
    print(f"\nAvailable recipe files:")
    for file in os.listdir(recipes_path):
        if file.endswith('.yaml'):
            print(f"  - {file}")
else:
    print(f"\nRecipes path not found: {recipes_path}")





# check_losses_corrected.py
from super_gradients.common.factories.losses_factory import LossesFactory
import torch

def check_available_losses():
    print("Checking available loss functions with proper parameters...")
    
    factory = LossesFactory()
    
    print("\nTesting detection loss instantiation with proper parameters:")
    
    detection_losses_to_test = [
        {
            'name': 'YoloXDetectionLoss',
            'params': {'strides': [8, 16, 32], 'num_classes': 9}
        },
        {
            'name': 'PPYoloELoss', 
            'params': {'num_classes': 9}
        },
        {
            'name': 'YoloXFastDetectionLoss',
            'params': {'strides': [8, 16, 32], 'num_classes': 9}
        },
        {
            'name': 'SSDLoss',
            'params': {'dboxes': torch.randn(8732, 4)}  # Example boxes
        }
    ]
    
    working_losses = []
    for loss_info in detection_losses_to_test:
        loss_name = loss_info['name']
        loss_params = loss_info['params']
        
        try:
            loss_instance = factory.get({loss_name: loss_params})
            print(f"✅ {loss_name}: Can instantiate with params {list(loss_params.keys())}")
            working_losses.append((loss_name, loss_params))
        except Exception as e:
            print(f"❌ {loss_name}: Error - {e}")
    
    return working_losses

if __name__ == '__main__':
    working_losses = check_available_losses()
    
    if working_losses:
        best_loss, best_params = working_losses[0]
        print(f"\nRecommended: Use '{best_loss}' with params: {best_params}")
    else:
        print("\n❌ No detection losses could be instantiated. Using CrossEntropy as fallback.")





# check_warmup_modes.py
import super_gradients.training.utils.callbacks.callbacks as callbacks_module

print("Available items in callbacks module:")
for name in dir(callbacks_module):
    if 'warmup' in name.lower() or 'WARMUP' in name or 'LR' in name:
        print(f"  - {name}")

# Also check what's in the training params
from super_gradients.training import training_hyperparams
import super_gradients.training.training_hyperparams as hyperparams_module

print("\nAvailable training hyperparams attributes:")
for name in dir(hyperparams_module):
    if not name.startswith('_'):
        print(f"  - {name}")





# debug_model.py
from super_gradients.training import models
import torch

# Load model
model = models.get("yolo_nas_l", num_classes=9)

# Create a dummy input
dummy_input = torch.randn(2, 3, 640, 640)  # batch_size=2, channels=3, height=640, width=640

# Set model to eval mode
model.eval()

# Get model outputs
with torch.no_grad():
    outputs = model(dummy_input)
    
print(f"Number of outputs: {len(outputs)}")
for i, output in enumerate(outputs):
    print(f"Output {i} type: {type(output)}")
    if isinstance(output, tuple):
        print(f"Output {i} is tuple with {len(output)} elements:")
        for j, element in enumerate(output):
            print(f"  Element {j}: type={type(element)}, shape={element.shape if hasattr(element, 'shape') else 'No shape'}")
    else:
        print(f"Output {i} shape: {output.shape}")





# check_yolo_nas_losses.py
from super_gradients.common.factories.losses_factory import LossesFactory

factory = LossesFactory()
print("Looking for YOLO-NAS specific losses:")
for loss_name in factory.type_dict.keys():
    if 'nas' in loss_name.lower() or 'yolo_nas' in loss_name.lower():
        print(f"  - {loss_name}")

# Also check what's available
print("\nAll available losses:")
for loss_name in sorted(factory.type_dict.keys()):
    if not loss_name.startswith('_'):
        print(f"  - {loss_name}")





import importlib

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} is available")
        return True
    except ImportError:
        print(f"✗ {module_name} is NOT available")
        return False

# Check all required modules
modules_to_check = [
    'torch',
    'super_gradients.training',
    'super_gradients.training.dataloaders.dataloaders',
    'super_gradients.training.losses',
    'super_gradients.training.metrics',
    'super_gradients.training.models.detection_models.pp_yolo_e'
]

print("Checking required modules...")
all_available = True

for module in modules_to_check:
    if not check_module(module):
        all_available = False

# Check specific classes within modules
print("\nChecking specific classes...")

# Check PPYoloELoss
try:
    from super_gradients.training.losses import PPYoloELoss
    print("✓ PPYoloELoss is available")
except ImportError:
    print("✗ PPYoloELoss is NOT available")
    all_available = False

# Check DetectionMetrics_050
try:
    from super_gradients.training.metrics import DetectionMetrics_050
    print("✓ DetectionMetrics_050 is available")
except ImportError:
    print("✗ DetectionMetrics_050 is NOT available")
    all_available = False

# Check PPYoloEPostPredictionCallback
try:
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
    print("✓ PPYoloEPostPredictionCallback is available")
except ImportError:
    print("✗ PPYoloEPostPredictionCallback is NOT available")
    all_available = False

# Check dataloader functions
try:
    from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
    print("✓ coco_detection_yolo_format_train and coco_detection_yolo_format_val are available")
except ImportError:
    print("✗ coco_detection_yolo_format_train and/or coco_detection_yolo_format_val are NOT available")
    all_available = False

# Check Trainer and models
try:
    from super_gradients.training import Trainer, models
    print("✓ Trainer and models are available")
except ImportError:
    print("✗ Trainer and/or models are NOT available")
    all_available = False

print(f"\nAll modules available: {all_available}")

if not all_available:
    print("\nTo install missing dependencies, run:")
    print("pip install super-gradients")
    print("pip install torch torchvision")
else:
    print("\nAll required modules are installed! You can proceed with your YOLO training code.")





from super_gradients.training import Trainer

# Initialize trainer first
trainer = Trainer(experiment_name="test", ckpt_root_dir="checkpoints")

# Get detailed help with all parameters
help(trainer.train)


'''


from super_gradients.training import transforms
import inspect

# List all available detection transforms
transform_classes = [name for name, obj in inspect.getmembers(transforms) 
                    if inspect.isclass(obj) and 'Detection' in name]
print("Available detection transforms:", transform_classes)