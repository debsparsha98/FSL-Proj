# train_visdrone.py
import torch
import os
import shutil
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils.distributed_training_utils import setup_device

def main():
    # Setup device with better configuration
    setup_device(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if we have a checkpoint to resume from
    checkpoint_dir = 'D:/datasets/visdrone/checkpoints/yolo_nas_visdrone_s/RUN_20250830_220846_216883'
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_latest.pth')
    
    # Check for various checkpoint file naming conventions
    possible_checkpoint_names = ['ckpt_latest.pth', 'clxpt_latest.pth', 'best_visdrone.pth']
    found_checkpoint = None
    
    for checkpoint_name in possible_checkpoint_names:
        potential_path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.exists(potential_path):
            found_checkpoint = potential_path
            break
    
    # Don't clean checkpoints if we're resuming
    if found_checkpoint:
        print(f"Resuming training from checkpoint: {found_checkpoint}")
        checkpoint_path = found_checkpoint
    else:
        print("No checkpoint found, starting fresh training")
        # Clean previous checkpoints only if not resuming
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            print("Cleaned previous checkpoints")

    # Dataset configuration with memory-efficient parameters
    dataset_params = {
        'data_dir': 'D:/datasets/visdrone',
        'train_images_dir': 'VisDrone2019-DET-train/images',
        'train_labels_dir': 'VisDrone2019-DET-train/labels',
        'val_images_dir': 'VisDrone2019-DET-val/images', 
        'val_labels_dir': 'VisDrone2019-DET-val/labels',
        'test_images_dir': 'VisDrone2019-DET-test-dev/images',
        'classes': ['ignored_region', 'pedestrian', 'people', 'bicycle', 'car',
                   'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'],
        'input_dim': [640, 640]  # Reduced for 6GB GPU
    }
    
    # Optimize CUDA settings for limited memory
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Set conservative GPU memory allocation
        torch.cuda.set_per_process_memory_fraction(0.7)  # Reduced from 0.9
    
    # DataLoaders with memory-efficient parameters
    print("Setting up DataLoaders...")

    train_loader = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes'],
            'input_dim': dataset_params['input_dim'],
            'cache_annotations': False,  # Disabled to save memory
            #'transforms': transforms
        },
        dataloader_params={
            'batch_size': 2,  # Reduced batch size for 6GB GPU
            'num_workers': 2,  # Reduced workers
            'shuffle': True,
            'pin_memory': True,
            'drop_last': True
        }
    )
    
    val_loader = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes'],
            'input_dim': dataset_params['input_dim'],
            'cache_annotations': False  # Disabled to save memory
        },
        dataloader_params={
            'batch_size': 2,
            'num_workers': 2,
            'shuffle': False,
            'pin_memory': True
        }
    )
    
    # Model initialization
    print("Loading model...")
    
    model = models.get(
        model_name='yolo_nas_s',
        num_classes=len(dataset_params['classes']),
        pretrained_weights=None  # Will be overwritten by checkpoint if resuming
    )
    
    # Loss function
    loss_fn = PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes'])
    )
   
    # Validation metrics
    post_prediction_callback = PPYoloEPostPredictionCallback(
        score_threshold=0.01,
        nms_top_k=1000,
        max_predictions=300,
        nms_threshold=0.7
    )

    val_metrics = DetectionMetrics_050(
        score_thres=0.1,
        top_k_predictions=300,
        num_cls=len(dataset_params['classes']),
        normalize_targets=True,
        post_prediction_callback=post_prediction_callback
    )
    
    # Trainer
    trainer = Trainer(
        experiment_name='yolo_nas_visdrone_s',
        ckpt_root_dir='./checkpoints'
    )
    
    # Memory-efficient training parameters
    training_params = {
        'max_epochs': 200,
        'resume': True if found_checkpoint else False,  # Set to True if checkpoint exists
        'resume_path': checkpoint_path if found_checkpoint else None,  # Explicit path to checkpoint
        'ckpt_name': 'ckpt_latest.pth',  # Default checkpoint name
        'optimizer': 'AdamW',
        'optimizer_params': {'weight_decay': 0.0001},
        'zero_weight_decay_on_bias_and_bn': True,
        
        'initial_lr': 0.001,
        'lr_mode': 'cosine',
        'cosine_final_lr_ratio': 0.01,
        'lr_warmup_epochs': 5,
        'warmup_mode': 'linear_epoch_step',
        'loss': loss_fn,
        
        'ema': False,  # Disabled to save memory
        'ema_params': {'decay': 0.9999, 'decay_type': 'exp'},
        
        'train_metrics_list': [],
        'valid_metrics_list': [val_metrics],
        'metric_to_watch': 'mAP@0.50',
        'greater_metric_to_watch_is_better': True,
        
        'save_ckpt_epoch_list': [25, 50, 75],
        'ckpt_best_name': 'best_visdrone.pth',
        'save_checkpoints': True,
        
        'average_best_models': False,  # Disabled to save memory
        'log_instances': False,  # Disabled to save memory
        'mixed_precision': True,  # Enabled to save memory
        'silent_mode': False,
        
        'batch_accumulate': 2,  # Gradient accumulation to simulate larger batch
        'run_validation_freq': 2,  # Validate less frequently
        'save_model': True,
        
        # Memory-saving settings
        'clip_grad_norm': 5.0,
      
    }
    
    # Start training with proper resume handling
    print("Starting training...")
    
    try:
        # Clear memory before starting
        torch.cuda.empty_cache()
        
       
        trainer.train(
            model=model,
            training_params=training_params,
            train_loader=train_loader,
            valid_loader=val_loader
        )
        
        print("Training completed successfully!")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU out of memory! Trying to recover...")
            torch.cuda.empty_cache()
            # Try with even smaller batch size
            print("Please try with even smaller batch size or input dimension")
        else:
            print(f"Training failed with error: {e}")
        
        # Save error information
        torch.save({
            'error': str(e)
        }, 'training_error_checkpoint.pth')
        print("Error information saved as 'training_error_checkpoint.pth'")
    
    except Exception as e:
        print(f"Training failed with error: {e}")

if __name__ == '__main__':
    main()