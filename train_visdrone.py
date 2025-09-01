# train_visdrone.py
import torch
import os
import shutil
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

def main():
     # Clean previous checkpoints to avoid loading issues
    checkpoint_dir = './checkpoints/yolo_nas_visdrone_l'
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print("Cleaned previous checkpoints")	

    # Dataset configuration
    dataset_params = {
        'data_dir': 'D:/datasets/visdrone',
        'train_images_dir': 'VisDrone2019-DET-train/images',
        'train_labels_dir': 'VisDrone2019-DET-train/labels',
        'val_images_dir': 'VisDrone2019-DET-val/images', 
        'val_labels_dir': 'VisDrone2019-DET-val/labels',
        'test_images_dir': 'VisDrone2019-DET-test-dev/images',
        'classes': ['ignored_region', 'pedestrian', 'people', 'bicycle', 'car',
                   'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
    }
    
    # DataLoaders
    print("Setting up DataLoaders...")

# Add these lines before training starts
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Optimizes CUDA operations
    torch.backends.cudnn.enabled = True

    train_loader = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes'],
	    'input_dim': [512, 512]  # Reduced image size for memory
        },
        dataloader_params={
            'batch_size': 2, #8
            'num_workers': 2, #4
            'shuffle': True
        }
    )
    
    val_loader = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes'],
	    'input_dim': [512, 512]  # Reduced image size for memory
        },
        dataloader_params={
            'batch_size': 2,
            'num_workers':2, #4
            'shuffle': False
        }
    )
    
    # Model - using local weights
    print("Loading model with local weights...")
    local_weights_path = "D:/datasets/visdrone/yolo_nas_l_coco.pth"
    
    model = models.get(
        model_name='yolo_nas_s',      # Must match your weights file
        num_classes=len(dataset_params['classes']),
        #checkpoint_path=local_weights_path
	pretrained_weights=None
    )
    
    # Loss function
    loss_fn = PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes'])
    )
   
    # Validation metrics with post-processing
    post_prediction_callback = PPYoloEPostPredictionCallback(
        score_threshold=0.01,
        nms_top_k=1000,
        max_predictions=300,
        nms_threshold=0.7
    )

    val_metrics = DetectionMetrics_050(
        score_thres=0.1,  # Added score_thres
        top_k_predictions=300,  # Added top_k_predictions
        num_cls=len(dataset_params['classes']),
        normalize_targets=True,
        post_prediction_callback=post_prediction_callback
    )
    
    # Trainer
    trainer = Trainer(
        experiment_name='yolo_nas_visdrone_s',
        ckpt_root_dir='./checkpoints'
    )
    
    # Training parameters
    training_params = {
        'max_epochs': 100,
        'optimizer': 'AdamW',
        'optimizer_params': {'weight_decay': 0.0001},
        'zero_weight_decay_on_bias_and_bn': True,
        
        #'lr': 0.001,
	'initial_lr': 0.001,
        'lr_mode': 'cosine',
        'cosine_final_lr_ratio': 0.01,
        'lr_warmup_epochs': 5,
        'warmup_mode': 'linear_epoch_step',
        'loss': loss_fn,
        #'criterion': loss_fn,
        'ema': True,
        'ema_params': {'decay': 0.9, 'decay_type': 'threshold'},
        
        'train_metrics_list': [],
        'valid_metrics_list': [val_metrics],
        'metric_to_watch': 'mAP@0.50',
        'greater_metric_to_watch_is_better': True,
        
        'save_ckpt_epoch_list': [25, 50, 75],
        'ckpt_best_name': 'best_visdrone.pth',
        'save_checkpoints': True,
        
        'average_best_models': False, #true
        'log_instances': True,
	    #addition
	    'mixed_precision': True,
        'silent_mode': False,
	    # Additional recommended parameters
        'batch_accumulate': 1,
        'run_validation_freq': 1,
        'save_model': True,
    }
    
    # Start training
    print("Starting training...")
    trainer.train(
        model=model,
        training_params=training_params,
        train_loader=train_loader,
        valid_loader=val_loader,
        #resume=True  # CRITICAL: This will resume from the last checkpoint
    )
    
    print("Training completed!")

if __name__ == '__main__':
    main()