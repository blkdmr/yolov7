python train.py --epochs 100 --workers 4 --device 0 --batch-size 32 \
--data dataset/potatoes/data.yaml --img 640 640 --cfg dataset/potatoes/conf.yaml \
--weights 'yolov7-tiny.pt' --name yolov7_tiny_potatoes --hyp yolov7/data/hyp.scratch.tiny.yaml


python test.py --weights runs/train/yolov7_tiny_potatoes/weights/best.pt --task test --data dataset/potatoes/data.yaml