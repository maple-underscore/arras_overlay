yolo task=detect mode=train model=yolo26s.pt \
    data=arras_data.yaml \
    epochs=25 \
    imgsz=256 \
    batch=16 \
    device=mps \
    workers=4 \
    name=arras_train_fast \
    val=False \
    half=True \
    augment=True
