# Projeto Final - Modelos Preditivos Conexionistas

### Jammesson Cabral

| **Tipo de Projeto**          | **Modelo Selecionado** | **Linguagem** |
| ---------------------------- | ---------------------- | ------------- |
| Classificação de Imagens<br> | YOLOv5                 | PyTorch       |

## Performance

O modelo treinado possui performance de **81%**.

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```text
  !python classify/train.py --model yolov5s-cls.pt --data $DATASET_NAME --epochs 100 --img 128 --pretrained weights/yolov5s-cls.pt

wandb: Currently logged in as: jamcabral. Use `wandb login --relogin` to force relogin
classify/train: model=yolov5s-cls.pt, data=car_mot_cam-2, epochs=100, batch_size=64, imgsz=128, nosave=False, cache=None, device=, workers=8, project=runs/train-cls, name=exp, exist_ok=False, pretrained=weights/yolov5s-cls.pt, optimizer=Adam, lr0=0.001, decay=5e-05, label_smoothing=0.1, cutoff=None, dropout=None, verbose=False, seed=0, local_rank=-1
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.2-227-g78ed31c Python-3.7.15 torch-1.12.1+cu113 CPU

TensorBoard: Start with 'tensorboard --logdir runs/train-cls', view at http://localhost:6006/
wandb: Tracking run with wandb version 0.13.5
wandb: Run data is saved locally in /content/yolov5/wandb/run-20221106_145745-22gwv7qk
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run legendary-water-6
wandb: ⭐️ View project at https://wandb.ai/jamcabral/YOLOv5-Classify
wandb: 🚀 View run at https://wandb.ai/jamcabral/YOLOv5-Classify/runs/22gwv7qk
albumentations: RandomResizedCrop(p=1.0, height=128, width=128, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1), HorizontalFlip(p=0.5), ColorJitter(p=0.5, brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[0, 0]), Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0), ToTensorV2(always_apply=True, p=1.0, transpose_mask=False)
Model summary: 149 layers, 4176323 parameters, 4176323 gradients, 10.5 GFLOPs
optimizer: Adam(lr=0.001) with parameter groups 32 weight(decay=0.0), 33 weight(decay=5e-05), 33 bias
Image sizes 128 train, 128 test
Using 1 dataloader workers
Logging results to runs/train-cls/exp4
Starting yolov5s-cls.pt training on car_mot_cam-2 dataset with 3 classes for 100 epochs...

     Epoch   GPU_mem  train_loss   test_loss    top1_acc    top5_acc
     1/100        0G        1.19        1.17      0.0172           1: 100% 5/5 [00:14<00:00,  2.90s/it]
     2/100        0G        1.04        1.32      0.0172           1: 100% 5/5 [00:13<00:00,  2.79s/it]
     3/100        0G        1.01        1.46      0.0172           1: 100% 5/5 [00:13<00:00,  2.78s/it]
     4/100        0G        1.03         1.6      0.0172           1: 100% 5/5 [00:13<00:00,  2.79s/it]
     5/100        0G        1.04        1.54      0.0172           1: 100% 5/5 [00:14<00:00,  2.82s/it]
     6/100        0G       0.966        1.43      0.0172           1: 100% 5/5 [00:14<00:00,  2.82s/it]
     7/100        0G       0.976        1.46      0.0172           1: 100% 5/5 [00:13<00:00,  2.77s/it]
     8/100        0G       0.957        1.47      0.0172           1: 100% 5/5 [00:13<00:00,  2.76s/it]
     9/100        0G       0.893        1.51      0.0172           1: 100% 5/5 [00:13<00:00,  2.78s/it]
    10/100        0G       0.915         1.5      0.0172           1: 100% 5/5 [00:13<00:00,  2.77s/it]
    11/100        0G       0.906         1.3      0.0172           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    12/100        0G       0.878        1.24      0.0172           1: 100% 5/5 [00:13<00:00,  2.78s/it]
    13/100        0G       0.867        1.37      0.0172           1: 100% 5/5 [00:13<00:00,  2.77s/it]
    14/100        0G       0.815        1.29      0.0172           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    15/100        0G       0.881        1.19      0.0172           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    16/100        0G       0.853        1.47      0.0172           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    17/100        0G       0.803        1.35      0.0517           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    18/100        0G       0.819        1.24       0.069           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    19/100        0G       0.807        1.15        0.19           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    20/100        0G        0.83        1.22      0.0862           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    21/100        0G       0.773        1.29      0.0862           1: 100% 5/5 [00:13<00:00,  2.78s/it]
    22/100        0G       0.839        1.12      0.0862           1: 100% 5/5 [00:13<00:00,  2.77s/it]
    23/100        0G       0.767         1.4      0.0345           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    24/100        0G       0.773        1.55      0.0345           1: 100% 5/5 [00:14<00:00,  2.87s/it]
    25/100        0G       0.763        1.42      0.0345           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    26/100        0G       0.762        1.42      0.0345           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    27/100        0G       0.736        1.33      0.0345           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    28/100        0G       0.731        1.47      0.0345           1: 100% 5/5 [00:13<00:00,  2.77s/it]
    29/100        0G       0.757        1.43      0.0172           1: 100% 5/5 [00:13<00:00,  2.79s/it]
    30/100        0G       0.744        1.43      0.0345           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    31/100        0G       0.771        1.41      0.0345           1: 100% 5/5 [00:13<00:00,  2.79s/it]
    32/100        0G       0.781        1.33      0.0345           1: 100% 5/5 [00:13<00:00,  2.79s/it]
    33/100        0G       0.744        1.55      0.0172           1: 100% 5/5 [00:13<00:00,  2.78s/it]
    34/100        0G        0.74        1.48      0.0345           1: 100% 5/5 [00:13<00:00,  2.79s/it]
    35/100        0G       0.779         1.2       0.103           1: 100% 5/5 [00:13<00:00,  2.78s/it]
    36/100        0G       0.734        1.21      0.0862           1: 100% 5/5 [00:13<00:00,  2.79s/it]
    37/100        0G       0.766        1.36      0.0172           1: 100% 5/5 [00:13<00:00,  2.77s/it]
    38/100        0G       0.754        1.44      0.0172           1: 100% 5/5 [00:13<00:00,  2.77s/it]
    39/100        0G       0.744        1.47      0.0172           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    40/100        0G       0.738        1.18      0.0517           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    41/100        0G       0.718        1.29      0.0517           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    42/100        0G       0.716        1.45      0.0345           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    43/100        0G       0.733        1.35      0.0345           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    44/100        0G       0.727        1.45      0.0345           1: 100% 5/5 [00:13<00:00,  2.79s/it]
    45/100        0G       0.737        1.59      0.0345           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    46/100        0G       0.725        1.67      0.0172           1: 100% 5/5 [00:13<00:00,  2.72s/it]
    47/100        0G       0.721        1.32      0.0345           1: 100% 5/5 [00:13<00:00,  2.72s/it]
    48/100        0G       0.697        1.56      0.0345           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    49/100        0G        0.71        1.52      0.0345           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    50/100        0G       0.728        1.17       0.155           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    51/100        0G       0.706        1.49      0.0172           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    52/100        0G       0.704        1.61      0.0172           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    53/100        0G       0.686        1.32      0.0345           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    54/100        0G       0.705         1.5      0.0862           1: 100% 5/5 [00:13<00:00,  2.71s/it]
    55/100        0G       0.701         1.5      0.0345           1: 100% 5/5 [00:13<00:00,  2.72s/it]
    56/100        0G       0.633        1.45      0.0862           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    57/100        0G       0.718        1.75      0.0517           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    58/100        0G       0.753        1.66      0.0517           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    59/100        0G       0.709         1.1       0.397           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    60/100        0G       0.665        1.42        0.19           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    61/100        0G       0.659       0.871        0.69           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    62/100        0G       0.665       0.842       0.672           1: 100% 5/5 [00:13<00:00,  2.72s/it]
    63/100        0G       0.676       0.845       0.707           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    64/100        0G       0.648        1.03       0.552           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    65/100        0G       0.604       0.774       0.776           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    66/100        0G       0.656        1.31       0.431           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    67/100        0G       0.574         1.8       0.172           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    68/100        0G       0.617       0.886       0.672           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    69/100        0G       0.604        1.24       0.397           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    70/100        0G       0.562        1.64       0.328           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    71/100        0G       0.592        1.09       0.517           1: 100% 5/5 [00:13<00:00,  2.76s/it]
    72/100        0G       0.537        1.35       0.414           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    73/100        0G       0.573        1.26       0.397           1: 100% 5/5 [00:13<00:00,  2.78s/it]
    74/100        0G       0.576        1.06       0.534           1: 100% 5/5 [00:13<00:00,  2.78s/it]
    75/100        0G       0.576        1.13       0.517           1: 100% 5/5 [00:13<00:00,  2.77s/it]
    76/100        0G       0.558        1.35       0.414           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    77/100        0G       0.577        1.86       0.155           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    78/100        0G       0.612        1.12       0.517           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    79/100        0G       0.574        1.12       0.552           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    80/100        0G       0.583         1.9       0.224           1: 100% 5/5 [00:13<00:00,  2.78s/it]
    81/100        0G        0.57        1.25       0.397           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    82/100        0G       0.548       0.609        0.81           1: 100% 5/5 [00:13<00:00,  2.72s/it]
    83/100        0G       0.552        1.21       0.534           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    84/100        0G       0.564        1.54       0.397           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    85/100        0G       0.535        1.54       0.414           1: 100% 5/5 [00:13<00:00,  2.72s/it]
    86/100        0G       0.549        1.34       0.483           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    87/100        0G       0.499         1.1       0.534           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    88/100        0G       0.555        1.01       0.569           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    89/100        0G       0.504        1.02       0.603           1: 100% 5/5 [00:13<00:00,  2.72s/it]
    90/100        0G       0.521        1.05       0.552           1: 100% 5/5 [00:13<00:00,  2.72s/it]
    91/100        0G       0.568        1.19         0.5           1: 100% 5/5 [00:13<00:00,  2.70s/it]
    92/100        0G       0.475        1.22       0.552           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    93/100        0G       0.482        1.32       0.483           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    94/100        0G       0.507        1.24         0.5           1: 100% 5/5 [00:13<00:00,  2.74s/it]
    95/100        0G        0.48        1.03       0.603           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    96/100        0G       0.509        1.02       0.603           1: 100% 5/5 [00:13<00:00,  2.73s/it]
    97/100        0G       0.484        1.07       0.603           1: 100% 5/5 [00:13<00:00,  2.75s/it]
    98/100        0G       0.512        1.16       0.586           1: 100% 5/5 [00:13<00:00,  2.77s/it]
    99/100        0G       0.449        1.21       0.517           1: 100% 5/5 [00:13<00:00,  2.77s/it]

100/100 0G 0.506 1.19 0.534 1: 100% 5/5 [00:13<00:00, 2.73s/it]

Training complete (0.385 hours)
Results saved to runs/train-cls/exp4
Predict: python classify/predict.py --weights runs/train-cls/exp4/weights/best.pt --source im.jpg
Validate: python classify/val.py --weights runs/train-cls/exp4/weights/best.pt --data car_mot_cam-2
Export: python export.py --weights runs/train-cls/exp4/weights/best.pt --include onnx
PyTorch Hub: model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train-cls/exp4/weights/best.pt')
Visualize: https://netron.app

```
</details>

### Evidências do treinamento

Nessa seção você deve colocar qualquer evidência do treinamento, como por exemplo gráficos de perda, performance, matriz de confusão etc.

Exemplo de adição de imagem:
![Descrição](https://picsum.photos/seed/picsum/500/300)

## Roboflow

Nessa seção deve colocar o link para acessar o dataset no Roboflow

Exemplo de link: [CAR_MOT_CAM](https://app.roboflow.com/dorpaciente/car_mot_cam/2)

## HuggingFace

Nessa seção você deve publicar o link para o HuggingFace
https://huggingface.co/spaces/Jammesson/rnn_jam/tree/main



ERRO HUGIE FACE:

Downloading:   0%|          | 0.00/8.46M [00:00<?, ?B/s]
Downloading: 100%|██████████| 8.46M/8.46M [00:00<00:00, 112MB/s]
/home/user/.local/lib/python3.8/site-packages/torch/hub.py:267: UserWarning: You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to {calling_fn}(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or load(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use load(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour
  warnings.warn(
Downloading: "https://github.com/ultralytics/yolov5/zipball/master" to /home/user/.cache/torch/hub/master.zip
YOLOv5 🚀 2022-11-8 Python-3.8.9 torch-1.13.0+cu117 CPU

Fusing layers...
Model summary: 117 layers, 4170531 parameters, 0 gradients, 10.4 GFLOPs
WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224).
/home/user/.local/lib/python3.8/site-packages/gradio/inputs.py:256: UserWarning: Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components
  warnings.warn(
/home/user/.local/lib/python3.8/site-packages/gradio/deprecation.py:40: UserWarning: `optional` parameter is deprecated, and it has no effect
  warnings.warn(value)
/home/user/.local/lib/python3.8/site-packages/gradio/outputs.py:42: UserWarning: Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components
  warnings.warn(
/home/user/.local/lib/python3.8/site-packages/gradio/interface.py:330: UserWarning: Currently, only the 'default' theme is supported.
  warnings.warn("Currently, only the 'default' theme is supported.")
Caching examples at: '/home/user/app/gradio_cached_examples/13/log.csv'
Traceback (most recent call last):
  File "app.py", line 30, in <module>
    gr.Interface(
  File "/home/user/.local/lib/python3.8/site-packages/gradio/interface.py", line 649, in __init__
    self.examples_handler = Examples(
  File "/home/user/.local/lib/python3.8/site-packages/gradio/examples.py", line 60, in create_examples
    utils.synchronize_async(examples_obj.create)
  File "/home/user/.local/lib/python3.8/site-packages/gradio/utils.py", line 359, in synchronize_async
    return fsspec.asyn.sync(fsspec.asyn.get_loop(), func, *args, **kwargs)
  File "/home/user/.local/lib/python3.8/site-packages/fsspec/asyn.py", line 96, in sync
    raise return_result
  File "/home/user/.local/lib/python3.8/site-packages/fsspec/asyn.py", line 53, in _runner
    result[0] = await coro
  File "/home/user/.local/lib/python3.8/site-packages/gradio/examples.py", line 258, in create
    await self.cache()
  File "/home/user/.local/lib/python3.8/site-packages/gradio/examples.py", line 292, in cache
    prediction = await Context.root_block.process_api(
  File "/home/user/.local/lib/python3.8/site-packages/gradio/blocks.py", line 982, in process_api
    result = await self.call_function(fn_index, inputs, iterator)
  File "/home/user/.local/lib/python3.8/site-packages/gradio/blocks.py", line 824, in call_function
    prediction = await anyio.to_thread.run_sync(
  File "/home/user/.local/lib/python3.8/site-packages/anyio/to_thread.py", line 31, in run_sync
    return await get_asynclib().run_sync_in_worker_thread(
  File "/home/user/.local/lib/python3.8/site-packages/anyio/_backends/_asyncio.py", line 937, in run_sync_in_worker_thread
    return await future
  File "/home/user/.local/lib/python3.8/site-packages/anyio/_backends/_asyncio.py", line 867, in run
    result = context.run(func, *args)
  File "app.py", line 18, in yolo
    results = model(im)  # inference
  File "/home/user/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/user/.cache/torch/hub/ultralytics_yolov5_master/models/common.py", line 508, in forward
    b, ch, h, w = im.shape  # batch, channel, height, width
  File "/home/user/.local/lib/python3.8/site-packages/PIL/Image.py", line 517, in __getattr__
    raise AttributeError(name)
AttributeError: shape
```
