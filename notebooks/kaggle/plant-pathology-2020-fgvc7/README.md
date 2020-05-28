00_starter_code:
model_name = 'efficientnet-b0'
hard_transforms отключен
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
auc/_mean: 0.9508, epoch=13/2


Задача 1:
Для модели efficientnet при отключенных аугментациях
замерить максимальный auc/_mean

00_efficientnet-b1:
model_name = 'efficientnet-b1'
hard_transforms отключен
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
auc/_mean: 0.9510, epoch=12/2


00_efficientnet-b2:
model_name = 'efficientnet-b3'
hard_transforms отключен
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
auc/_mean: 0.9330, epoch=9/2


00_efficientnet-b3:
model_name = 'efficientnet-b3'
hard_transforms отключен
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
auc/_mean: 0.9513, epoch=30/2

00_efficientnet-b4:
model_name = 'efficientnet-b4'
hard_transforms отключен
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
auc/_mean: 0.9474, epoch=7/2
auc/_mean: 0.9300, epoch=8/2 ранняя остановка по loss
модели начинают сильно переобучаться





Задача 1:
Повторить задачу 0 при включенных аугментациях
замерить максимальный auc/_mean


01_starter_code:
model_name = 'efficientnet-b0'
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
TTTA = 5
auc/_mean: 0.9221, epoch=39/2
LB 733, auc/_mean: 0.947


01_efficientnet-b4:
model_name = 'efficientnet-b4'
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
TTA = 11
0.977
auc/_mean: 0.977, epoch=32/2
LB 462, auc/_mean: 0.964
TTA = 5
LB 446, auc/_mean: 0.965

TODO можно увеличить размер батча


01_efficientnet-b5:
model_name = 'efficientnet-b5'
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
auc/_mean: 0.929, epoch=32/2
LB auc/_mean: 0.961


Задача 2:
Повторить эксперимент 1 с модифицированными LabelSmoothingCrossEntropy и планеровщиком
Повторить задачу 0 при включенных аугментациях
замерить максимальный auc/_mean


02_starter_code:
model_name = 'efficientnet-b0'
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
TTA = 5
tts=0.2
auc_mean=0.976, epoch=125/2
LB 454 -> X, auc_mean=0.945


02_efficientnet-b4:
model_name = 'efficientnet-b4'
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
TTA = 11
tts=0.2
auc_mean=0.987, epoch=128/2
LB 454 -> X, auc_mean=0.951


02_efficientnet-b0_tts=03
tts=0.33
LB auc_mean=0.933


Задача 3:
Сделать крос валидачию 5-fold


03_starter_code:
Разбить данные на стратифицированные 5-fold
model_name = 'efficientnet-b0'
LabelSmoothingCrossEntropy
обучение классификатора до минимального лоса,
дообучение всей модели до минимального АУК
TTTA = 5
auc/_mean: 0.743, epoch=39/2













from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler

optimizer = SGD(
    [
        {"params": model.base.parameters(), 'lr': 0.001),
        {"params": model.fc.parameters(), 'lr': 0.01),
    ]
)

scheduler1 = LinearCyclicalScheduler(optimizer, 'lr', 1e-7, 1e-5, len(train_loader), param_group_index=0)
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler1, "lr (base)")

scheduler2 = CosineAnnealingScheduler(optimizer, 'lr', 1e-5, 1e-3, len(train_loader), param_group_index=1)
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler2, "lr (fc)")








def activated_output_transform(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y

roc_auc = ROC_AUC(activated_output_transform)






def comp_metric(preds, targs, labels=range(len(LABEL_COLS))):
    # One-hot encode targets
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_roc_auc(*args):
    return comp_metric(*args, labels=[0])

def multiple_diseases_roc_auc(*args):
    return comp_metric(*args, labels=[1])

def rust_roc_auc(*args):
    return comp_metric(*args, labels=[2])

def scab_roc_auc(*args):
    return comp_metric(*args, labels=[3])






























