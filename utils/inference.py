import threading
import time

import nvsmi
import torch
from tqdm import tqdm


def calc_speed(model, inputs, number_speed_trials):
    """returns inference speed (ms) of the model."""

    # Just ensure model is in evaluation mode (user may not be aware)
    model.eval()

    # Warm-up
    for _ in range(10):
        _ = model(inputs)
    torch.cuda.synchronize()

    timing = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in tqdm(range(number_speed_trials)):
            start.record()
            _ = model(inputs)
            end.record()
            torch.cuda.synchronize()
            timing.append(start.elapsed_time(end))

    accumulated_time = sum(timing)

    return accumulated_time / number_speed_trials


def calc_acc(model, dataloader):
    hit = 0
    for i, (inputs, labels) in enumerate(tqdm(dataloader)):
        outputs = model(inputs.cuda()).cpu()
        preds = torch.argmax(outputs, axis=1).cpu()
        hit += torch.sum(labels == preds).item()

    acc = hit / (i + 1) / dataloader.batch_size
    return acc


def get_memory(mem_list, gpu_id=0):
    time.sleep(1)

    for i in range(10):
        mem = list(nvsmi.get_gpus())[gpu_id]
        mem_used = mem.mem_total - mem.mem_free
        time.sleep(0.5)

        if mem_used > 1e3:
            mem_list.append(mem_used)


def calc_memory(model, inputs, num_samples=500, gpu_id=0):

    mem_list = []
    t1 = threading.Thread(target=calc_speed, args=(model, inputs, num_samples))
    t2 = threading.Thread(target=get_memory, args=(mem_list, gpu_id))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    mem_used = torch.FloatTensor(mem_list).mean().item()
    return mem_used
