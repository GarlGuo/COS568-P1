import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity, schedule


def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10, use_amp=False):
    model.train()

    scaler = GradScaler(enabled=use_amp)  # Only scales if use_amp=True
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        #output = model(data)
        #train_loss = loss(output, target)
        #with autocast(enabled=use_amp):
        with autocast(device_type='cuda', enabled=use_amp):
            output = model(data)
            train_loss = loss(output, target)
        
        total += train_loss.item() * data.size(0)
        #train_loss.backward()
        #optimizer.step()
        
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update() 
        
        
        
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

@torch.inference_mode()
def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    inference_time = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        output = model(data)
         
        end_event.record()
        # cannot be deleted for correct timing!!!
        torch.cuda.synchronize()
        
        inference_time += start_event.elapsed_time(end_event) / 1e3   # milliseconds
        
        total += loss(output, target).item() * data.size(0)
        _, pred = output.topk(5, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct1 += correct[:,:1].sum().item()
        correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5, inference_time


@torch.inference_mode()
def benchmark(model, dataloader, device, step=1, warmup=5):
    model.eval()
    trace_addr = f"trace.json"

    sched = schedule(warmup=warmup, active=step, wait=0)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=lambda p: p.export_chrome_trace(trace_addr),
    ) as prof:
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            model(data)
            prof.step()
            torch.cuda.synchronize()

    print(f'trace file saved at {trace_addr}')


def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, use_amp=False):
    test_loss, accuracy1, accuracy5, inference_time = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5, inference_time]]
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose, use_amp=use_amp)
        test_loss, accuracy1, accuracy5, inference_time = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, accuracy1, accuracy5, inference_time]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy', 'inference_time']
    return pd.DataFrame(rows, columns=columns)


