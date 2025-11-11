import time

from django.shortcuts import render
from . import models
from django.http import JsonResponse, HttpResponse
from datetime import datetime
import json
import csv
import pandas as pd

from django.views.decorators.csrf import csrf_exempt

import os
import sys

# 找到 mindware 的根目录，添加到 sys.path 中
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )
    )
)

from mindware.utils.data_manager import DataManager
from mindware.modules.cash.base_cash import BaseCASH
from mindware.modules.hpo.base_hpo import BaseHPO
from mindware.modules.fe.base_fe import BaseFE
from mindware.modules.cashfe.base_cashfe import BaseCASHFE

progress = 0


# 使用 @csrf_exempt 装饰器将视图函数标记为允许 CSRF 跨站点请求。
@csrf_exempt
def submit_task(request):
    global progress
    if request.method == 'POST':
        if progress != 0:
            return JsonResponse({'status_code': 400, 'message': 'some Task is running, please wait!'})
        # 从 POST 请求中读取表单数据

        task_id = request.POST.get('taskId')
        task_type = request.POST.get('taskType')
        opt_type = request.POST.get('optType')
        optimizer = request.POST.get('optimizer')

        options = request.POST.getlist('options')

        total_iterations = int(request.POST.get('itrs'))  # 假设总共有 100 次迭代任务

        file = request.FILES.get('dataFile')

        train_node = process_data(file)
        metric = request.POST.get('metric')
        evaluation = request.POST.get('evalType')

        opt = None
        if opt_type == 'CASH':
            opt = BaseCASH(
                include_algorithms=options, sub_optimizer='tpe',
                metric=metric,
                data_node=train_node, evaluation=evaluation, resampling_params=None,
                optimizer=optimizer, per_run_time_limit=600,
                time_limit=1024, amount_of_resource=total_iterations,
                output_dir='./data', seed=1, n_jobs=1,
                ensemble_method="blending", ensemble_size=5
            )
        elif opt_type == 'CHASFE':
            opt = BaseCASHFE(
                include_algorithms=options, sub_optimizer='tpe',
                metric=metric,
                data_node=train_node, evaluation=evaluation, resampling_params=None,
                optimizer=optimizer, per_run_time_limit=600,
                time_limit=1024, amount_of_resource=total_iterations,
                output_dir='./data', seed=1, n_jobs=1,
                ensemble_method="blending", ensemble_size=5
            )
        elif opt_type == 'HPO':
            opt = BaseHPO(
                estimator_id=options[0],
                metric=metric,
                data_node=train_node, evaluation=evaluation, resampling_params=None,
                optimizer=optimizer, per_run_time_limit=600,
                time_limit=1024, amount_of_resource=total_iterations,
                output_dir='./data', seed=1, n_jobs=1,
                ensemble_method="blending", ensemble_size=5
            )
        elif opt_type == 'FE':
            opt = BaseFE(
                estimator_id=options[0],
                metric=metric,
                data_node=train_node, evaluation=evaluation, resampling_params=None,
                optimizer=optimizer, per_run_time_limit=600,
                time_limit=1024, amount_of_resource=total_iterations,
                output_dir='./data', seed=1, n_jobs=1,
                ensemble_method="blending", ensemble_size=5
            )
        else:
            return JsonResponse({'status_code': 400, 'message': 'Invalid optType'})

        start_time = time.time()
        times = []
        for i in range(total_iterations):
            # 执行迭代任务的逻辑，例如处理数据、计算等等

            if not (opt.early_stop_flag or opt.timeout_flag):
                opt.iterate()

            # 计算进度
            progress = round((i + 1) / total_iterations * 100, 2)

            times.append(time.time() - start_time)
            start_time = time.time()

        model = models.Task(
            task_id=task_id,
            task_type=task_type,
            opt_type=opt_type,
            task_start_time=datetime.fromtimestamp(opt.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'),
            task_end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            task_info=json.dumps(options),
        )

        model.save()

        # 保存每一轮的结果
        for i in range(total_iterations):
            observation = models.Observation(
                task_id="%s_%s" % (task_id, datetime.fromtimestamp(opt.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')),
                idx=i,
                configuration=json.dumps(opt.optimizer.configs[i].get_dictionary()),
                performance=opt.optimizer.perfs[i],
                time_consumed=times[i]
            )
            observation.save()

        # 返回一个简单的 JSON 响应，表示成功接收到数据
        progress = 0
        response_data = {'status_code': 200, 'task_id': task_id, 'task_type': task_type, 'optType': opt_type,
                         'options': options}
        return JsonResponse(response_data)
    else:
        # 如果不是 POST 请求，返回 405 方法不被允许
        return HttpResponse(status=405)


def show_process(request):
    # 返回一个简单的 JSON 响应，表示成功接收到数据
    response_data = {'progress': progress}
    return JsonResponse(response_data)


def process_data(csv_file):
    data = pd.read_csv(csv_file)

    x = data.iloc[:, 0:]
    y = data.iloc[:, 0].values

    dm = DataManager(x, y, feature_names=list(x.columns))

    train_data_node = dm.get_data_node(x, y)
    train_data_node = dm.preprocess_fit(train_data_node)

    return train_data_node


def get_tasks(request):
    tasks = models.Task.objects.all()
    task_list = []
    for task in tasks:
        task_list.append({
            'task_id': task.task_id,
            'task_type': task.task_type,
            'opt_type': task.opt_type,
            'task_start_time': task.task_start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'task_end_time': task.task_end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'task_info': task.task_info,
        })

    return JsonResponse({'tasks': task_list})


@csrf_exempt
def delete_task(request):
    task_id = request.POST.get('task_id')
    task_start_time = request.POST.get('task_start_time')
    task = models.Task.objects.get(task_start_time=task_start_time)
    task.delete()

    observations = models.Observation.objects.filter(task_id="%s_%s" % (task_id, task_start_time))
    for observation in observations:
        observation.delete()

    return JsonResponse({'status_code': 200, 'message': 'Task deleted successfully!'})


@csrf_exempt
def get_task_details(request):
    task_id = request.GET.get('task_id')
    task_start_time = request.GET.get('task_start_time')

    task = models.Task.objects.get(task_start_time=task_start_time)
    observations = models.Observation.objects.filter(task_id="%s_%s" % (task_id, task_start_time)).order_by('idx')

    observation_list = []
    for observation in observations:
        config_str = observation.configuration
        if len(config_str) > 35:
            config_str = config_str[1:35]
        else:
            config_str = config_str[1:-1]

        observation_list.append({
            'idx': observation.idx,
            'config_str': config_str,
            'configuration': observation.configuration,
            'performance': round(observation.performance, 5),
            'time_consumed': round(observation.time_consumed, 5),
        })

    return render(request, 'task_details.html', {
        'task_id': task_id,
        'task_start_time': task.task_start_time,
        'task_type': task.task_type,
        'opt_type': task.opt_type,
        'task_end_time': task.task_end_time,
        'task_info': task.task_info,
        'observations': observation_list,
        'performances': [[idx + 1, observation['performance']] for (idx, observation) in enumerate(observation_list)],
    })


def index(request):
    if request.method == 'GET':
        return render(request, 'main.html')


def history(request):
    if request.method == 'GET':
        return render(request, 'history.html')


def new_task(request):
    if request.method == 'GET':
        return render(request, 'new_task.html')
