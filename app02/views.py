from django.shortcuts import render, redirect
from django.http import HttpResponse

from app02.models import Department, Employee, UserInfo
import utils
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import UserInfo  # 确保你的模型导入正确


from django.shortcuts import render, redirect, HttpResponse
from django.urls import reverse
from .models import UserInfo  # 确保导入了 UserInfo 模型

def users_list(request):
    if request.method == 'GET':
        # 显示用户列表或其他内容
        return render(request, 'users_list.html')
    elif request.method == 'POST':
        name = request.POST.get('name')
        pwd = request.POST.get('pwd')
        email = request.POST.get('email')
        age = request.POST.get('age')
        prior = request.POST.get('prior')

        # 检查用户名是否已存在
        if UserInfo.objects.filter(name=name).exists():
            # 如果用户名已存在，返回错误提示
            return HttpResponse("Registration failed. Username already exists.")

        # 创建用户信息
        UserInfo.objects.create(name=name, password=pwd, email=email, age=age, prior=prior)

        # 登录逻辑（这里假设用户名和密码匹配即可登录）
        user = UserInfo.objects.filter(name=name, password=pwd).first()
        if user:
            # 如果用户存在且密码匹配，跳转到 show_image 页面
            return redirect('show_image')  # 假设你已经在 urls.py 中定义了 show_image 的 URL 名称
        else:
            # 如果登录失败，返回错误信息或重定向到登录页面
            return HttpResponse("Login failed. Please check your credentials.")

def orm(request):
    Department.objects.create(name='Finance', location='Chicago')
    UserInfo.objects.create(name='John', email='john@example.com', password='password', age=30, prior='common')
    UserInfo.objects.create(name='Mary', email='mary@example.com', password='password', age=25, prior='common')
    return HttpResponse("This is the ORM page.")
    # We will use the utils.py file to fetch data from the database

def info_list(request):
    Info_list = UserInfo.objects.all()
    # This is the new way of fetching data from the database
    # return HttpResponse("This is the info list page.")
    # This is the old way of rendering a template

    return render(request, 'info_list.html', {'Info_list': Info_list})

def login(request):
    return render(request, 'login.html')


Info_list = UserInfo.objects.all()
# This is the new way of rendering a templat
def info_delete(request):
    nid=request.GET.get('nid')
    UserInfo.objects.filter(id=nid).delete()
    return redirect('/info_list')

for info in Info_list:
    print(info.name)
    print(info.email)
    print(info.password)
    print(info.age)
    print(info.prior)
def process_excel_data(df):
    # 调用本地服务器算法处理数据
    # 假设算法返回处理结果
    return "Processed data"
'''
def model_pf(request):
    return render(request, 'model_pf.html')
'''
import os
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd

def model_pf(request):
    if request.method == 'POST':
        # 获取上传的文件
        uploaded_file = request.FILES.get('excel_file')
        if not uploaded_file:
            return HttpResponse("No file uploaded.", status=400)

        # 保存上传的文件到临时目录
        temp_file_path = 'temp_excel_file.xlsx'
        with open(temp_file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        print(f"File saved to {temp_file_path}")
        # 使用 pandas 读取 Excel 文件
        try:
            df = pd.read_excel(temp_file_path)
        except Exception as e:
            return HttpResponse(f"Error reading Excel file: {e}", status=400)

        # 调用本地服务器算法处理数据
        result = process_excel_data(df)

        # 删除临时文件
        os.remove(temp_file_path)

        # 返回处理结果
        return HttpResponse(f"Processing result: {result}")

    return render(request, 'model_pf.html')


import os
import uuid
import tempfile
import shutil
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from .forms import ExcelProcessingForm
from .tasks import process_file_task
from .utils import process_excel_data, process_excel_data2
import logging

logger = logging.getLogger(__name__)


def upload_excel(request):
    if request.method == 'POST':
        form = ExcelProcessingForm(request.POST, request.FILES)

        if form.is_valid():
            uploaded_file = form.cleaned_data['excel_file']
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()

            try:
                # 创建安全临时目录
                temp_dir = tempfile.mkdtemp(dir=settings.TEMP_UPLOAD_DIR)
                safe_filename = f"{uuid.uuid4().hex}{file_ext}"
                temp_path = os.path.join(temp_dir, safe_filename)

                # 分块写入文件
                with open(temp_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                # 同步处理数据（快速返回结果）
                processed_data = process_excel_data(temp_path)

                # 清理临时文件
                os.unlink(temp_path)
                shutil.rmtree(temp_dir)

                # 直接返回处理结果
                return render(request, 'result.html', {'result': processed_data})

            except Exception as e:
                logger.error(f"处理失败: {str(e)}")
                return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse({'errors': form.errors}, status=400)

    return render(request, 'upload.html', {'form': ExcelProcessingForm()})
from django.http import FileResponse

from django.http import FileResponse, JsonResponse
from django.shortcuts import render
from .forms import ExcelProcessingForm
from .utils import process_excel_data
import os
import uuid
import tempfile
import shutil
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

from zipfile import ZipFile
from django.http import FileResponse
import os
import shutil
import tempfile
from django.conf import settings
import uuid

def upload_excel2(request):
    if request.method == 'POST':
        form = ExcelProcessingForm(request.POST, request.FILES)

        if form.is_valid():
            uploaded_file = form.cleaned_data['excel_file']
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()

            try:
                # 创建安全临时目录
                temp_dir = tempfile.mkdtemp(dir=settings.TEMP_UPLOAD_DIR)
                safe_filename = f"{uuid.uuid4().hex}{file_ext}"
                temp_path = os.path.join(temp_dir, safe_filename)
                print(temp_path)
                # 分块写入文件
                with open(temp_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                # 同步处理数据，生成结果文件
                result_file_path, result_indices_path, png_path = process_excel_data2(temp_path)

                # 创建ZIP文件
                zip_temp_dir = tempfile.mkdtemp(dir=settings.TEMP_UPLOAD_DIR)
                zip_filename = f"{uuid.uuid4().hex}.zip"
                zip_path = os.path.join(zip_temp_dir, zip_filename)

                with ZipFile(zip_path, 'w') as zipf:
                    zipf.write(result_file_path, os.path.basename(result_file_path))
                    zipf.write(result_indices_path, os.path.basename(result_indices_path))
                    zipf.write(png_path, os.path.basename(png_path))

                # 清理原始上传文件和结果文件
                os.unlink(temp_path)
                os.unlink(result_file_path)
                os.unlink(result_indices_path)
                os.unlink(png_path)

                # 提供ZIP文件下载
                response = FileResponse(open(zip_path, 'rb'), as_attachment=True, filename="results.zip")

                return response

            except Exception as e:
                logger.error(f"处理失败: {str(e)}")
                return JsonResponse({'error': str(e)}, status=500)

            finally:
                # 清理临时目录
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(zip_temp_dir, ignore_errors=True)

        return JsonResponse({'errors': form.errors}, status=400)

    return render(request, 'upload.html', {'form': ExcelProcessingForm()})


from django.shortcuts import render, redirect
from django.http import HttpResponse

# 假设你有一个模型类 ModelInfo 来存储模型的信息
from .models import ModelInfo


def show_image(request):
    # 从数据库中获取所有模型的信息
    models = ModelInfo.objects.all()

    # 如果没有数据库模型，可以直接定义一个上下文字典
    models = [
         {
            'id': 1,
            'name': 'Model 1',
            'image_path': '/static/model1.png',
            'performance': '95% Accuracy',
             'description': 'This model is designed for image classification tasks.'
         },
    {
            'id': 2,
            'name': 'Model 2',
            'image_path': '/static/model2.png',
            'performance': '92% Accuracy',
             'description': 'This model is optimized for natural language processing tasks.'
        },
        {
            'id': 3,
             'name': 'Model 3',
             'image_path': '/static/model3.png',
            'performance': '90% Accuracy',
            'description': 'This model is used for time series forecasting.'
        }
    ]

    context = {
        'models': models
    }
    return render(request, 'show_image.html', context)


def use_model(request, model_id):
    # 根据 model_id 获取模型信息
    model = ModelInfo.objects.get(id=model_id)

    context = {
        'model': model
    }
    return render(request, 'use_model.html', context)


from django.urls import reverse
from .models import UserInfo  # 确保导入了 UserInfo 模型

def login(request):
    error_message = ""  # 初始化错误信息
    if request.method == 'POST':
        name = request.POST.get('name')
        pwd = request.POST.get('pwd')

        # 检查用户名和密码是否匹配
        user = UserInfo.objects.filter(name=name, password=pwd).first()
        if user:
            # 如果用户存在且密码匹配，跳转到 show_image 页面
            return redirect('show_image')  # 假设你已经在 urls.py 中定义了 show_image 的 URL 名称
        else:
            # 如果登录失败，设置错误信息
            error_message = "Login failed. Please check your credentials."

    # 如果是 GET 请求或登录失败，显示登录表单
    return render(request, 'login.html', {'error_message': error_message})