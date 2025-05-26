import os
from pathlib import Path

# 项目路径
BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)
print(os.path.join(BASE_DIR, 'app02','templates'))
# 项目秘钥
SECRET_KEY = 'django-insecure-4pa1wnzjm%y*o-3q(9pvs9hx((1h6pydr77p(3au2u55a$wm*j'

# 调试模式
DEBUG = True

# 允许的主机
ALLOWED_HOSTS = []

# 注册app
INSTALLED_APPS = [
    'simpleui',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app02.apps.App02Config',
]

# 中间件
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# 根路由
ROOT_URLCONF = 'My_project.urls'

# 前端模板文件
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'app02','templates')],
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
            'builtins': [
                # 在模板中使用static标签
                'django.templatetags.static'
            ],
        },
    },
]

# 数据库配置
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'moadata',  # 数据库名称
        'USER': 'root',  # 账号
        'PASSWORD': 'Why200410why!!!',  # 密码
        'HOST': 'localhost',
        'PORT': '3306'  # 端口
    }
}

# 密码验证
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# 国际化配置
LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'app02'/'static',
]
# 主键自增
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# settings.py
import os
from pathlib import Path

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 添加临时文件目录配置
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, 'temp_uploads')
# 在 settings.py 末尾添加
if not os.path.exists(TEMP_UPLOAD_DIR):
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
# 隐藏右侧SimpleUI广告链接和使用分析
SIMPLEUI_HOME_INFO = False
SIMPLEUI_ANALYSIS = False
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
