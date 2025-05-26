from django.db import models


class UserInfo(models.Model):
    objects = models.Manager()  # 避免后续IDE找不到objects属性
    name = models.CharField(max_length=50)
    email = models.EmailField(max_length=50)
    password = models.CharField(max_length=50)
    age = models.IntegerField()
    prior=models.CharField(max_length=50,default='common')

class Department(models.Model):
    objects = models.Manager()  # 避免后续IDE找不到objects属性
    name = models.CharField(max_length=50)
    location = models.CharField(max_length=50)

class Employee(models.Model):
    objects = models.Manager()  # 避免后续IDE找不到objects属性
    name = models.CharField(max_length=50)
    department = models.ForeignKey(Department, on_delete=models.CASCADE)


class ProcessedFile(models.Model):
    file_name = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10)
    processed_at = models.DateTimeField(auto_now_add=True)
    row_count = models.PositiveIntegerField()
'''
create table userInfor (
    id int(11) not null auto_increment,
    name varchar(50) not null,
    email varchar(50) not null,
    password varchar(50) not null,
    primary key (id)
);
'''
from django.db import models

class ModelInfo(models.Model):
    # 模型名称
    name = models.CharField(max_length=100, verbose_name="Model Name")
    # 模型图片路径
    image_path = models.CharField(max_length=255, verbose_name="Image Path")
    # 模型性能说明
    performance = models.CharField(max_length=100, verbose_name="Performance")
    # 模型描述
    description = models.TextField(verbose_name="Description")

    def __str__(self):
        # 在 Django 管理后台显示模型名称
        return self.name

    class Meta:
        # 元数据，用于定义模型的额外信息
        verbose_name = "Model Information"
        verbose_name_plural = "Model Informations"
