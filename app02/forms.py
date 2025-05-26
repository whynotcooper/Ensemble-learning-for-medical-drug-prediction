from django import forms
from django.core.validators import FileExtensionValidator

class ExcelProcessingForm(forms.Form):
    excel_file = forms.FileField(
        label="选择 Excel 文件",
        validators=[FileExtensionValidator(allowed_extensions=['csv','xlsx', 'xls'])],
        help_text="支持'.csv' .xlsx 和 .xls 格式文件"
    )