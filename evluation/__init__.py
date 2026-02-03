## evaluation/__init__.py
# 评估模块导入接口
from .pixel_based_eval import evaluate_point_level
from .eval_like_paper import eval_paper_style
# 其他评估函数根据需要导入

__all__ = [
    'evaluate_point_level',
    'eval_paper_style'
]