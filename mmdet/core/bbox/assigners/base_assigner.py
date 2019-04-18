from abc import ABCMeta, abstractmethod

# 定义一个抽象基类的方法是将一个类的元类设置为abc.ABCMeta
class BaseAssigner(metaclass=ABCMeta):
    # 用@abstractmethod声明一个基类中的函数使虚函数。除了该装饰器外，还有@abstractproperty声明一个抽象属性。
    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        pass
