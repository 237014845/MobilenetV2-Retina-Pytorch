from .xml_style import XMLDataset
# from .voc import VOCDataset


class SARDataset(XMLDataset):

    CLASSES = ('ship',)

    def __init__(self, **kwargs):
        super(SARDataset, self).__init__(**kwargs)
        self.abc = 1