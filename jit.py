import yaml
from backbone import EfficientDetBackbone
import torch

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


params = Params('projects/coco.yml')
model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=0,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
_ = model.load_state_dict(torch.load("./weights/efficientdet-d0.pth"), strict=False)
model.eval()
_ = model(torch.Tensor(1,3,512,512))

trace_model = torch.jit.trace(model, (torch.Tensor(1,3,512,512), ))
# trace_model.save('./weights/efficientdet-d0_jit.pt')
trace_model.save('./weights/efficientdet.pt')
