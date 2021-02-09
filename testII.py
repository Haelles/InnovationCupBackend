

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.stageII_multiatt3_model import StageII_MultiAtt3_Model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = StageII_MultiAtt3_Model(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
if opt.test_color:
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_color' % (opt.phase, opt.which_epoch))
else:
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
count = 0
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['img_path']
    for b in range(generated.shape[0]):
        count = count + 1
        print('[%d]process image... %s' % (count, img_path[b]))
        # incompleted_synthesized_origin = torch.cat([data_i['color'][b], data_i['mask_color'][b], data_i['incompleted_image'][b], generated[b].cpu(), data_i['original_image'][b]], dim=2)
        # incompleted_synthesized_origin = torch.cat([data_i['mask_color'][b], data_i['incompleted_image'][b], generated[b]], dim=2)
        # visuals = OrderedDict([('incompleted_synthesized_origin', incompleted_synthesized_origin)])

        mask = data_i['mask_onehot'].cpu()
        synthesized_image = generated.cpu() * mask + data_i['original_image'].cpu() * (1 - mask)
        visuals = OrderedDict([
            ('original_image', data_i['original_image'][b]),
            ('incompleted_image', data_i['incompleted_image'][b]),
            ('incompleted_image2', data_i['incompleted_image_2'][b]),
            # ('stageI_label', data_i['label'][b].unsqueeze(0)),
            ('synthesized_image', synthesized_image[b]),
            # ('mask', data_i['mask'][b]),
            # ('composed_mask', data_i['part_RGB'][b]),
            # ('mask_edge', data_i['mask_edge'][b]),
            # ('mask_noise', data_i['mask_noise'][b]),
            # ('color', data_i['mask_color'][b]),
        ])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1], opt.dataroot, opt.stage)

webpage.save()
