

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.stageI_parsing_model import StageI_Parsing_Model
from util.visualizer import Visualizer
from util import html


opt = TestOptions().parse()  # 注意是Test的Option了  由于没有显式init，所以调用父类init

dataloader = data.create_dataloader(opt)

model = StageI_Parsing_Model(opt)
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

    generated = model(data_i, mode='inference').unsqueeze(0)
    print(generated.shape)

    img_path = data_i['img_path']  # print b: 1
    for b in range(generated.shape[0]):  # generated.shape[0] == batch size
        count = count + 1
        print(generated.shape)
        print('[%d]process image... %s' % (count, img_path[b]))
        visuals = OrderedDict([
            ('synthesized_parsing_RGB', generated[b].unsqueeze(0)),
            ('synthesized_parsing_onechannel', generated[b]),
            ('incompleted_label', data_i['incompleted_label'][b].unsqueeze(0)),
            ('original_label', data_i['label'][b].unsqueeze(0)),

            # ('original_image', data_i['original_image'][b]),
            # ('incompleted_image', data_i['incompleted_image'][b]),
            # ('mask', data_i['mask'][b]),
            # ('mask_edge', data_i['mask_edge'][b]),
            # ('mask_noise', data_i['mask_noise'][b]),
            # ('color', data_i['mask_color'][b]),

        ])



        visualizer.save_images(webpage, visuals, img_path[b:b + 1], opt.dataroot, opt.stage)

webpage.save()


