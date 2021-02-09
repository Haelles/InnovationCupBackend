

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch
from util.util import label_2_onehot_batch

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
# iter_counter = IterationCounter(opt, len(dataloader))
iter_counter = IterationCounter(opt, len(dataloader.dataset))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        if not opt.no_GAN:
            trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            origin_incompleted_synthesized = torch.cat([data_i['incompleted_image'], trainer.get_latest_generated().cpu(), data_i['original_image']], dim=3)
            visuals = OrderedDict([
                                   # ('input_label', data_i['label']),
                                   # ('synthesized_image', trainer.get_latest_generated()),
                                   # ('original_image', data_i['original_image']),
                                   # ('incompleted_image', data_i['incompleted_image']),
                                   ('origin_incompleted_synthesized', origin_incompleted_synthesized),
                                   # ('mask', data_i['mask']),
                                   # ('mask_edge', data_i['mask_edge']),
                                   # ('mask_noise', data_i['mask_noise']),
                                   # ('color', data_i['mask_color']),

                                   # ('face_RGB', data_i['face_RGB']),
                                   # ('m_onehot_RGB', data_i['mask_onehot_RGB']),
                                   # ('part_RGB', data_i['part_RGB'])
                                   ])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
