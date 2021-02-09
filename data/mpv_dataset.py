
import os
from data.pix2pix_dataset import Pix2pixDataset

class MPVDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='original')
        parser.set_defaults(load_size=320)  ##  not effect
        parser.set_defaults(crop_size=320)  ##  not effect
        parser.set_defaults(z_dim=320)
        parser.set_defaults(display_winsize=320)
        parser.set_defaults(label_nc=20)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        image_paths = []
        label_paths = []
        edge_paths = []
        color_paths = []
        mask_paths = []
        color_mask_paths = []


        lines = open('./datasets/mpv_train_test.txt').readlines()
        for l in lines:
            sub_path, tag = l.split()[0], l.split()[-1]
            if tag == opt.phase or (tag in ['train', 'test'] and opt.phase == 'train_test'):
                image_path = os.path.join(opt.dataroot, 'MPV_320_512_image', sub_path)
                if opt.phase == 'test' and opt.stage == 25:
                    label_path = os.path.join(opt.dataroot, 'MPV_320_512_parsing_synthesized', sub_path).replace('.jpg', '.png')

                    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch),
                    #                        'images', 'synthesized_parsing_onechannel', 'MPV_320_512_parsing_synthesized')
                    # label_path = os.path.join(web_dir, sub_path).replace('.jpg', '.png')
                else:
                    label_path = os.path.join(opt.dataroot, 'MPV_320_512_parsing', sub_path).replace('.jpg', '_gray.png')
                edge_path = os.path.join(opt.dataroot, 'MPV_320_512_edge', sub_path)
                color_path = os.path.join(opt.dataroot, 'MPV_320_512_color', sub_path)
                image_paths.append(image_path)
                label_paths.append(label_path)
                edge_paths.append(edge_path)
                color_paths.append(color_path)


        lines = open('./datasets/mask_train_test.txt').readlines()
        for l in lines:
            sub_path, tag = l.split()[0], l.split()[-1]
            if tag == opt.phase or (tag in ['train', 'test'] and opt.phase == 'train_test'):
                mask_path = os.path.join(opt.dataroot_mask, sub_path)
                mask_paths.append(mask_path)


        lines = open('./datasets/color_mask_train_test.txt').readlines()
        for l in lines:
            sub_path, tag = l.split()[0], l.split()[-1]
            if tag == opt.phase or (tag in ['train', 'test'] and opt.phase == 'train_test'):
                color_mask_path = os.path.join(opt.dataroot_color_mask, opt.phase, sub_path)
                color_mask_paths.append(color_mask_path)

        return label_paths, image_paths, edge_paths, color_paths, mask_paths, color_mask_paths
