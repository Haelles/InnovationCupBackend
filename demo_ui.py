import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
import cv2
import numpy as np
import time
from collections import OrderedDict
from util.util import *
import data
from options.test_options import TestOptions
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from imageio import imread
from models.stageI_parsing_model import StageI_Parsing_Model
from models.stageII_multiatt3_model import StageII_MultiAtt3_Model

from skimage.feature import canny
from skimage.color import rgb2gray
# from scipy.misc import imread, imresize,imsave
from imageio import imread
global file
RED = [0, 0, 255]
rect = (0, 0, 1, 1)
drawing = False
rectangle = False
rect_over = False
rect_or_mask = 100
thickness = 2
posx = None
posy = None
extraedge = False


class Main(QWidget):
    def __init__(self):
        super().__init__()

        self.widget = ImageWithMouseControl(self)
        self.widget.setGeometry(0, 0, 320, 512)

        self.setWindowTitle('Image with mouse control')
        palette = QPalette()  # 调色板
        #print(file + "qaq")
        palette.setBrush(QPalette.Background, QBrush(QPixmap('./check/show_incom.png')))
        self.setPalette(palette)

    def closeEvent(self, e):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No);
        if reply == QMessageBox.Yes:
            e.accept()

        else:
            e.ignore()


class ImageWithMouseControl(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.img = QPixmap('./check/show_ROI.png')
        self.a = None
        self.point = QPoint(0, 0)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image with mouse control')

    def paintEvent(self, e):

        painter = QPainter()
        painter.begin(self)
        self.draw_img(painter)
        painter.end()

    def draw_img(self, painter):
        painter.drawPixmap(self.point, self.img)

    def mouseMoveEvent(self, e):
        if self.left_click:
            self._endPos = e.pos() - self._startPos
            self.point = self.point + self._endPos
            self._startPos = e.pos()
            self.repaint()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self._startPos = e.pos()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            # print(e.pos())
            global posx,posy,extraedge
            self.a = self.point.x()
            self.b = self.point.y()
            posx = self.a
            posy = self.b
            print(posy)
            extraedge = True
            print("mouse release")
            # print(pos)
            # print(self.a)

            self.left_click = False

        elif e.button() == Qt.RightButton:

            self.point = QPoint(0, 0)
            self.scaled_img = self.img.scaled(self.size())
            self.repaint()


class Ex(QWidget, Ui_Form):  # python支持多重继承
    def __init__(self, model1, opt, model2, opt2):
        super().__init__()
        self.setupUi(self)  # 参数Form为Ex对象自己  ui.py
        self.show()
        # self.model1 = model1
        # self.config1 = opt
        # self.model2 = model2
        # self.config2 = opt2
        self.output_img = None
        self.edge_change = None
        self.sub_floder = None
        self.mat_img = None
        self.what_img = None
        self.ld_mask = None
        self.img2 = None
        self.ld_sk = None
        self.mask_3 = None
        self.edge_numpy = np.zeros((512, 320, 3))
        self.modes = [0, 0, 0, 0, 0]
        self.mouse_clicked = False
        self.dst = None
        self.rectangle = None
        self.rect = None
        self.rect_or_mask = None
        self.ix = -1
        self.iy = -1
        self.rect_over = None
        self.drawinng = None
        self.scene = GraphicsScene(self.modes)  # mouse_event.py中的类
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scene_stroke = GraphicsScene(self.modes)  # mouse_event.py中的类
        self.graphicsView_stroke.setScene(self.scene_stroke)  # 在ui.py中定义
        self.graphicsView_stroke.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_stroke.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_stroke.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        '''self.scene_sketch = GraphicsScene(self.modes)
        self.graphicsView_sketch.setScene(self.scene_sketch)
        self.graphicsView_sketch.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_sketch.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_sketch.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)'''

        self.result_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.result_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

    def mode_select(self, mode):
        for i in range(len(self.modes)):
            self.modes[i] = 0
        self.modes[mode] = 1

    def choosemodel(self):
        return




    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath())
        if fileName:
            global file
            image = QPixmap(fileName)
            file = fileName
            print(fileName)
            mat_img = cv2.imread(fileName)
            index = -fileName[::-1].find('/')
            subfold = fileName.split('/')[-2]
            self.realname = subfold + "=" + fileName[index:-4]
            print(self.realname)
            #try :
                #self.edge_numpy = cv2.imread(fileName[:index]+"edge/"+fileName[index:])
                #self.parsing_numpy = cv2.imread(fileName[:index]+"edge/"+fileName[index:])
                #self.edge_image = Image.open(fileName[:index]+"edge/"+fileName[index:])
            #except:
                #print("can't find picture!")
            self.what_img = imread(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return

            # redbrush = QBrush(Qt.red)
            # blackpen = QPen(Qt.black)
            # blackpen.setWidth(5)
            self.image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            #mat_img = cv2.resize(mat_img, (320, 512), interpolation=cv2.INTER_CUBIC)
            self.mat_img = mat_img
            #print(mat_img.shape)
            #self.mat_img = np.expand_dims(mat_img, axis=0)
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items()
            self.scene.addPixmap(self.image)

            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(self.image)

    def mask_mode(self):
        self.mode_select(0)

    def sketch_more_mode(self):
        self.mode_select(3)

    def sketch_mode(self):
        self.mode_select(1)

    def stroke_mode(self):
        if not extraedge:
            if not self.color:
                self.color_change_mode()
            self.scene.get_stk_color(self.color)
        else:
            if not self.color:
                self.color_change_mode()
            self.scene_stroke.get_stk_color(self.color)
        self.mode_select(2)

    def stroke_more_mode(self):
        if not extraedge:
            if not self.color:
                self.color_change_mode()
            self.scene.get_stk_color(self.color)
        else:
            if not self.color:
                self.color_change_mode()
            self.scene_stroke.get_stk_color(self.color)
        self.mode_select(4)

    def color_change_mode(self):
        self.dlg.exec_()  # QColorDialog(self.graphicsView)
        self.color = self.dlg.currentColor().name()
        self.pushButton_4.setStyleSheet("background-color: %s;" % self.color)  # stroke mode
        if not extraedge:
            self.scene.get_stk_color(self.color)
        else:
            self.scene_stroke.get_stk_color(self.color)

    def complete(self):
        a = "%06d" % random.randint(0, 10000)

        self.realname = self.realname + '=' + a
        print(self.realname)
        global extraedge
        print(extraedge)
        if extraedge:
            mask = imread('./model_input/' + self.realname[:-7] + '_mask.png')
            erase = self.make_mask(self.scene_stroke.mask_points)
            cv2.imwrite('./model_input/' + self.realname + '_mask_more.png', erase)
            mask_final = erase + mask
            cv2.imwrite('./model_input/' + self.realname + '_mask_final.png', mask_final)
            sketch = self.make_sketch(self.scene_stroke.sketch_points)
            stroke = self.make_stroke(self.scene_stroke.stroke_points)
            stroke_more = self.stroke_more(self.scene_stroke.stroke_more_points)
            sketch_more = self.sketch_more(self.scene_stroke.sketch_more_points)
            mask_input = (mask_final == 0).astype(np.uint8)*255
            mask_1 = rgb2gray(mask_input)
            mask_input1 = np.expand_dims(mask_1, axis=2)
            erase_input = (erase == 0).astype(np.uint8)*255
            erase_1 = rgb2gray(erase_input)
            erase_input_1 = np.expand_dims(erase_1, axis=2)
            noise = self.make_noise()
            noise = noise * mask_final
        else:
            sketch = self.make_sketch(self.scene.sketch_points)
            stroke = self.make_stroke(self.scene.stroke_points)
            mask = self.make_mask(self.scene.mask_points)
            print(mask.shape)
            cv2.imwrite('./model_input/' + self.realname + '_mask_final.png', mask)
            noise = self.make_noise()
            print(noise.shape)
            noise = noise * mask
            mask_input1 = mask

        transform_list = []
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_image = transforms.Compose(transform_list)
        sketch = sketch
        if extraedge:

            mask_3 = np.asarray(mask_final[:, :, 0] / 255, dtype=np.uint8)
            mask_3 = np.expand_dims(mask_3, axis=2)

            stroke = stroke * mask_3 + stroke_more
        else:
            stroke = stroke*self.mask_3

        cv2.imwrite("./model_input/" + self.realname + "_noise.png", noise)
        cv2.imwrite("./model_input/" + self.realname + "_sketch.png", sketch)
        cv2.imwrite("./model_input/" + self.realname + "_stroke.png", stroke)
        img_path = file
        parsing_path = img_path.replace('image', 'parsing').replace('.jpg', '_gray.png')
        print(parsing_path, "*" * 10)
        label = Image.open(parsing_path)
        # label = Image.open("./model_input/parsing.png")
        params = get_params(self.config1, label.size)
        transform_label = get_transform(self.config1, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0



        if extraedge:
            incompleted_image = self.mat_img*mask_input1
        else :
            incompleted_image = self.mat_img*(1-mask)
        cv2.imwrite("./model_input/"+ self.realname +"_incom.png",incompleted_image)
        mask_onehot = imread("./model_input/"+ self.realname +"_mask_final.png")
        mask_onehot = rgb2gray(mask_onehot)
        mask_onehot = (mask_onehot > 0).astype(np.uint8)
        mask_arr = mask_onehot * 255
        mask_2 = Image.fromarray(mask_arr).convert('RGB')
        mask_tensor = transform_image(mask_2)
        #original_label = imread("./model_input/parsing.png")
        original_label = imread(parsing_path)
        incompleted_label = original_label * (1 - mask_onehot)
        incompleted_label = Image.fromarray(incompleted_label)
        incompleted_label_tensor = transform_label(incompleted_label) * 255.0

        # incompleted image

        incompleted_image_1 = imread("./model_input/"+ self.realname +"_incom.png")
        print(incompleted_image_1.shape)
        incompleted_image_1 = Image.fromarray(incompleted_image_1)
        incompleted_image_tensor = transform_image(incompleted_image_1)
        print(incompleted_image_tensor.size())
        image = Image.fromarray(self.what_img)
        image_tensor = transform_image(image)


        #img_tensor = img_tensor.permute(0,1,3,2)
        noise_tensor = torch.randn_like(image_tensor)
        mask_noise_tensor = noise_tensor * (1 + mask_tensor) / 2.0 - (1 - mask_tensor) / 2.0
        if extraedge:
            edge_masked = imread("./check/out_edge.png")
            more_edge_numpy = cv2.imread("./model_input/"+ self.realname +"_sketch.png")
            edge_final = edge_masked + more_edge_numpy
            edge_final = edge_final*erase_input_1 + sketch_more
            cv2.imwrite("./model_input/"+ self.realname +"_edge_masked.png", edge_final)
            edge_masked_final = imread("./model_input/"+ self.realname +"_edge_masked.png")

        else:

            more_edge_numpy = cv2.imread("./model_input/"+ self.realname +"_sketch.png")
            edge_final = more_edge_numpy
            cv2.imwrite("./model_input/"+ self.realname +"_edge_masked.png",edge_final)
            edge_masked_final = imread("./model_input/"+ self.realname +"_edge_masked.png")
        edge_masked_final_1 = Image.fromarray(edge_masked_final).convert('RGB')
        mask_edge_tensor =transform_image(edge_masked_final_1)

        color = imread("./model_input/"+ self.realname +"_stroke.png")
        color_1 = Image.fromarray(color).convert('RGB')
        color_tensor = transform_image(color_1)





        # incompleted label
        bg_mask_numpy = (original_label == 0).astype(np.uint8)
        part_mask_numpy = (mask_onehot * (1 - bg_mask_numpy) + bg_mask_numpy).astype(np.uint8)
        part_mask_numpy = 1 - part_mask_numpy
        part_mask = Image.fromarray(part_mask_numpy)
        part_mask_tensor = transform_label(part_mask) * 255

        face_mask_numpy = (original_label == 13).astype(np.uint8)
        face_mask_img = Image.fromarray(face_mask_numpy)
        face_mask_tensor = transform_label(face_mask_img) * 255
        mask_onehot_img = Image.fromarray(mask_onehot)
        mask_onehot_tensor = transform_label(mask_onehot_img) * 255

        # face_mask = Image.fromarray(face_mask_numpy * 255).convert('RGB')
        # mask_onehot = Image.fromarray(mask_onehot * 255).convert('RGB')
        part_mask = Image.fromarray(part_mask_numpy * 255).convert('RGB')
        # face_RGB = transform_image(face_mask)
        # mask_onehot_RGB = transform_image(mask_onehot)
        part_RGB = transform_image(part_mask)

        label_tensor = label_tensor.unsqueeze(0)
        incompleted_label_tensor = incompleted_label_tensor.unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)
        incompleted_image_tensor = incompleted_image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        mask_edge_tensor = mask_edge_tensor.unsqueeze(0)
        mask_noise_tensor = mask_noise_tensor.unsqueeze(0)
        color_tensor = color_tensor.unsqueeze(0)
        face_mask_tensor = face_mask_tensor.unsqueeze(0)
        mask_onehot_tensor = mask_onehot_tensor.unsqueeze(0)
        part_mask_tensor = part_mask_tensor.unsqueeze(0)
        part_RGB = part_RGB.unsqueeze(0)


        data_1 = {
            'label': label_tensor,
            'incompleted_label': incompleted_label_tensor,
            'original_image': image_tensor,
            'incompleted_image': incompleted_image_tensor,
            'mask': mask_tensor[:,0:1,:,:],
            'mask_edge': mask_edge_tensor[:,0:1,:,:],
            'mask_noise': mask_noise_tensor[:,0:1,:,:],
            'mask_color': color_tensor,
            'part_mask': part_mask_tensor,
            'face_mask': face_mask_tensor,
            'mask_onehot': mask_onehot_tensor,
            'part_RGB': part_RGB,
        }

        start_t = time.time()

        result = self.model1(data_1, mode='inference')




        end_t = time.time()

        print(type(result))
        print('model1_inference time : {}'.format(end_t - start_t))




        result1 = parsing2im_batch_by20chnl(result,0)
        second_input = parsing_2_onechannel(result[0])

        cv2.imwrite("./model_input/"+self.realname+"_model1_output.png",second_input)

        label1 = Image.open("./model_input/"+self.realname+"_model1_output.png")
        params = get_params(self.config2, label1.size)
        transform_label = get_transform(self.config2, params, method=Image.NEAREST, normalize=False)
        label_tensor1 = transform_label(label1) * 255.0

        original_label1 = imread("./model_input/"+self.realname+"_model1_output.png")
        incompleted_label1 = original_label1 * (1 - mask_onehot)
        incompleted_label1 = Image.fromarray(incompleted_label1)
        incompleted_label_tensor1 = transform_label(incompleted_label1) * 255.0

        label_tensor1 = label_tensor1.unsqueeze(0)
        incompleted_label_tensor1 = incompleted_label_tensor1.unsqueeze(0)
        data_2 = {
            'label': label_tensor1,
            'incompleted_label': incompleted_label_tensor1,
            'original_image': image_tensor,
            'incompleted_image': incompleted_image_tensor,
            'mask': mask_tensor[:, 0:1, :, :],
            'mask_edge': mask_edge_tensor[:, 0:1, :, :],
            'mask_noise': mask_noise_tensor[:, 0:1, :, :],
            'mask_color': color_tensor,
            'part_mask': part_mask_tensor,
            'face_mask': face_mask_tensor,
            'mask_onehot': mask_onehot_tensor,
            'part_RGB': part_RGB,
        }

        self.output_img = result1

        start_t1 = time.time()
        result2 = self.model2(data_2, mode='inference')
        end_t1 = time.time()
        print('model1_inference time : {}'.format(end_t1 - start_t1))

        result_final = tensor2im(result2[0])
        print(result_final.shape)

        result_final = np.concatenate([result_final[:, :, :1], result_final[:, :, 1:2], result_final[:, :, 2:3]], axis=2)
        result_show_1 = np.concatenate([result1[:, :, 2:3], result1[:, :, 1:2], result1[:, :, :1]], axis=2)
        result_show = np.concatenate([result_final[:, :, 2:3], result_final[:, :, 1:2], result_final[:, :, :1]], axis=2)
        cv2.imwrite("./result/"+self.realname+"result1.png", result_show_1)
        cv2.imwrite("./result/"+self.realname+"result2.png", result_show)
        f = (edge_masked_final>100).astype(np.uint8)*255.0
        g = (mask_arr > 240).astype(np.uint8)*255.0
        g = np.expand_dims(g,axis=2)
        e = rgb2gray(color)
        e = np.expand_dims(e,axis=2)
        e = (e >0).astype(np.uint8)*255.0
        q = np.zeros((512,320,3))
        image_mid = q + g - e + color + incompleted_image_1 -f
        image_mid = np.concatenate((image_mid[:,:,2:3],image_mid[:,:,1:2],image_mid[:,:,:1]),axis=2)
        cv2.imwrite("./model_input/"+self.realname+"_image_mid.png",image_mid)

        first = np.pad(self.mat_img,((3,3),(3,3),(0,0)),'constant',constant_values = (0,0))
        second = np.pad(image_mid,((3,3),(3,3),(0,0)),'constant',constant_values = (0,0))
        third = np.pad(result_show,((3,3),(3,3),(0,0)),'constant',constant_values = (0,0))
        fourth = np.pad(result_show_1,((3,3),(3,3),(0,0)),'constant',constant_values = (0,0))

        final = np.concatenate((first,second,third,fourth),axis = 1)
        final3 = np.concatenate((first,second,third),axis = 1)


        cv2.imwrite("./full_pic/"+self.realname+"_full4pic.png",final)
        cv2.imwrite("./full_pic/"+self.realname+"_full3pic.png",final3)


        qim = QImage(result_final.data, result_final.shape[1], result_final.shape[0], result_final.strides[0], QImage.Format_RGB888)
        self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim))
        self.realname = self.realname[:-7]
        print(self.realname)



    def make_noise(self):
        noise = np.zeros([512, 320, 1],dtype=np.uint8)
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise/255,dtype=np.uint8)
        #noise = np.expand_dims(noise,axis=0)
        return noise

    def make_mask(self, pts):
        if len(pts)>0:
            mask = np.zeros((512,320,3))
            for pt in pts:
                cv2.line(mask,pt['prev'],pt['curr'],(255,255,255),6)
            mask_3 = np.asarray(mask[:,:,0]/255,dtype=np.uint8)
            mask_3 = np.expand_dims(mask_3,axis=2)
            self.mask_3 = mask_3
            #mask = np.expand_dims(mask,axis=0)
        else:
            mask = np.zeros((512,320,3))
            #mask = np.asarray(mask[:,:,0]/255,dtype=np.uint8)
            #mask = np.expand_dims(mask,axis=2)
            #mask = np.expand_dims(mask,axis=0)
        return mask

    def make_sketch(self, pts):
        if len(pts)>0:
            sketch = np.zeros((512,320,3))
            # sketch = 255*sketch
            for pt in pts:
                cv2.line(sketch,pt['prev'],pt['curr'],(255,255,255),1)
            #sketch = np.asarray(sketch[:,:,0]/255,dtype=np.uint8)
            #sketch = np.expand_dims(sketch,axis=2)
            #sketch = np.expand_dims(sketch,axis=0)
        else:
            sketch = np.zeros((512,320,3))
            # sketch = 255*sketch
            #sketch = np.asarray(sketch[:,:,0]/255,dtype=np.uint8)
            #sketch = np.expand_dims(sketch,axis=2)
            #sketch = np.expand_dims(sketch,axis=0)
        return sketch

    def sketch_more(self, pts):
        if len(pts)>0:
            sketch_more = np.zeros((512,320,3))
            for pt in pts:
                cv2.line(sketch_more,pt['prev'],pt['curr'],(255,255,255),1)

        else:
            sketch_more = np.zeros((512,320,3))

        return sketch_more

    def make_stroke(self, pts):
        if len(pts)>0:
            stroke = np.zeros((512,320,3))
            for pt in pts:
                c = pt['color'].lstrip('#')
                color = tuple(int(c[i:i+2], 16) for i in (0, 2 ,4))
                color = (color[2],color[1],color[0])
                cv2.line(stroke,pt['prev'],pt['curr'],color,4)
            #stroke = stroke/127.5 - 1
            #stroke = np.expand_dims(stroke,axis=0)
        else:
            stroke = np.zeros((512,320,3))
            #stroke = stroke/127.5 - 1
            #stroke = np.expand_dims(stroke,axis=0)
        return stroke

    def stroke_more(self, pts):
        if len(pts)>0:
            stroke_more = np.zeros((512,320,3))
            for pt in pts:
                c = pt['color'].lstrip('#')
                color = tuple(int(c[i:i+2], 16) for i in (0, 2 ,4))
                color = (color[2],color[1],color[0])
                cv2.line(stroke_more,pt['prev'],pt['curr'],color,4)
            #stroke = stroke/127.5 - 1
            #stroke = np.expand_dims(stroke,axis=0)
        else:
            stroke_more = np.zeros((512,320,3))
            #stroke = stroke/127.5 - 1
            #stroke = np.expand_dims(stroke,axis=0)
        return stroke_more


    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                    QDir.currentPath())
            cv2.imwrite(fileName+'.jpg',self.output_img)

    def rectangle_roi(self,event, x, y,flags, param):


        # Draw Rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.edge_change = self.img2.copy()
                cv2.rectangle(self.edge_change, (self.ix, self.iy), (x, y), RED, thickness)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.rect_or_mask = 0

        elif event == cv2.EVENT_LBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv2.rectangle(self.edge_change, (self.ix, self.iy), (x, y), RED, thickness)
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.rect_or_mask = 0
            print(" Now press the key 'n' a few times until no further change \n")

    def ROI(self):  # region of interest
        filename, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath())

        if filename:
            self.edge_change = cv2.imread(filename)
            self.img2 = self.edge_change.copy()
            cv2.namedWindow('input')
            cv2.setMouseCallback('input', self.rectangle_roi)

            print(" Draw a rectangle around the object using right mouse button \n")
            while (1):
                cv2.imshow('input', self.edge_change)
                k = cv2.waitKey(1)
                if k == 27:  # esc
                    break
                elif k == ord('r'):
                    print("resetting \n")
                    self.rect = (0, 0, 1, 1)
                    self.drawing = False
                    self.rectangle = False
                    self.rect_or_mask = 100
                    self.rect_over = False
                    self.edge_change = self.img2.copy()
                elif k == ord('n'):
                    if (self.rect_or_mask == 0):
                        x1, y1, x2, y2 = self.rect
                        #print(self.rect)
                        self.dst = self.edge_change[(y1 + 2):(y1 + y2 - 2), (x1 + 2):(x1 + x2 - 2)]
                        cv2.imshow("dst", self.dst)
                        cv2.imwrite("./check/ROI.png", self.dst)
                        dst_2 = (self.dst < 100).astype(np.uint8) * 255
                        cv2.imwrite("./check/show_ROI.png", dst_2)



                elif k == ord("s"):
                    x1, y1, x2, y2 =self.rect
                    #self.dst = self.edge_change[(y1 + 2):(y1 + y2 - 2), (x1 + 2):(x1 + x2 - 2)]
                elif k == ord("q"):
                    break

            cv2.destroyAllWindows()


    def paste(self):

        mask = self.make_mask(self.scene.mask_points)
        cv2.imwrite("./check/mask_draw.png", mask)
        incompleted_image = self.mat_img * (1+mask)
        cv2.imwrite("./check/show_incom.png", incompleted_image)
        ex1 = Main()
        ex1.show()

    def convert(self):

        global extraedge
        ROI = cv2.imread("./check/show_ROI.png")
        imgh = ROI.shape[0]
        imgw = ROI.shape[1]
        base_img = Image.open('./check/show_incom.png')

        mask_final =Image.open('./check/mask_draw.png')



        cv2.imwrite("./check/edge_masked.png",self.edge_numpy)
        edge_img_2 = Image.open('./check/edge_masked.png')

        self.a = np.zeros((imgh, imgw, 3))
        self.a = (self.a == 0).astype(np.uint8)*255
        cv2.imwrite("./check/mask_more.png", self.a)

        box = (posx, posy, posx + imgw, posy + imgh)
        tmp_img = Image.open('./check/show_ROI.png')
        tmp_img_2 = Image.open('./check/ROI.png')
        tmp_img_3 = Image.open('./check/mask_more.png')
        region = tmp_img
        region_2 = tmp_img_2
        region_3 = tmp_img_3
        base_img.paste(region, box)
        edge_img_2.paste(region_2,box)
        mask_final.paste(region_3,box)


        base_img.save('./check/out.png')
        edge_img_2.save('./check/out_edge.png')
        mask_final.save('./model_input/'+ self.realname +'_mask.png')
        masked_img =QPixmap('./check/out.png')
        extraedge = True

        self.scene_stroke.addPixmap(masked_img)


    def undo(self):
        self.scene.undo()
        self.scene_stroke.undo()

    def clear(self):
        global extraedge
        self.scene.reset_items()
        self.scene.reset()
        self.scene_stroke.reset_items()
        self.scene_stroke.reset()
        extraedge = False
        if type(self.image):
            self.scene.addPixmap(self.image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # opt = TestOptions().parse()
    # opt.dataset_mode = 'fashionE'
    # opt.netG = 'parsing'
    # opt.norm_G = 'batch'
    # opt.stage = 1
    # opt.input_nc = 26
    # opt.output_nc = 20
    # opt.name = 'fashionE_stageI_noflip_batchNorm_L110_Feat10_Parsing10_gan1_batch20_20190520'
    # print(opt.dataset_mode)
    # opt.which_epoch = 20
    # model1 = StageI_Parsing_Model(opt)
    # model1.eval()
    #
    # opt2 = TestOptions().parse()
    # opt2.dataset_mode = 'fashionE'
    # opt2.output_nc = 3
    # opt2.input_nc = 29
    # opt2.stage = 25
    # opt2.netG = 'multiatt3'
    # opt2.norm_G = 'batch'
    # opt2.ngf = 64
    # opt2.input_ANLs_nc = 5
    # opt2.which_epoch = 20
    # opt2.name = 'fasionE_stageII_matt3_noFeat_noL1_tv10_style200_20190520'
    # model2 = StageII_MultiAtt3_Model(opt2)
    model1 = 1
    model2 = 2
    opt = 1
    opt2 = 2
    ex = Ex(model1, opt, model2, opt2)

    sys.exit(app.exec_())
