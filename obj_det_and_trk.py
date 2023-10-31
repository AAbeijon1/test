import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, 
                            check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                            scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from datetime import datetime
import tensorflow as tf



ruta_completa = "C:/Users/AAbeijon/Desktop/yolov5-object-tracking/detections.txt"  # Cambia esto a la ruta donde deseas guardar el archivo
fechaloger = datetime.now()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

#---------------Object Tracking---------------
import skimage
from sort import *


#-----------Object Blurring-------------------
blurratio = 40


#.................. Tracker Functions .................
'''Computer Color for every box and track'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, 
                names=None, color_box=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id)

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, color,-1)
        else:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),(255,191,0), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, (255,191,0),-1)
    return img
#..............................................................................


@torch.no_grad()
def detect(weights=ROOT / 'yolov5n.pt',
           crop_save_path='./crops',
        source=ROOT / 'yolov5/data/images', 
        data=ROOT / 'yolov5/data/coco128.yaml',  
        imgsz=(640, 640),conf_thres=0.25,iou_thres=0.45,  
        max_det=1000, device='cpu',  view_img=False,  
        save_txt=False, save_conf=False, save_crop=True, 
        nosave=False, classes=None,  agnostic_nms=False,  
        augment=False, visualize=False,  update=False,  
        project=ROOT / 'runs/detect',  name='exp',  
        exist_ok=False, line_thickness=2,hide_labels=False,  
        hide_conf=False,half=False,dnn=False,display_labels=False,
        blur_obj=False,color_box = False,):
    global saved_ids
    save_img = not nosave and not source.endswith('.txt')
    dets_to_sort = np.empty((0,6))
    print ("inicio") 
    try:
        with open("C:/Users/AAbeijon/Desktop/yolov5-object-tracking/last_id.txt", "r") as f:
            content = f.read().strip()
            last_id = int(float(content))
    except (FileNotFoundError, ValueError):
        print("Error al leer el archivo last_id.txt")
        last_id = 0

    #.... Initialize SORT .... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) 
    track_color_id = 0
    #......................... 
    
    
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  

    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  
    if pt or jit:
        model.model.half() if half else model.model.float()

    if webcam:
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset) 
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1 
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    t0 = time.time()
    
    dt, seen = [0.0, 0.0, 0.0], 0
    
    for path, im, im0s, vid_cap, s in dataset:
        id_save_count = {}
        fechaloger = datetime.now()
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        saved_ids = set()
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results

                
                #class_name = names[int(cls)]
                current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
                clases_deseadas = ["person", "bicycle", "car", "motorcycle", "bus"]
                tracked_dets = []

                # Luego, dentro del bucle de detección, actualiza tracked_dets con el resultado de sort_tracker.update():
                tracked_dets = sort_tracker.update(dets_to_sort)
                for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    identities = None
                    id = int(tracked_dets[i][8]) if i < len(tracked_dets) else 0  # Asegurarse de que no se exceda el índice

                    if cls is not None:
                        class_name = names[int(cls)]
                    if save_crop and class_name in clases_deseadas:
                        crop_obj = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        class_folder_path = os.path.join(opt.crop_save_path, class_name)
                        if not os.path.exists(class_folder_path):
                            os.makedirs(class_folder_path)
                        
                        # Usa el ID en el nombre del archivo
                        crop_save_path = os.path.join(class_folder_path, f"{id}.jpg")
                        cv2.imwrite(crop_save_path, crop_obj)



                    if blur_obj:
                        crop_obj = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                        blur = cv2.blur(crop_obj,(blurratio,blurratio))
                        im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur
                    else:
                        continue
                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                              np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))
                 
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()
                

                #loop over tracks
                for track in tracks:
                    if color_box:
                        color = compute_color_for_labels(track_color_id)
                        [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                                (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                                color, thickness=3) for i,_ in  enumerate(track.centroidarr) 
                                if i < len(track.centroidarr)-1 ] 
                        track_color_id = track_color_id+1
                    else:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                                (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                                (124, 252, 0), thickness=3) for i,_ in  enumerate(track.centroidarr) 
                                if i < len(track.centroidarr)-1 ] 
                # Extract the highest ID from tracked_dets
                max_id_in_frame = max([d[8] for d in tracked_dets]) if len(tracked_dets) > 0 else 0
                last_id = max(last_id, max_id_in_frame)
                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names,color_box)
                        
                # Clases que deseas incluir


# Define una función para redondear el color al color principal más cercano


                #def map_color_to_name(rgb):

                # Añade una función para convertir RGB a HSV
                def rgb_to_hsv(rgb):
                    color = np.uint8([[list(rgb)]])
                    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
                    return tuple(hsv_color[0][0])

                def map_color_to_name(rgb):
                    if rgb is None:
                        return "Desconocido"  # O cualquier otro valor que desees para manejar los casos de ROI vacía o errores
                    else:
                    # Convertimos el color de entrada a HSV
                        h, s, v = rgb_to_hsv(rgb)
                    
                    # Definimos rangos de colores en HSV (similar al código C++)
                    color_ranges = {
                        "Azul": [(0, 50, 50), (10, 255, 255)],
                        "Verde": [(35, 50, 50), (85, 255, 255)],
                        "Amarillo": [(15, 50, 50), (35, 255, 255)],
                        "Rojo": [(0, 200, 0), (19, 255, 255)],
                        "Rojo": [(170, 50, 50), (180, 255, 255)],
                        "Blanco": [(0, 0, 200), (180, 30, 255)],  # H no importa, S baja y V alta
                        "Negro": [(0, 0, 0), (180, 255, 30)]    # H y S no importan, V bajo
                    }

                    # Determinamos a qué rango de color pertenece
                    for color_name, (hsv_min, hsv_max) in color_ranges.items():
                        h_min, s_min, v_min = hsv_min
                        h_max, s_max, v_max = hsv_max
                        if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
                            return color_name
                    
                    return "Desconocido"  # si no cae en ninguno de los rangos definidos

                # Definir tus clases deseadas
                clases_deseadas = ["person", "bicycle", "car", "motorcycle", "bus"]

                # Función para obtener el color predominante de una ROI
                def get_dominant_color(image, bbox, k=2):
                    x1, y1, x2, y2 = map(int, bbox)
                    roi = image[y1:y2, x1:x2]

                    try:
                        if roi.size > 0:  # Verifica si la ROI no está vacía
                            roi = roi.reshape((roi.shape[0] * roi.shape[1], 3))

                            # Aplicar k-means
                            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
                            _, labels, centers = cv2.kmeans(roi.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                            # Encontrar el cluster más prominente (basado en la frecuencia)
                            _, counts = np.unique(labels, return_counts=True)
                            dominant_color = centers[np.argmax(counts)].astype(int)

                            return tuple(dominant_color)
                        else:
                            #print("ROI vacía, se descartará esta ROI.")
                            return None
                    except cv2.error:
                        # Manejar el error de k-means aquí (por ejemplo, imprimir un mensaje o devolver None)
                        #print("Error de k-means en ROI, se descartará esta ROI.")
                        return None

                # Recopilar detecciones, sus IDs y colores
                # Recopilar detecciones, sus IDs y colores
                detections_with_ids_colors = []
                for i in range(len(names)):
                    if names[i] in clases_deseadas:
                        # Obtener detecciones de esta clase
                        class_detections = [d for d in tracked_dets if d[4] == i]
                        
                        # Obtener IDs y nombres de colores de estas detecciones
                        ids_color_names = [(int(d[8]), map_color_to_name(get_dominant_color(im0, d[:4]))) for d in class_detections]
                        
                        # Construir la cadena de salida para esta clase
                        class_string = f"{len(class_detections)}{names[i]} ("
                        if ids_color_names:
                            class_string += " - ID -"
                            for id, color in ids_color_names:
                                class_string += f" {id} {color} -"
                        class_string += " )"
                        
                        detections_with_ids_colors.append(class_string)

                # Formar la cadena final
                if detections_with_ids_colors:
                    LOGGER.info(f"ZONAX, CAMX, {', '.join(detections_with_ids_colors)}, {fechaloger.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]}")
                else:
                    LOGGER.info("No detections")

                contenido = f"ZONAX, CAMX, {', '.join(detections_with_ids_colors)}, {fechaloger.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]}\n"
                with open(ruta_completa, "a") as archivo:
                    archivo.write(contenido)



                    

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1) 
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path: 
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  
                        if vid_cap: 
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        print("Frame Processing!")
        print(f"Guardando last_id: {last_id}")
        with open("C:/Users/AAbeijon/Desktop/yolov5-object-tracking/last_id.txt", "w") as f:
            f.write(str(last_id))

    print("Video Exported Success")

    if update:
        strip_optimizer(weights)
    
    if vid_cap:
        vid_cap.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop-save-path', type=str, default='./crops', help='path to save cropped images')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--blur-obj', action='store_true', help='Blur Detected Objects')
    parser.add_argument('--color-box', action='store_true', help='Change color of every box and track')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)