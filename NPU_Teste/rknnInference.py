import os, cv2, time, numpy as np
from utils import *
from rknnlite.api import RKNNLite

# Definição de constantes e caminhos para configuração do modelo e entrada/saída
conf_thres = 0.25
iou_thres = 1
input_width = 640
input_height = 480
result_path = "./result"
image_path = "./dataset/1.jpg"
video_path = "/home/orangepi/AMII.mp4"
camera_inference = True
video_inference = False
RKNN_MODEL = f'last-480-640.rknn'
CLASSES = ['Carga']
#CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def best(detections):
    # Inicializando a maior pontuação de confiança e a detecção correspondente
    max_confidence = 0
    best_detection = None
    for det in detections:
        # Pontuação de confiança está no índice 4
        confidence = det[4]
        if confidence > max_confidence:
            max_confidence = confidence
            best_detection = det
        boxes, masks, shape = results
        # Processar apenas a detecção com maior confiança
        if best_detection is not None:
            best_boxes = [best_detection]
            # Encontrar o índice da melhor detecção em 'boxes'
            best_mask_index = None
            for i, box in enumerate(boxes):
                if np.array_equal(box, best_detection):
                    best_mask_index = i
                    break

            # Verificar se encontramos a melhor máscara
            if best_mask_index is not None and masks is not None:
                best_masks = [masks[best_mask_index]]
            else:
                best_masks = [np.zeros_like(image_3c)]

            # Convertendo best_masks para um array NumPy
            best_masks = np.array(best_masks)
            if type(best_masks) != list and best_masks.ndim == 3:
                return vis_result(image_3c, [best_boxes, best_masks, shape], colorlist, CLASSES, result_path)
            else:
                return None

# Início do script principal
if __name__ == '__main__':
    # Verificar se o caminho do resultado existe, senão cria
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)
    
    # Inicializar o RKNN Lite e carregar o modelo
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    ret = rknn_lite.init_runtime()
    colorlist = gen_color(len(CLASSES))
    # Processamento para vídeo
    if camera_inference == True:
        # Inicializa a captura de vídeo com o índice da câmera
        cap = cv2.VideoCapture(0)

        while(True):
            ret, image_3c = cap.read()
            if not ret:
                break
            # Preprocessamento e inferência no frame capturado
            image_4c, image_3c = preprocess(image_3c, input_height, input_width)
            start = time.time()
            outputs = rknn_lite.inference(inputs=[image_3c])
            stop = time.time()
            fps = round(1/(stop-start), 2)
            print("FPS: ", fps)
            tempo = stop - start
            print("Tempo: ", tempo)
            outputs[0] = np.squeeze(outputs[0])
            outputs[0] = np.expand_dims(outputs[0], axis=0)
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES))
            results = results[0]
            detections = results[0]
        
            # Chamando vis_result com apenas a melhor detecção
            if best(detections=detections) is not None:
                mask_img, vis_img = best(detections=detections)
                # Exibe a imagem com a máscara
                cv2.imshow("Vis Image with Mask", vis_img)  

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libera a captura e fecha todas as janelas do OpenCV
        cap.release()
        cv2.destroyAllWindows()
    elif video_inference == True:
        cap = cv2.VideoCapture(video_path)
        while(True):
            ret, image_3c = cap.read()
            if not ret:
                break
            # Processamento de cada frame do vídeo
            print('--> Running model for video inference')
            image_4c, image_3c = preprocess(image_3c, input_height, input_width)
            ret = rknn_lite.init_runtime()
            start = time.time()
            outputs = rknn_lite.inference(inputs=[image_3c])

            # Pós-processamento e visualização dos resultados
            stop = time.time()
            fps = round(1/(stop-start), 2)
            outputs[0]=np.squeeze(outputs[0])
            outputs[0] = np.expand_dims(outputs[0], axis=0)
            colorlist = gen_color(len(CLASSES))
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
            results = results[0]              ## batch=1
            boxes, masks, shape = results
            if type(masks) != list and masks.ndim == 3:
                mask_img, vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
                cv2.imshow("mask_img", mask_img)
                cv2.imshow("vis_img", vis_img)
            else:
                print("No inference")
            cv2.waitKey(10)
    else:
        image_3c = cv2.imread(image_path)
        image_4c, image_3c = preprocess(image_3c, input_height, input_width)
        ret = rknn_lite.init_runtime()
        start = time.time()
        outputs = rknn_lite.inference(inputs=[image_3c])
        stop = time.time()
        fps = round(1/(stop-start), 2)
        outputs[0]=np.squeeze(outputs[0])
        outputs[0] = np.expand_dims(outputs[0], axis=0)
        colorlist = gen_color(len(CLASSES))
        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
        results = results[0]              ## batch=1
        detections = results[0]

        # Chamando vis_result com apenas a melhor detecção
        if best(detections=detections) is not None:
            mask_img, vis_img = best(detections=detections)
            # Exibe a imagem com a máscara
            cv2.imshow("Vis Image with Mask", vis_img)  
        else:
            print("No inference")
    print("RKNN inference finish")
    rknn_lite.release()
    cv2.destroyAllWindows()
