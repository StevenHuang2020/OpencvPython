#python3 steven
import cv2
import argparse
import numpy as np

#----------------------------------------------
#usgae: python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights ./darknet-master/yolov3.weights  --classes yolov3.txt -s test.jpg
#----------------------------------------------


def cmd_line():
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                    help = 'path to input image')
    ap.add_argument('-c', '--config', required=True,
                    help = 'path to yolo config file')
    ap.add_argument('-w', '--weights', required=True,
                    help = 'path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', required=True,
                    help = 'path to text file containing class names')
    ap.add_argument('-s', '--save', required=True,
                    help = 'save the detection image')
    
    return ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    label = str(classes[class_id])
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    print(color)
    #text = label
    percentage = "{0:.0%}".format(confidence)
    text = '%s[%s]'%(label, percentage)

    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX #cv2.FONT_HERSHEY_PLAIN  # 
    fontColor = (0, 0, 0)
    '''
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    text_offset_x = x
    text_offset_y = y-10
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width, text_offset_y + text_height))
    cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
    '''
    
    cv2.putText(img, text, (x, y-10), font, font_scale, fontColor, 2)

def main():
    args = cmd_line()
    image = cv2.imread(args.image)

    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    classes = None
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet(args.weights, args.config)
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes)

    cv2.imshow("object detection", image)
    cv2.waitKey()
        
    if args.save:
        print(args.save)
        cv2.imwrite(args.save, image)
    else:
        cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
